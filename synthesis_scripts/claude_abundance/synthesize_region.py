#!/usr/bin/env python3
"""
One synthesis of a free wavelength range with chosen inputs, drawn over the observed spectrum. Nothing is fitted.
Meant for molecular bands (CH G band, C2 Swan, CN) and A(C), 12C/13C, but works for any element and range.

Run it as a plain script: edit the settings below, then `python synthesize_region.py`.

The stellar parameters and broadening come from `synthesis:` of the star in config/stars.yaml (teff, logg, feoh, vmic,
vmac, vmac_profile), as in find_element_abundance.py; a setting here overrides them. The other elements come from
config/reference_abundance.yaml and ABUNDANCE_OVERRIDES, Fe follows FEH (A(Fe) = 7.45 + FEH). Only ELEMENT is set
explicitly (A_X or X_FE). Everything is LTE unless NLTE = "T" and ELEMENT has a departure-coefficient grid (there is none
for C, N, O: molecular bands are LTE anyway).

Drawn (each one bsyn run, in parallel):
    nominal          A(ELEMENT), 12C/13C = C12_C13
    band             A(ELEMENT) -/+ BAND_OFFSET, shaded               (BAND_OFFSET = None or 0: off)
    without          ELEMENT removed (A = -20): where it absorbs      (SHOW_WITHOUT)
    extra ratios     the nominal A at other 12C/13C                   (EXTRA_RATIOS)
    extra abundances the nominal model at other A(ELEMENT)            (EXTRA_ABUNDANCES)
babsma runs at the nominal A(ELEMENT) for all of them (continuous opacity and equation of state). The residual panel shows
observed - model; chi^2_red and rms of each model over RANGE minus EXCLUDE are printed and written to summary.yaml, to
compare models by number, not only by eye. chi^2 uses sigma = 1/SNR of stars.yaml.

Output: runs/<star>/synthesis/<time>_<ELEMENT>_<lo>-<hi>/
    region.png      observed, models, band, residuals (the range split into rows of PANEL_WIDTH)
    synth.csv       observed and every model on the observed pixels (after RV shift, broadening, CONT_SCALE)
    summary.yaml    inputs, abundances given to Turbospectrum, chi^2_red / rms per model
    atm/            the interpolated atmosphere; copies of the scripts used
"""

from __future__ import annotations

import dataclasses
import math
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))     # find fe_line_fit.py next to this script
from fe_line_fit import (C_KMS, CONFIG_DIR, FALTBON_PROFILE, RUNS_DIR, SCRATCH_DIR, LineSynth, individual_abundances,
                         load_spectrum, parse_atm_name, resolve_path, trim_linelists)
from element_line_fit import SOLAR_TS, load_element_config, make_atmosphere, start_abundance

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

SELECTED_STAR = "HD196944"
RANGE = (4295.0, 4330.0)      # synthesised and plotted range [A], rest frame of the star
ELEMENT = "C"                 # the element set by A_X / X_FE (and removed in the "without" model)

# A(ELEMENT): A_X (absolute, log eps with H = 12), else X_FE ([X/Fe] on Turbospectrum's solar scale with FEH as [Fe/H]),
# else the reference file / ABUNDANCE_OVERRIDES, else scaled solar.
A_X = None
X_FE = None
C12_C13 = None                # None: the reference file (isotope_ratios: C12_C13), else Turbospectrum's solar mix (~89)
MOLECULES = None              # None: `molecular_line_lists` of stars.yaml; else groups of line_lists.yaml, e.g. ["CH"]
                              # or ["CH", "CC", "CN"]. [] = no molecules at all
ABUNDANCE_OVERRIDES = {       # other elements, A(X); beats the reference file. O sets how much C is locked in CO, N the CN
    # "O": 7.43,
    # "N": 7.99,
}
USE_REFERENCE_ABUNDANCES = True    # config/reference_abundance.yaml for SELECTED_STAR
REFERENCE_SCALING = "absolute"     # "absolute", "fe" or "xfe", see find_stellar_parameters

# None = stars.yaml (`synthesis:` teff, logg, vmic, feoh, vmac, vmac_profile); a value here overrides it.
TEFF = None
LOGG = None
VMIC = None
FEH = None
VMAC = None                   # extra broadening, faltbon's argument in km/s (gauss: FWHM; radtan: zeta_RT / 1.433)
VMAC_PROFILE = None           # "gauss" or "radtan"
MASS = None                   # None: from `model_atmosphere` in stars.yaml (1.0 spherical, 0.0 plane-parallel)
NLTE = "F"                    # "T": ELEMENT in NLTE (needs its grid in data/original/NLTE/dc/), the rest LTE

RV_SHIFT = 0.0                # km/s, on top of `rv` in stars.yaml (the Fe fits give the residual shift: dv_kms)
CONT_SCALE = 1.0              # every model is multiplied by this (a continuum placement check; 1 = trust the normalisation)

BAND_OFFSET = 0.2             # dex; None or 0: no band
SHOW_WITHOUT = True           # also the model without ELEMENT
EXTRA_RATIOS = []             # e.g. [4, 20]: more 12C/13C at the nominal A(ELEMENT)
EXTRA_ABUNDANCES = []         # e.g. [7.0, 7.8]: more A(ELEMENT) at the nominal 12C/13C
EXCLUDE = []                  # [(lo, hi), ...] left out of chi^2 / rms (shaded in the plot), e.g. a bad pixel or Mg b

N_JOBS = 8                    # parallel syntheses (each runs babsma + bsyn over the whole range)
SYNTH_PAD = 2.0               # synthesis extends this far beyond RANGE [A] (RV shift, broadening kernel)
PANEL_WIDTH = 12.0            # A per row of region.png
YLIM = (None, 1.05)           # flux axis of the spectrum panels (None: automatic)
RESID_YLIM = 0.10             # residual panels: +/- this

WITHOUT = -20.0               # abundance that removes ELEMENT's lines and molecules


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def synth_one(job: dict) -> tuple[np.ndarray, np.ndarray]:
    """Worker: one model -> (wavelength, flux) after the instrumental and extra broadening."""
    workdir = Path(tempfile.mkdtemp(prefix="reg_", dir=SCRATCH_DIR))
    try:
        cfg = dataclasses.replace(job["cfg"], c12_c13=job["ratio"])
        synth = LineSynth(cfg, job["atm"], job["wmin"], job["wmax"], workdir, workdir / "log.txt")
        wl, fl = synth.broadened(job["abu"], job["nlte"], job["broad"], job["profile"])
        if len(wl) < 2:
            raise RuntimeError(f"empty synthetic spectrum for {job['key']}")
        return wl, fl
    except Exception as exc:                              # keep the log tail: Turbospectrum errors are in it
        log = workdir / "log.txt"
        tail = "".join(log.read_text(errors="replace").splitlines(True)[-40:]) if log.exists() else ""
        raise RuntimeError(f"{job['key']}: {exc}\n--- Turbospectrum log (tail) ---\n{tail}") from None
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) index pairs of the True runs of a boolean array."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def plot_region(path: Path, wl: np.ndarray, obs: np.ndarray, curves: list[dict], band: np.ndarray | None,
                band_label: str, used: np.ndarray, title: str) -> None:
    """curves[0] is the nominal model (bold residuals); the others get thin residuals unless resid=False."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = RANGE
    n = max(1, math.ceil((hi - lo) / PANEL_WIDTH - 1e-9))
    edges = np.linspace(lo, hi, n + 1)
    fig, axes = plt.subplots(2 * n, 1, figsize=(20, 4.6 * n), squeeze=False,
                             gridspec_kw=dict(height_ratios=[3, 1.1] * n, hspace=0.08))
    axes = axes[:, 0]
    for k in range(n):
        ax, rax = axes[2 * k], axes[2 * k + 1]
        a, b = edges[k], edges[k + 1]
        sel = (wl >= a - 0.5) & (wl <= b + 0.5)
        for i0, i1 in _runs(~used & sel):
            for axx in (ax, rax):
                axx.axvspan(wl[i0], wl[i1 - 1], color="tab:red", alpha=0.08, lw=0)
        if band is not None:
            ax.fill_between(wl[sel], band[0][sel], band[1][sel], color="tab:blue", alpha=0.25, lw=0,
                            label=band_label if k == 0 else None)
        ax.plot(wl[sel], obs[sel], "-", color="0.45", lw=0.9, label="observed" if k == 0 else None)
        rax.axhline(0, color="0.3", lw=0.8)
        for j, c in enumerate(curves):
            ax.plot(wl[sel], c["flux"][sel], color=c["colour"], ls=c.get("ls", "-"), lw=c.get("lw", 1.0),
                    label=c["label"] if k == 0 else None, zorder=3 if j == 0 else 2)
            if c.get("resid", True):
                rax.plot(wl[sel], (obs - c["flux"])[sel], color=c["colour"], ls=c.get("ls", "-"),
                         lw=0.9 if j == 0 else 0.6, alpha=1.0 if j == 0 else 0.7)
        for axx in (ax, rax):
            axx.set_xlim(a, b)
            axx.grid(alpha=0.3)
        ax.set_ylim(*YLIM)
        ax.set_ylabel("normalised flux")
        ax.tick_params(labelbottom=False)
        rax.set_ylim(-RESID_YLIM, RESID_YLIM)
        rax.set_ylabel("obs − model")
        if k == 0:
            ax.set_title(title, loc="left", fontsize=10)
            ax.legend(loc="lower right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("wavelength [Å]")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def stellar_parameters(cfg) -> dict:
    """TEFF, LOGG, VMIC, FEH, VMAC, VMAC_PROFILE: the setting here if given, else `synthesis:` in stars.yaml."""
    syn = cfg.raw["synthesis"]
    wanted = {"TEFF": ("teff", TEFF), "LOGG": ("logg", LOGG), "VMIC": ("vmic", VMIC), "FEH": ("feoh", FEH),
              "VMAC": ("vmac", VMAC), "VMAC_PROFILE": ("vmac_profile", VMAC_PROFILE)}
    found, missing = {}, []
    for name, (key, override) in wanted.items():
        value = syn.get(key) if override is None else override
        if value is None:
            missing.append(f"{key} ({name})")
        else:
            found[name] = value if name == "VMAC_PROFILE" else float(value)
    if missing:
        raise SystemExit(f"{cfg.star}: no value for {', '.join(missing)}: add it under `synthesis:` in config/stars.yaml, "
                         "or set it at the top of this script")
    if found["VMAC_PROFILE"] not in FALTBON_PROFILE:
        raise SystemExit(f"vmac_profile must be one of {list(FALTBON_PROFILE)}, not {found['VMAC_PROFILE']!r}")
    return found


def main() -> None:
    lo, hi = map(float, RANGE)
    if hi <= lo:
        raise SystemExit(f"RANGE must be (lo, hi) with lo < hi, not {RANGE}")
    cfg = load_element_config(SELECTED_STAR, ELEMENT, nlte=NLTE == "T")
    if cfg.spectrum_file is None:
        raise SystemExit(f"{SELECTED_STAR}: no normalized spectrum listed in stars.yaml")
    p = stellar_parameters(cfg)
    teff, logg, vmic, feh, broad, profile = p["TEFF"], p["LOGG"], p["VMIC"], p["FEH"], p["VMAC"], p["VMAC_PROFILE"]
    mass = MASS if MASS is not None else (parse_atm_name(cfg.yaml_atm.name).get("mass", 1.0) if cfg.yaml_atm else 1.0)

    # line lists: master + the chosen molecules
    if MOLECULES is not None:
        ll_cfg = yaml.safe_load((CONFIG_DIR / "line_lists.yaml").read_text())
        unknown = [m for m in MOLECULES if m not in ll_cfg["molecular"]]
        if unknown:
            raise SystemExit(f"MOLECULES {unknown} not in line_lists.yaml (have {list(ll_cfg['molecular'])})")
        cfg.linelists = ([resolve_path(x) for x in ll_cfg["master"]]
                         + [resolve_path(x) for m in MOLECULES for x in ll_cfg["molecular"][m]])
    molecules = list(MOLECULES) if MOLECULES is not None else list(cfg.raw["synthesis"].get("molecular_line_lists") or [])

    # abundances
    if not USE_REFERENCE_ABUNDANCES:
        cfg.reference_abundances, cfg.reference_notes, cfg.relative_abundances = {}, {}, {}
        cfg.c12_c13 = None
    cfg.reference_scaling = REFERENCE_SCALING
    cfg.abundance_overrides = dict(ABUNDANCE_OVERRIDES)
    if C12_C13 is not None:
        if C12_C13 <= 0:
            raise SystemExit(f"C12_C13 must be positive, not {C12_C13}")
        cfg.c12_c13 = float(C12_C13)
    if A_X is not None:
        a0 = float(A_X)
    elif X_FE is not None:
        a0 = SOLAR_TS[ELEMENT] + feh + float(X_FE)
    else:
        a0 = start_abundance(cfg, feh)
    a0 = round(a0, 3)
    cfg.babsma_abundance = a0
    others = individual_abundances(cfg, feh)                 # everything but ELEMENT (Fe follows FEH)

    def xfe(a: float) -> float:
        return a - SOLAR_TS[ELEMENT] - feh

    mode = "NLTE" if NLTE == "T" else "LTE"
    tag = f"{time.strftime('%Y%m%d_%H%M%S')}_{ELEMENT}_{lo:.0f}-{hi:.0f}"
    outdir = RUNS_DIR / SELECTED_STAR / "synthesis" / tag
    outdir.mkdir(parents=True)
    here = Path(__file__)
    for f in (here, here.with_name("element_line_fit.py"), here.with_name("fe_line_fit.py")):
        shutil.copy2(f, outdir / f.name)

    ratio_txt = f"{cfg.c12_c13:g}" if cfg.c12_c13 is not None else "solar"
    print(f"{SELECTED_STAR}: Teff={teff:.0f} logg={logg:.2f} vmic={vmic:.2f} [Fe/H]={feh:+.2f}  {mode} {ELEMENT}  "
          f"broadening {broad:g} km/s ({profile})")
    print(f"  A({ELEMENT}) = {a0:.3f}  [{ELEMENT}/Fe] = {xfe(a0):+.2f};  12C/13C = {ratio_txt};  molecules {molecules}")
    print("  other abundances (the rest scaled solar): " + "  ".join(f"{el} {a:.2f}" for el, a in others.items()))
    print(f"-> {outdir}")

    print("Building atmosphere...", end="\r", flush=True)
    atm, _ = make_atmosphere(teff, logg, feh, ELEMENT, a0, mass, vmic, outdir / "atm", nlte=NLTE == "T")
    print("Building atmosphere...  DONE")

    # observed spectrum (rest frame: stars.yaml rv + RV_SHIFT)
    obs_wl, obs_fl = load_spectrum(cfg.spectrum_file, cfg.rv)
    obs_wl = obs_wl / (1.0 + RV_SHIFT / C_KMS)
    sel = (obs_wl >= lo) & (obs_wl <= hi)
    if sel.sum() < 10:
        raise SystemExit(f"only {sel.sum()} observed pixels in {lo}-{hi} A: check RANGE against the spectrum "
                         f"({obs_wl.min():.1f}-{obs_wl.max():.1f} A)")
    wl, fl = obs_wl[sel], obs_fl[sel]
    used = np.ones(len(wl), bool)
    for a, b in EXCLUDE:
        used &= ~((wl >= a) & (wl <= b))

    # the models
    off = float(BAND_OFFSET or 0.0)
    models = [dict(key="nominal", abu=a0, ratio=cfg.c12_c13)]
    if off:
        models += [dict(key="band-", abu=a0 - off, ratio=cfg.c12_c13), dict(key="band+", abu=a0 + off, ratio=cfg.c12_c13)]
    if SHOW_WITHOUT:
        models.append(dict(key="without", abu=WITHOUT, ratio=cfg.c12_c13))
    models += [dict(key=f"ratio {r:g}", abu=a0, ratio=float(r)) for r in EXTRA_RATIOS]
    models += [dict(key=f"A {a:.2f}", abu=float(a), ratio=cfg.c12_c13) for a in EXTRA_ABUNDANCES]

    t0 = time.time()
    trim_dir = Path(tempfile.mkdtemp(prefix="reg_ll_", dir=SCRATCH_DIR))
    try:
        cfg = dataclasses.replace(cfg, linelists=trim_linelists(cfg.linelists, lo - SYNTH_PAD - 10, hi + SYNTH_PAD + 10,
                                                                trim_dir))
        jobs = [dict(cfg=cfg, atm=atm, wmin=lo - SYNTH_PAD, wmax=hi + SYNTH_PAD, nlte=NLTE, broad=broad, profile=profile,
                     **m) for m in models]
        print(f"{len(jobs)} synthes{'is' if len(jobs) == 1 else 'es'} over {hi - lo + 2 * SYNTH_PAD:.1f} A ...")
        with ProcessPoolExecutor(max_workers=max(1, min(len(jobs), N_JOBS))) as ex:
            spectra = dict(zip([m["key"] for m in models], ex.map(synth_one, jobs)))
    finally:
        shutil.rmtree(trim_dir, ignore_errors=True)
    flux = {k: CONT_SCALE * np.interp(wl, *spectra[k]) for k in spectra}

    # chi^2 / rms per model
    stats = {}
    for k, f in flux.items():
        r = (fl - f)[used]
        stats[k] = dict(chi2_red=round(float(np.sum((r * cfg.snr) ** 2) / max(len(r), 1)), 3),
                        rms=round(float(np.sqrt(np.mean(r ** 2))), 5))
    for m in models:
        k = m["key"]
        print(f"  {k:12s} A({ELEMENT}) = {m['abu']:7.3f}  12C/13C = {m['ratio'] if m['ratio'] is not None else 'solar'!s:>6}"
              f"  chi2_red = {stats[k]['chi2_red']:8.2f}  rms = {stats[k]['rms']:.4f}")

    # plot
    curves = [dict(flux=flux["nominal"], colour="black", lw=1.2,
                   label=f"A({ELEMENT}) = {a0:.2f} ([{ELEMENT}/Fe] = {xfe(a0):+.2f}), ¹²C/¹³C = {ratio_txt}")]
    if SHOW_WITHOUT:
        curves.append(dict(flux=flux["without"], colour="tab:orange", lw=0.9, ls="--", label=f"without {ELEMENT}",
                           resid=False))
    colours = iter(["tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"] * 3)
    for r in EXTRA_RATIOS:
        curves.append(dict(flux=flux[f"ratio {r:g}"], colour=next(colours), lw=0.9, label=f"¹²C/¹³C = {r:g}"))
    for a in EXTRA_ABUNDANCES:
        curves.append(dict(flux=flux[f"A {a:.2f}"], colour=next(colours), lw=0.9, ls="-.",
                           label=f"A({ELEMENT}) = {a:.2f} ([{ELEMENT}/Fe] = {xfe(a):+.2f})"))
    band = np.vstack([flux["band-"], flux["band+"]]) if off else None
    title = (f"{SELECTED_STAR}  {lo:.1f}–{hi:.1f} Å   Teff={teff:.0f} logg={logg:.2f} vmic={vmic:.2f} [Fe/H]={feh:+.2f}  "
             f"{mode}   broad {broad:g} km/s {profile}   RV {RV_SHIFT:+.2f} km/s   cont ×{CONT_SCALE:g}   "
             f"χ²_red {stats['nominal']['chi2_red']:.2f}")
    plot_region(outdir / "region.png", wl, fl, curves, band, f"A({ELEMENT}) ∓ {off:g} dex", used, title)

    cols = {"wavelength": wl, "flux_obs": fl, "used": used.astype(int)}
    cols |= {f"flux_{k.replace(' ', '_')}": f for k, f in flux.items()}
    np.savetxt(outdir / "synth.csv", np.column_stack(list(cols.values())), delimiter=",", comments="",
               header=",".join(cols), fmt=["%.4f", "%.6f", "%d"] + ["%.6f"] * len(flux))
    summary = dict(
        star=SELECTED_STAR, range=[lo, hi], element=ELEMENT, nlte=NLTE == "T", teff=teff, logg=logg, vmic=vmic,
        feh_atmosphere=feh, mass=mass, atmosphere=atm.atm.name, broadening=broad, broad_profile=profile,
        rv_shift=RV_SHIFT, cont_scale=CONT_SCALE, A_nominal=a0, xfe_nominal=round(xfe(a0), 3),
        solar_abundance_scale="asp2007 (Turbospectrum default)", c12_c13=cfg.c12_c13, molecules=molecules,
        abundance_overrides=dict(ABUNDANCE_OVERRIDES), other_abundances=others, exclude=[list(x) for x in EXCLUDE],
        spectrum=str(cfg.spectrum_file), snr=cfg.snr, n_pixels=int(used.sum()),
        models={m["key"]: dict(A=m["abu"], c12_c13=m["ratio"], **stats[m["key"]]) for m in models},
        runtime_s=round(time.time() - t0),
    )
    (outdir / "summary.yaml").write_text(yaml.dump(summary, sort_keys=False))
    print(f"Plot: {outdir / 'region.png'}")


if __name__ == "__main__":
    main()
