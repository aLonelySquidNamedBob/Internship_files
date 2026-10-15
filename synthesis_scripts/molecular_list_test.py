#!/usr/bin/env python3
"""
Which molecular line lists matter for a star? One Turbospectrum synthesis without any molecular list (the
reference) and one per molecule group of config/line_lists.yaml (CC, CN, CH, ...: all isotopologue files of the
group together), all with the star's atmosphere and abundances. The difference with the reference is the
absorption the group adds. Groups whose absorption exceeds IMPACT_THRESHOLD are recommended for
`molecular_line_lists` in config/stars.yaml.

Run it as a plain script: edit the settings below, then `python molecular_list_test.py`.

Everything is taken from the config files:
    stars.yaml               atmosphere + departure coefficients, metallicity, alpha/Fe, vmic, abu_c, spectrum,
                             instrument_fwhm (default -3 km/s), vmac (rad-tan, optional)
    line_lists.yaml          the master lists (always included) and the molecular groups that are tested
    reference_abundance.yaml A(X) of the other elements (C, N, O drive the molecules) and isotope_ratios: C12_C13
Abundances are assembled exactly as in claude_abundance/fe_line_fit.py (reference file, REFERENCE_SCALING, abu_c,
ABUNDANCE_OVERRIDES); A(Fe) = 7.45 + metallicity (Turbospectrum / MARCS solar scale).

Output in runs/<star>/molecular_tests/<wmin>-<wmax>_<time>/ :
    summary.csv / summary.yaml   per group: deepest added absorption, absorption added (mA), A of spectrum
                                 affected, strongest wavelength, effect on the Fe line windows; recommendation
    spectra/<group>.cvl          the syntheses (after the instrumental profile)
    plots/<group>.png            added absorption over the whole range + zoom on the strongest feature
                                 (with the observed spectrum when there is one)
    plots/impact_overview.png    all groups side by side
"""

from __future__ import annotations

import csv
import dataclasses
import multiprocessing
import shutil
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))      # synthesis_scripts/, so claude_abundance/ is found
from claude_abundance.fe_line_fit import (ATOMIC_NUMBER, BABSMA, BSYN, CONFIG_DIR, FALTBON, FALTBON_PROFILE,  # noqa: E402
                         RUNS_DIR, SCRATCH_DIR, SOLAR_FE, SPHERICAL_RT, TS_DATA,
                         individual_abundances, isotope_block, load_lines, load_spectrum, load_star_config, resolve_path,
                         run_program)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

SELECTED_STAR = "HD2454"

WMIN, WMAX = 3800.0, 9000.0          # A
DW = 0.02                            # synthesis step [A]
NLTE = "T"                           # "T": Fe in NLTE with the departure coefficients of stars.yaml; "F": LTE
VMIC = None                          # km/s; None -> `vmic` in stars.yaml (default 1.0)

GROUPS = None                        # None: every group in line_lists.yaml `molecular`; or e.g. ["CH", "CN", "CC"]
IMPACT_THRESHOLD = 0.005             # a group is recommended if it adds absorption deeper than this anywhere
                                     # (normalised flux; 0.005 = 0.5 %, about the noise of a SNR ~200 spectrum)
LINE_WINDOW = 0.4                    # A: half-width of the Fe line windows checked separately (if a line file exists)

# Abundances of the other elements, as in find_stellar_parameters
USE_REFERENCE_ABUNDANCES = True      # config/reference_abundance.yaml for SELECTED_STAR (null entries are not used)
REFERENCE_SCALING = "fe"             # "absolute": A(X) as published; "fe": keep the published [X/Fe]
ABUNDANCE_OVERRIDES = {              # element: A(X) (log eps, H = 12), or None = back to scaled solar
    # "N": 7.5,
}
USE_ISOTOPE_RATIOS = True            # 12C/13C from `isotope_ratios: C12_C13` in reference_abundance.yaml

EXTRA_BROADENING = None              # None -> `vmac` in stars.yaml if set (rad-tan, faltbon FWHM argument in km/s),
                                     # else none; or e.g. ("radtan", 5.5) / ("gauss", 8.0) / ("rot", 15.0) (v sin i).
                                     # Matters: unbroadened molecular features look far deeper than observed ones
N_JOBS = 12                          # parallel syntheses (one per group + the reference)
KEEP_SPECTRA = True


PROFILES = dict(FALTBON_PROFILE, rot=4)   # faltbon profile types; "rot": rotation, the number is v sin i

# ---------------------------------------------------------------------------
# Turbospectrum inputs
# ---------------------------------------------------------------------------

def abundance_block(abund: dict[str, float]) -> str:
    """'INDIVIDUAL ABUNDANCES' block, absolute log eps (H = 12), sorted by atomic number."""
    pairs = sorted((ATOMIC_NUMBER[el], a) for el, a in abund.items())
    return f"'INDIVIDUAL ABUNDANCES:'   '{len(pairs)}'\n" + "".join(f"{z}  {a:.4f}\n" for z, a in pairs)


def babsma_input(atm: Path, opac: Path, feoh: float, alpha: float, vmic: float, abund: dict) -> str:
    return f"""\
'LAMBDA_MIN:'  '{WMIN}'
'LAMBDA_MAX:'  '{WMAX}'
'LAMBDA_STEP:' '{DW}'
'MODELINPUT:' '{atm}'
'MARCS-FILE:' '.false.'
'MODELOPAC:' '{opac}'
'METALLICITY:'    '{feoh}'
'ALPHA/Fe   :'    '{alpha}'
'HELIUM     :'    '0.00'
'R-PROCESS  :'    '0.00'
'S-PROCESS  :'    '0.00'
{abundance_block(abund)}'XIFIX:' 'T'
{vmic}
"""


def bsyn_input(job: dict, result: Path) -> str:
    lls = job["linelists"]
    return f"""\
'NLTE :'          {job['nlte']}
'NLTEINFOFILE:'   '{job['nlte_info']}'
'LAMBDA_MIN:'     '{WMIN}'
'LAMBDA_MAX:'     '{WMAX}'
'LAMBDA_STEP:'    '{DW}'
'INTENSITY/FLUX:' 'Flux'
'MODELOPAC:' '{job['opac']}'
'RESULTFILE :' '{result}'
'METALLICITY:'    '{job['feoh']}'
'ALPHA/Fe   :'    '{job['alpha']}'
'HELIUM     :'    '0.00'
'R-PROCESS  :'    '0.00'
'S-PROCESS  :'    '0.00'
{abundance_block(job['abund'])}{isotope_block(job['c12_c13'])}'NFILES   :' '{len(lls)}'
""" + "".join(f"{p}\n" for p in lls) + f"""\
'SPHERICAL:'  '{"T" if SPHERICAL_RT else "F"}'
  30
  300.00
  15
  1.30
"""


def faltbon_input(spec: Path, out: Path, fwhm: float, profile: int) -> str:
    return f"""\
{spec}
{out}
{fwhm}  FWHM OF CONVOLU.PROFILE=  MILLIANGSTROM, OR KM/S IF < 0.
{profile} PROFILE TYPE=                (1=EXP, 2=GAUSS, 3=RAD-TAN, 4=ROT)
2 FLUX-SCALE = (1=REL.FLUX,  2=ABS.FLUX))
1 NBIN = (1 = NO IN BIN )
"""


def synthesise(job: dict) -> dict:
    """Worker: bsyn for one set of line lists, then the instrumental profile (+ extra broadening). Own scratch
    directory; the ~15 MB-per-call Turbospectrum log is only kept (its tail) when something fails."""
    wd = Path(tempfile.mkdtemp(prefix="mol_", dir=SCRATCH_DIR))
    (wd / "DATA").symlink_to(TS_DATA)
    log = wd / "log.txt"
    t0 = time.time()
    try:
        spec, cvl = wd / "synth.spec", wd / "synth.cvl"
        run_program(BSYN, bsyn_input(job, spec), log, wd)
        run_program(FALTBON, faltbon_input(spec, cvl, job["inst_fwhm"], 2), log, wd)
        if job["extra"] is not None and job["extra"][1] > 0:
            prof, value = job["extra"]
            out = wd / "synth_broad.cvl"
            run_program(FALTBON, faltbon_input(cvl, out, -abs(value), PROFILES[prof]), log, wd)
            cvl = out
        shutil.copy2(cvl, job["out"])
        return dict(name=job["name"], ok=True, seconds=time.time() - t0)
    except Exception as exc:                                  # noqa: BLE001 - report, keep the others going
        tail = "".join(deque(log.open(errors="replace"), maxlen=60)) if log.exists() else ""
        return dict(name=job["name"], ok=False, seconds=time.time() - t0, error=f"{type(exc).__name__}: {exc}",
                    log_tail=tail)
    finally:
        shutil.rmtree(wd, ignore_errors=True)


# ---------------------------------------------------------------------------
# Impact of a group
# ---------------------------------------------------------------------------

def impact(wl: np.ndarray, ref: np.ndarray, test: np.ndarray, line_windows: list[tuple[str, float]]) -> dict:
    """Absorption added by a group: a = ref - test (positive where the group absorbs)."""
    a = ref - test
    i = int(np.argmax(a))
    out = dict(max_depth=round(float(a[i]), 5), at_wavelength=round(float(wl[i]), 2),
               added_absorption_mA=round(float(np.clip(a, 0, None).sum() * DW * 1000.0), 1),
               angstrom_above_threshold=round(float((a > IMPACT_THRESHOLD).sum() * DW), 2),
               max_brightening=round(float(-a.min()), 5))       # > 0 would mean the group raises the flux
    if line_windows:
        per_line = []
        for lid, w in line_windows:
            m = np.abs(wl - w) <= LINE_WINDOW
            if m.any():
                per_line.append((lid, float(a[m].max())))
        hit = [(lid, d) for lid, d in per_line if d > IMPACT_THRESHOLD]
        out.update(fe_windows_max_depth=round(max((d for _, d in per_line), default=0.0), 5),
                   fe_windows_affected=len(hit),
                   fe_windows_affected_ids=" ".join(lid for lid, _ in sorted(hit, key=lambda x: -x[1])))
    return out


def plot_group(path: Path, name: str, wl, ref, test, obs, info: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = ref - test
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), gridspec_kw=dict(height_ratios=[1, 1.3]))
    ax1.plot(wl, a, color="tab:purple", lw=0.5)
    ax1.axhline(IMPACT_THRESHOLD, color="tab:red", ls="--", lw=0.8, label=f"threshold {IMPACT_THRESHOLD:g}")
    ax1.set_xlim(WMIN, WMAX)
    ax1.set_ylabel("added absorption\n(reference - with group)")
    ax1.set_title(f"{SELECTED_STAR}: {name}   max {info['max_depth']:.4f} at {info['at_wavelength']:.2f} A,  "
                  f"{info['added_absorption_mA']:.0f} mA in total,  {info['angstrom_above_threshold']:.1f} A above threshold",
                  fontsize=10)
    ax1.legend(fontsize=8, loc="upper right")
    w0 = info["at_wavelength"]
    m = np.abs(wl - w0) <= 3.0
    if obs is not None:
        mo = np.abs(obs[0] - w0) <= 3.0
        ax2.plot(obs[0][mo], obs[1][mo], "k.", ms=2, label="observed")
    ax2.plot(wl[m], ref[m], color="tab:blue", lw=1.0, label="reference (no molecules)")
    ax2.plot(wl[m], test[m], color="tab:red", lw=1.0, label=f"with {name}")
    ax2.set_xlim(w0 - 3, w0 + 3)
    ax2.set_xlabel("Wavelength [A]")
    ax2.set_ylabel("normalised flux")
    ax2.set_title(f"strongest feature of {name} (+/- 3 A)", fontsize=9)
    ax2.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_overview(path: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in rows if r["ok"]]
    names = [r["group"] for r in rows]
    depth = [max(r["max_depth"], 1e-6) for r in rows]
    colors = ["tab:red" if r["recommended"] else "0.6" for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(names, depth, color=colors)
    ax.axhline(IMPACT_THRESHOLD, color="tab:red", ls="--", lw=0.8, label=f"threshold {IMPACT_THRESHOLD:g}")
    ax.set_yscale("log")
    ax.set_ylabel("deepest added absorption")
    ax.set_title(f"{SELECTED_STAR}: impact of the molecular line lists ({WMIN:.0f}-{WMAX:.0f} A); red = recommended",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_star_config(SELECTED_STAR)
    if cfg.yaml_atm is None or cfg.yaml_dc is None or cfg.feoh is None or cfg.alpha_fe is None:
        raise SystemExit(f"{SELECTED_STAR}: stars.yaml needs model_atmosphere, departure_coefficients, metallicity "
                         "and alpha_fe")
    if not USE_REFERENCE_ABUNDANCES:
        cfg.reference_abundances, cfg.reference_notes = {}, {}
    cfg.reference_scaling = REFERENCE_SCALING
    cfg.abundance_overrides = dict(ABUNDANCE_OVERRIDES)
    feoh, alpha = cfg.feoh, cfg.alpha_fe
    vmic = VMIC if VMIC is not None else cfg.vmic
    abund = individual_abundances(cfg, feoh)
    abund["Fe"] = round(SOLAR_FE + feoh, 4)                  # Fe is not fitted here
    c12_c13 = cfg.c12_c13 if USE_ISOTOPE_RATIOS else None
    vmac = (cfg.raw.get("synthesis") or {}).get("vmac")
    extra = EXTRA_BROADENING if EXTRA_BROADENING is not None else (("radtan", float(vmac)) if vmac else None)

    ll_cfg = yaml.safe_load((CONFIG_DIR / "line_lists.yaml").read_text())
    master = [resolve_path(p) for p in ll_cfg["master"]]
    groups = GROUPS or list(ll_cfg["molecular"])
    unknown = [g for g in groups if g not in ll_cfg["molecular"]]
    if unknown:
        raise SystemExit(f"groups not in line_lists.yaml: {unknown}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = RUNS_DIR / SELECTED_STAR / "molecular_tests" / f"{WMIN:.0f}-{WMAX:.0f}_{stamp}"
    (outdir / "spectra").mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), outdir / Path(__file__).name)

    print(f"{SELECTED_STAR}: {cfg.yaml_atm.name}\n  [Fe/H] {feoh:+.2f}  [alpha/Fe] {alpha:+.2f}  vmic {vmic:.2f}  "
          f"{'NLTE' if NLTE == 'T' else 'LTE'}  instrument {cfg.inst_fwhm} km/s  extra broadening {extra}  "
          f"12C/13C {c12_c13 if c12_c13 is not None else 'solar'}")
    print("  individual abundances: " + ", ".join(f"{el} {a:.2f}" for el, a in sorted(abund.items(), key=lambda x: ATOMIC_NUMBER[x[0]])))
    print(f"  {WMIN:.0f}-{WMAX:.0f} A, step {DW} A; groups: {', '.join(groups)}\n-> {outdir}")

    # shared files: continuous opacity (babsma once) and the NLTE info file
    shared = Path(tempfile.mkdtemp(prefix="molshared_", dir=SCRATCH_DIR))
    try:
        (shared / "DATA").symlink_to(TS_DATA)
        opac = shared / "cont.opac"
        nlte_info = shared / "nlte_infofile.dat"
        nlte_info.write_text("#\n# path for model atom files\n" f"{cfg.atom_path}/\n" "# path for departure files\n"
                             f"{cfg.yaml_dc.parent}/\n"
                             f"{cfg.Z} '{cfg.element}' 'nlte' '{cfg.model_atom}' '{cfg.yaml_dc.name}' 'ascii'\n")
        t0 = time.time()
        run_program(BABSMA, babsma_input(cfg.yaml_atm, opac, feoh, alpha, vmic, abund), shared / "babsma.log", shared)
        print(f"  babsma: {time.time() - t0:.0f} s")

        base = dict(nlte=NLTE, nlte_info=nlte_info, opac=opac, feoh=feoh, alpha=alpha, abund=abund, c12_c13=c12_c13,
                    inst_fwhm=cfg.inst_fwhm, extra=extra)
        jobs = [dict(base, name="reference", linelists=master, out=outdir / "spectra" / "reference.cvl")]
        for g in groups:
            files = [resolve_path(p) for p in ll_cfg["molecular"][g]]
            missing = [str(p) for p in files if not p.exists()]
            if missing:
                raise SystemExit(f"{g}: line list(s) not found: {missing}")
            jobs.append(dict(base, name=g, linelists=master + files, out=outdir / "spectra" / f"{g}.cvl"))

        results = {}
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=N_JOBS, mp_context=multiprocessing.get_context("fork")) as ex:
            futures = [ex.submit(synthesise, j) for j in jobs]
            for fut in as_completed(futures):
                r = fut.result()
                results[r["name"]] = r
                print(f"  {r['name']:10s} {'done' if r['ok'] else 'FAILED: ' + r['error']}  ({r['seconds']:.0f} s)", flush=True)
                if not r["ok"]:
                    (outdir / f"{r['name']}_log_tail.txt").write_text(r["log_tail"])
        print(f"  all syntheses: {time.time() - t0:.0f} s")
    finally:
        shutil.rmtree(shared, ignore_errors=True)

    if not results["reference"]["ok"]:
        raise SystemExit("the reference synthesis failed; see reference_log_tail.txt")
    wl, ref = np.loadtxt(outdir / "spectra" / "reference.cvl", usecols=(0, 1), unpack=True)

    obs = None
    if cfg.spectrum_file is not None and cfg.spectrum_file.exists():
        ow, of = load_spectrum(cfg.spectrum_file, cfg.rv)
        m = (ow >= WMIN) & (ow <= WMAX)
        obs = (ow[m], of[m])
    line_windows = []
    if cfg.line_file.exists():
        line_windows = [(r["line_id"], r["wavelength"]) for r in load_lines(cfg.line_file)]
    else:
        print(f"  (no line file {cfg.line_file}: the Fe line windows are not checked separately)")

    rows = []
    for g in groups:
        row = dict(group=g, ok=results[g]["ok"], files=" ".join(Path(p).name for p in ll_cfg["molecular"][g]))
        if row["ok"]:
            w2, test = np.loadtxt(outdir / "spectra" / f"{g}.cvl", usecols=(0, 1), unpack=True)
            if len(w2) != len(wl) or not np.allclose(w2, wl):
                test = np.interp(wl, w2, test)
            info = impact(wl, ref, test, line_windows)
            row.update(info, recommended=info["max_depth"] > IMPACT_THRESHOLD)
            plot_group(outdir / "plots" / f"{g}.png", g, wl, ref, test, obs, info)
        else:
            row.update(recommended=None, error=results[g]["error"])
        rows.append(row)
    rows.sort(key=lambda r: -r.get("max_depth", -1))
    plot_overview(outdir / "plots" / "impact_overview.png", rows)

    with (outdir / "summary.csv").open("w", newline="") as f:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    recommended = [r["group"] for r in rows if r.get("recommended")]
    (outdir / "summary.yaml").write_text(yaml.dump(dict(
        star=SELECTED_STAR, atmosphere=cfg.yaml_atm.name, departure_coefficients=cfg.yaml_dc.name, nlte=NLTE == "T",
        wavelength_range=[WMIN, WMAX], step=DW, metallicity=feoh, alpha_fe=alpha, vmic=vmic,
        instrument_fwhm=cfg.inst_fwhm, extra_broadening=list(extra) if extra else None,
        individual_abundances={el: float(a) for el, a in abund.items()}, c12_c13=c12_c13,
        reference_scaling=REFERENCE_SCALING if USE_REFERENCE_ABUNDANCES else None,
        abundance_overrides=dict(ABUNDANCE_OVERRIDES), impact_threshold=IMPACT_THRESHOLD,
        recommended=recommended, groups=rows), sort_keys=False))

    print(f"\n{'group':6s} {'max depth':>9s} {'at [A]':>9s} {'added mA':>9s} {'A > thr':>8s}"
          + ("  Fe windows > thr" if line_windows else ""))
    for r in rows:
        if not r["ok"]:
            print(f"{r['group']:6s}  failed: {r['error']}")
            continue
        flag = "  <- include" if r["recommended"] else ""
        extra_col = f"  {r['fe_windows_affected']:3d} (max {r['fe_windows_max_depth']:.4f})" if line_windows else ""
        print(f"{r['group']:6s} {r['max_depth']:9.4f} {r['at_wavelength']:9.2f} {r['added_absorption_mA']:9.0f} "
              f"{r['angstrom_above_threshold']:8.1f}{extra_col}{flag}")
    print(f"\nrecommended for stars.yaml (absorption deeper than {IMPACT_THRESHOLD:g} somewhere in {WMIN:.0f}-{WMAX:.0f} A):")
    print("    molecular_line_lists:" + ("".join(f'\n      - "{g}"' for g in recommended) if recommended else " null"))
    print(f"Results in {outdir}")


if __name__ == "__main__":
    main()
