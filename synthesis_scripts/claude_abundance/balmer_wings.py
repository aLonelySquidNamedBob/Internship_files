#!/usr/bin/env python3
"""
Synthetic H-alpha and H-beta over the observed spectrum, as a check of Teff from the Balmer wings.

Run it as a plain script: edit the settings below, then `python balmer_wings.py`.

Takes a find_stellar_parameters run (default: the newest one of the star), synthesises the two lines for its
atmosphere and parameters (same line lists, abundances and broadening as its Fe fits) and for the same star with
Teff +/- DTEFF (SHOW_DTEFF) and for CUSTOM parameters (e.g. from a paper), and plots them over the normalised spectrum together with the residuals (observed - synthetic).
Nothing is fitted or masked. Output: runs/<star>/balmer_wings/<time>_<run>/H-alpha.png and H-beta.png
"""

from __future__ import annotations

import dataclasses
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))     # find fe_line_fit.py next to this script
from fe_line_fit import (C_KMS, RUNS_DIR, SOLAR_FE, AtmosphereSpec, LineSynth, load_spectrum, load_star_config,
                         parse_atm_name, trim_linelists)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

SELECTED_STAR = "HD2454"
RUN = None                    # run directory (or name) in runs/<star>/fe_abundance/; None: the newest one
SHOW_DTEFF = True            # False: leave out the Teff - DTEFF and Teff + DTEFF syntheses
DTEFF = 300                   # the two extra syntheses are Teff - DTEFF and Teff + DTEFF [K]

# One more synthesis with any parameters, e.g. the ones of a paper: a dict, or None to leave it out.
# teff, logg, feh are required; vmic is optional (default: the Fe run's), so is mass (default: the Fe run's; 0.0 =
# plane-parallel, the spherical MARCS grid stops at logg 3.5). The atmosphere is interpolated like the others. The
# abundances are those of the Fe run shifted by feh - [Fe/H] of the run (the same [X/Fe]), A(Fe) = 7.45 + feh.

SHOW_CUSTOM = False
CUSTOM = dict(label="Riano", teff=6565, logg=4.11, feh=-0.21, vmic=1.6) if SHOW_CUSTOM else None


LINES = {"H-alpha": 6562.80, "H-beta": 4861.30}    # centres [A]
HALF_WIDTH = 40.0             # plotted and synthesised range around each line: centre +/- this [A]
YLIM = (0.80, 1.03)           # flux range of the spectrum panels (the core is cut off to show the wings)
RESID_YLIM = 0.04             # residual panels: +/- this

_FSP = Path(__file__).with_name("find_stellar_parameters")


def synth_one(job: dict):
    """Worker: one line, one atmosphere -> (wavelength, flux) after the instrumental and extra broadening."""
    cfg, atm, lo, hi = job["cfg"], job["atm"], job["lo"], job["hi"]
    workdir = Path(tempfile.mkdtemp(prefix="bw_"))
    try:
        cfg = dataclasses.replace(cfg, linelists=trim_linelists(cfg.linelists, lo - 10, hi + 10, workdir))
        synth = LineSynth(cfg, atm, lo, hi, workdir, workdir / "log.txt")
        return synth.broadened(job["abu"], job["nlte"], job["broad"], job["profile"])
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> None:
    # --- the Fe run ---
    base = RUNS_DIR / SELECTED_STAR / "fe_abundance"
    if RUN:
        run_dir = Path(RUN) if Path(RUN).is_absolute() or Path(RUN).exists() else base / RUN
    else:
        run_dir = sorted(base.glob("*/summary.yaml"), key=lambda f: f.stat().st_mtime)[-1].parent
    summary = yaml.safe_load((run_dir / "summary.yaml").read_text())
    teff, logg, vmic, feh = summary["teff"], summary["logg"], summary["vmic"], summary["feh_atmosphere"]
    mass = float(summary.get("mass", 1.0))
    nlte = "T" if summary["nlte"] else "F"
    fs = summary["fit_settings"]

    cfg = load_star_config(SELECTED_STAR)
    cfg.reference_abundances, cfg.reference_notes, cfg.reference_scaling = {}, {}, "absolute"
    cfg.abundance_overrides = dict(summary["individual_abundances"])         # the abundances of the Fe run

    # RV: median shift of the Fe lines used in the run (observed wavelengths / (1 + dv/c), as in the line fits)
    import csv
    with (run_dir / "lines.csv").open() as f:
        dvs = [float(r["dv_kms"]) for r in csv.DictReader(f) if r["selection"] == "used"]
    rv = float(np.median(dvs)) if dvs else 0.0
    print(f"{SELECTED_STAR}: {run_dir.name}  (RV shift {rv:+.2f} km/s)")

    outdir = RUNS_DIR / SELECTED_STAR / "balmer_wings" / f"{time.strftime('%Y%m%d_%H%M%S')}_{run_dir.name}"
    outdir.mkdir(parents=True)
    shutil.copy2(Path(__file__), outdir / Path(__file__).name)

    # --- atmospheres: the run's, Teff -/+ DTEFF, and CUSTOM ---
    loader = SourceFileLoader("find_stellar_parameters", str(_FSP))
    fsp = module_from_spec(spec_from_loader("find_stellar_parameters", loader))
    loader.exec_module(fsp)                                    # for make_atmosphere (the interpolator)
    # every model: key, legend label, colour, line width, linestyle, atmosphere, config (abundances), A(Fe)
    run_abu = float(summary["results"]["A1"])
    models = [dict(key=0, label=f"Teff = {teff:.0f} K (Fe run)", colour="grey", lw=1.8, ls="-", cfg=cfg, abu=run_abu,
                   atm=AtmosphereSpec(atm=run_dir / "atm" / summary["atmosphere"],
                                      dc=run_dir / "atm" / summary["departure_coefficients"], feoh=feh,
                                      alpha_fe=parse_atm_name(summary["atmosphere"])["alpha_fe"], vmic=vmic,
                                      teff=teff, logg=logg))]
    if SHOW_DTEFF:
        for d, colour in ((-DTEFF, "#2a78d6"), (+DTEFF, "#e34948")):
            atm = fsp.make_atmosphere(teff + d, logg, feh, round(SOLAR_FE + feh, 2), mass, cfg.element, vmic,
                                      outdir / "atm" / f"T{teff + d:.0f}")
            models.append(dict(key=d, label=f"Teff {'+' if d > 0 else '−'} {abs(d)} K", colour=colour, lw=1.2,
                               ls="-", cfg=cfg, abu=run_abu, atm=atm))
    if CUSTOM:
        t, g, z = CUSTOM["teff"], CUSTOM["logg"], CUSTOM["feh"]
        v, m = CUSTOM.get("vmic", vmic), CUSTOM.get("mass", mass)
        atm = fsp.make_atmosphere(t, g, z, round(SOLAR_FE + z, 2), m, cfg.element, v, outdir / "atm" / "custom")
        ccfg = dataclasses.replace(cfg, abundance_overrides={el: a + (z - feh) for el, a in
                                                             summary["individual_abundances"].items()})
        models.append(dict(key="custom", colour="#1baf7a", lw=1.5, ls="--", cfg=ccfg, abu=SOLAR_FE + z, atm=atm,
                           label=f"{CUSTOM.get('label', 'custom')}: Teff={t:.0f} logg={g:.2f} [Fe/H]={z:+.2f} "
                                 f"vmic={v:.2f}"))

    # --- syntheses ---
    jobs = [dict(cfg=m["cfg"], atm=m["atm"], lo=c - HALF_WIDTH - 5, hi=c + HALF_WIDTH + 5, abu=m["abu"],
                 nlte=nlte, broad=float(fs["broadening"]), profile=fs["broad_profile"], key=(name, m["key"]))
            for name, c in LINES.items() for m in models]
    with ProcessPoolExecutor(max_workers=min(len(jobs), 12)) as ex:
        synth = dict(zip([j["key"] for j in jobs], ex.map(synth_one, jobs)))

    # --- plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obs_wl, obs_fl = load_spectrum(cfg.spectrum_file, 0.0)
    obs_wl = obs_wl / (1.0 + rv / C_KMS)

    for name, c in LINES.items():
        fig, (ax, rax) = plt.subplots(2, 1, figsize=(20, 9), sharex=True,
                                      gridspec_kw=dict(height_ratios=[3, 1.3], hspace=0.05))
        sel = np.abs(obs_wl - c) <= HALF_WIDTH
        ax.plot(obs_wl[sel] - c, obs_fl[sel], ".", ms=2, color="black", label="observed")
        rax.axhline(0, color="0.3", lw=0.8)
        for m in models:
            wl, fl = synth[(name, m["key"])]
            ax.plot(wl - c, fl, color=m["colour"], lw=m["lw"], ls=m["ls"], label=m["label"], alpha=1.0 if m["key"] == 0 else 0.6)
            rax.plot(obs_wl[sel] - c, obs_fl[sel] - np.interp(obs_wl[sel], wl, fl), color=m["colour"], ls=m["ls"],
                     lw=0.9 if m["key"] == 0 else 0.7, alpha=1.0 if m["key"] == 0 else 0.6)
        ax.set_title(f"{SELECTED_STAR}  {name}  ({c:.1f} Å)     Fe run: Teff={teff:.0f} logg={logg:.2f} "
                     f"vmic={vmic:.2f} [Fe/H]={feh:+.2f}", loc="left")
        ax.set_ylim(*YLIM)
        ax.set_xlim(-HALF_WIDTH, HALF_WIDTH)
        ax.set_ylabel("normalised flux")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
        rax.set_ylim(-RESID_YLIM, RESID_YLIM)
        rax.set_xlabel("λ − λ$_0$ [Å]")
        rax.set_ylabel("observed − synthetic")
        rax.grid(alpha=0.3)
        fig.savefig(outdir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot: {outdir / (name + '.png')}")


if __name__ == "__main__":
    main()
