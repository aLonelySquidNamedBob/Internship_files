#!/usr/bin/env python3
"""
A(X) of every selected line of one element in one star, for ONE fixed set of stellar parameters: the element in NLTE,
every other element (Fe included) in LTE. The counterpart of find_stellar_parameters for elements other than Fe (it also
works for Fe). The converged stellar parameters and the broadening are read from config/stars.yaml; the settings below
can override them for a run.

Run it as a plain script: edit the settings below, then `python find_element_abundance.py`.

What it does (one pass, nothing is iterated, so a run is fully set by the settings below):
  1. interpolates a MARCS atmosphere + the departure coefficients (DCs) of the element at (TEFF, LOGG, FEH) with
     interp_nlte_4.py, for the starting A(X): DC_ABUNDANCE, else the reference file, else scaled solar,
  2. reports which selected lines have model-atom levels in the line list (a line without them is LTE even in an NLTE
     run; no level is invented),
  3. fits A(X) line by line with fe_line_fit.fit_lines (see fe_line_fit.py for the method),
  4. writes the per-line table and a summary per ion: A(X), [X/H], [X/Fe] (with FEH as [Fe/H]).

The grid: data/original/NLTE/dc/<element>/ with dc.bin and master.dat, and an entry in config/nlte.yaml. The DCs are not
interpolated in abundance: each corner of the interpolation uses the record closest to the requested A(X). A warning is
printed when that is more than 0.1 dex away (outside the grid, or a gap in it), and a note when the fitted A(X) ends up more than
0.1 dex from the requested one: set DC_ABUNDANCE to the fitted value and run again.

The lines to fit: the rows of ELEMENT in data/selected_lines/<star>/<star>_kept_lines.csv, one file with the lines of every
species but Fe (Fe stays in fe_kept_lines.csv), same columns as fe_kept_lines.csv (only `species` and `wavelength` are
required). Notes on lines: <el>_line_flags.yaml next to it (review_fits.py writes it), as for Fe.

Everything goes to  ~/NICO/runs/<star>/<el>_abundance/<time>_T<teff>_g<logg>_v<vmic>_z<feh>_<NLTE|LTE>/ :
    lines.csv            one row per line (columns as in the Fe runs; other_depth_* = depth of the profile without the element)
    summary.yaml         parameters, DCs requested and used, per-ion results, which lines were used or rejected, settings
    nlte_levels.csv      NLTE runs: model-atom levels of every selected line, from the line list
    overview.png         all line fits in one picture, numbered and coloured by selection
    diagnostics.png      only with MAKE_DIAGNOSTICS: A(X) against EP, REW and wavelength
    lines/<line>/        fit.png, bestfit.csv, chi2_curve.csv and the synthetic spectra of every trial abundance
    atm/                 the interpolated atmosphere and departure coefficients
"""

from __future__ import annotations

import dataclasses
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))     # find fe_line_fit.py next to this script
from fe_line_fit import (FALTBON_PROFILE, RUNS_DIR, FitSettings, StarConfig, fit_lines, individual_abundances, is_reject,
                         line_flags_path, load_line_flags, load_lines, load_spectrum, parse_atm_name, plot_overview,
                         resolve_path, save_line_flags, write_csv)
from element_line_fit import (analyze_element, ba_mix, describe_dc, element_totals_text, eu_mix, isotope_fractions,
                              load_element_config, main_ion, make_atmosphere, nlte_level_report,
                              plot_element_diagnostics, print_element_summary, start_abundance)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

SELECTED_STAR = "HD196944"
ELEMENT = "Ba"

# None = the value under `synthesis:` for the star in config/stars.yaml (teff, logg, vmic, feoh, vmac, vmac_profile); a
# value here overrides it for this run. A value that is in neither place stops the run.
TEFF = None                          # K
LOGG = None                          # dex (cgs)
VMIC = None                          # km/s
FEH = None                           # [Fe/H] of the atmosphere, and of the Fe lines (LTE): A(Fe) = 7.45 + FEH
VMAC = None                          # km/s, the extra broadening: faltbon's argument for VMAC_PROFILE (gauss: the FWHM;
                                     # radtan: zeta_RT / 1.433). Not needed when fit_broadening=True
VMAC_PROFILE = None                  # "gauss" or "radtan" (needed even when the broadening is fitted)
NLTE = "T"                          # "T" = the element in NLTE (everything else stays LTE), "F" = all LTE

MASS = None                         # MARCS mass: None -> from `model_atmosphere` in stars.yaml (1.0 = spherical,
                                     # 0.0 = plane-parallel)
DC_ABUNDANCE = None                  # A(X) of the departure coefficients, of babsma and of the first guess; None -> the
                                     # reference file (with REFERENCE_SCALING) or ABUNDANCE_OVERRIDES, else scaled solar
MAKE_DIAGNOSTICS = False             # True: also write diagnostics.png (A(X) against EP, REW and wavelength)
ISOTOPE_FRACTIONS = {                # fixed isotopic mixtures for bsyn (empty: Turbospectrum's, from makeabund.f), e.g.
    # "Ba": ba_mix(f_odd=0.46),      # f_odd = N(135Ba + 137Ba) / N(Ba), keeping the ratios within the odd and even isotopes
    # "Eu": eu_mix(f151=0.50),       # f151 = N(151Eu) / N(Eu)
    # "Ba": {135: 0.10, 137: 0.15},  # or per isotope; the isotopes not given keep Turbospectrum's fraction
}
RUN_NOTE = ""                        # what is special about this run; goes into summary.yaml
LINE_FILE = None                     # None -> data/selected_lines/<star>/<star>_kept_lines.csv (rows of ELEMENT only)
ONLY_LINES = None                    # e.g. ["Ba2_4554.0330", "Ba2_6141.7130"]: fit only these (quick checks)
FIT_SETTINGS = FitSettings(          # all options and their meaning: class FitSettings in fe_line_fit.py
    n_jobs=10,
    half_window=0.6,                 # default fit region = line centre +/- this [A] at window_ref_wavelength, scaled
    window_ref_wavelength=5000.0,    # with lambda (constant in km/s). Strong lines (Ba II 4554 and 4934, Mg b, Sr II ...)
                                     # need more: set chi2_left / chi2_right for them in the line CSV (or with review_fits.py)
    fit_broadening=False,           # True: fit it per line (0..broad_max); False: VMAC / `vmac` of stars.yaml for every line
    broad_max=20.0,
    fit_rv=True,                     # True: fit a small RV shift per line (+/- rv_max, 3 km/s); False: rv_fixed km/s
    rv_fixed=0.0,
    sigma_clip=False,                # True: clip around the mean of each ion, with at least 5 lines
    clip_sigma=3.00,
    max_misfit=None,                 # reject lines whose systematic misfit is larger (`misfit` in lines.csv); None = no cut
    chi2_factor=None,
    min_ep=None,                     # None = no EP cut: the resonance lines (Ba II 4554, Sr II 4077, Ca I 4226 ...) have EP = 0
    fit_continuum=True,              # True: also fit a local continuum scale per line
    cont_method="median_clip",       # "lsq": scale fitted with the line; "median_clip": clipped median over continuum pixels
    fit_region="window",             # pixels in chi^2: "window", "line" (fit_left..fit_right of the CSV) or "core"
    a_step_warm=0.10,
    a_tol=0.001,
    a_local=0.05,
    abund_band=True,                 # also synthesise every line at best-fit A(X) -/+ abund_band_offset and shade it
    abund_band_offset=0.2,
    assign_nlte_levels=False,        # DO NOT INVENT NLTE LEVELS. (Fe only anyway)
)
# Abundances of the other elements (blends, molecules, electron donors), as in find_stellar_parameters. Anything not set
# here or in the reference file stays scaled solar for FEH; Fe follows FEH; the fitted element gets its trial value.
USE_REFERENCE_ABUNDANCES = True      # config/reference_abundance.yaml for SELECTED_STAR (null entries are not used)
REFERENCE_SCALING = "absolute"       # "absolute", "fe" or "xfe", see find_stellar_parameters
ABUNDANCE_OVERRIDES = {              # element: A(X), or None = back to scaled solar. Beats the reference file.
}
USE_ISOTOPE_RATIOS = True            # 12C/13C from the reference file (bsyn ISOTOPES)
KEEP_LINE_FILES = True               # keep lines/<line>/ (fit.png, bestfit.csv, synth/, chi2_curve.csv)
LINE_FLAGS_FILE = None               # None -> <el>_line_flags.yaml next to the line CSV
ONLY_FLAGGED = False                 # True: fit only the flagged lines (run directory gets _flagged)

DC_NOTE_DEX = 0.1                    # note when the fitted A(X) is this far from the abundance of the DCs


# ---------------------------------------------------------------------------
# One parameter set
# ---------------------------------------------------------------------------

def _plain(v):
    """numpy scalars -> python numbers, for yaml."""
    return v.item() if isinstance(v, np.generic) else v


# Column names of fit_lines that are about Fe in the Fe runs and about "everything but the element" here
RENAMED = {"nofe_depth_centre": "other_depth_centre", "nofe_depth_max": "other_depth_max", "no_fe_status": "no_element_status"}


def run_point(cfg: StarConfig, lines: list[dict], spectrum, teff: float, logg: float, vmic: float, feh: float, nlte: str,
              mass: float, start: float, outdir: Path, settings: FitSettings, flags: dict[str, str] | None = None,
              note: str = "") -> tuple[list[dict], dict]:
    """Interpolate the atmosphere and the departure coefficients, fit every line, write lines.csv / summary.yaml /
    overview.png to outdir. `start`: A(X) of the departure coefficients, of babsma and of the first guess."""
    flags = flags or {}
    outdir.mkdir(parents=True, exist_ok=True)
    mode = "NLTE" if nlte == "T" else "LTE"
    head = f"{cfg.star} {cfg.element} {mode}: Teff={teff:.0f} logg={logg:.2f} vmic={vmic:.2f} [Fe/H]={feh:+.2f}"
    print("Building atmosphere...", end="\r", flush=True)
    atm, dc = make_atmosphere(teff, logg, feh, cfg.element, start, mass, vmic, outdir / "atm", nlte=nlte == "T")
    print("Building atmosphere...  DONE")
    dc_text = describe_dc(dc, cfg.element) if nlte == "T" else []
    for t in dc_text:
        print(t)

    cfg = dataclasses.replace(cfg, babsma_abundance=start)
    levels = nlte_level_report(cfg, lines)                    # also fills in a missing EP, so before the fit
    if nlte == "T":
        write_csv(outdir / "nlte_levels.csv", [levels[r["line_id"]] for r in lines])
        lte = [i for i, v in levels.items() if v["levels_used"] == "LTE"]
        partly = [i for i, v in levels.items() if v["how"].startswith("partly")]
        print(f"NLTE levels in the line list: {len(lines) - len(lte) - len(partly)} line(s) fully NLTE, "
              f"{len(partly)} partly, {len(lte)} LTE only" + (f" ({', '.join(lte)})" if lte else ""))

    t0 = time.time()
    rows = fit_lines(lines, atm, cfg, spectrum, outdir, settings=settings, nlte=nlte, default_abundance=start, levels=levels)
    rows = [{RENAMED.get(k, k): v for k, v in r.items()} for r in rows]
    info = analyze_element(rows, settings, flags, cfg.element, feh)
    for r, sel, flag in zip(rows, info["selection"], info["flag"]):
        r["selection"], r["flag"] = sel, flag
    write_csv(outdir / "lines.csv", rows)
    if MAKE_DIAGNOSTICS:
        plot_element_diagnostics(rows, info, outdir / "diagnostics.png", head)
    plot_overview(rows, outdir, outdir / "overview.png",
                  windows={r["line_id"]: (r["fit_left"], r["fit_right"]) for r in lines}, title=head,
                  totals=element_totals_text(info), fit_settings=dataclasses.asdict(settings))
    if not KEEP_LINE_FILES:
        shutil.rmtree(outdir / "lines", ignore_errors=True)

    k = main_ion(info)
    if nlte == "T" and dc["abundances_per_model"] > 1 and k is not None and abs(info[f"A{k}"] - start) > DC_NOTE_DEX:
        print(f"NOTE: A({cfg.element}) fitted = {info[f'A{k}']:.2f} is {info[f'A{k}'] - start:+.2f} dex from the abundance of the "
              f"departure coefficients ({start:.2f}): set DC_ABUNDANCE = {info[f'A{k}']:.2f} and run again")

    summary = dict(
        star=cfg.star, element=cfg.element, note=note, teff=teff, logg=logg, vmic=vmic, feh_atmosphere=feh,
        nlte=(nlte == "T"), mass=mass, atmosphere=atm.atm.name, start_abundance=start,
        departure_coefficients=atm.dc.name if nlte == "T" else None,
        dc_abundance_requested=dc["requested"] if nlte == "T" else None,
        dc_abundance_used=dict(zip(dc["corners"], dc["used"])) if nlte == "T" else None,
        dc_models_far_from_request=sorted({c for c, off in zip(dc["corners"], dc["far"]) if off}) if nlte == "T" else None,
        dc_grid_is_subset=dc["subset"] if nlte == "T" else None,
        nlte_levels={i: f"{v['levels_used']} ({v['how']})" for i, v in levels.items()} if nlte == "T" else None,
        isotope_fractions=dict(cfg.isotope_fractions), c12_c13=cfg.c12_c13,
        solar_abundance_scale="asp2007 (Turbospectrum default)",
        reference_scaling=cfg.reference_scaling if cfg.reference_abundances else None,
        abundance_overrides=dict(cfg.abundance_overrides),
        individual_abundances=individual_abundances(cfg, atm.feoh),     # everything but the fitted element
        manual_fit_regions={r["line_id"]: [r["chi2_left"], r["chi2_right"]] for r in rows if r["region_source"] == "manual"},
        spectrum=str(cfg.spectrum_file), snr=cfg.snr, line_file=str(cfg.line_file),
        results={key: _plain(v) for key, v in info.items()
                 if key not in ("used_ids", "clipped_ids", "flagged_ids", "rejected_by_hand_ids", "max_synth_ids",
                                "selection", "flag")},
        used_lines=info["used_ids"],
        sigma_clipped_lines={r["line_id"]: r["selection"] for r in rows if r["selection"].startswith("sigma")},
        flagged_lines={r["line_id"]: f"{r['flag']}  [{r['selection']}]" for r in rows
                       if r["flag"] and r["line_id"] in info["flagged_ids"]},
        max_synth_reached=info["max_synth_ids"],
        rejected_lines={r["line_id"]: r["selection"][len("rejected: "):] for r in rows if r["selection"].startswith("rejected")},
        fit_settings=dataclasses.asdict(settings),
        runtime_s=round(time.time() - t0),
    )
    (outdir / "summary.yaml").write_text(yaml.dump(summary, sort_keys=False))
    return rows, info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def stellar_parameters(cfg: StarConfig, settings: FitSettings) -> dict[str, float | str]:
    """TEFF, LOGG, VMIC, FEH, VMAC_PROFILE and VMAC (only when the broadening is not fitted): the setting at the top of
    this script if there is one, else `synthesis:` of the star in stars.yaml. A value that is in neither stops the run: a
    default would silently bias the abundances of converged stars."""
    syn = cfg.raw["synthesis"]
    wanted = {"TEFF": ("teff", TEFF), "LOGG": ("logg", LOGG), "VMIC": ("vmic", VMIC), "FEH": ("feoh", FEH),
              "VMAC_PROFILE": ("vmac_profile", VMAC_PROFILE)}
    if not settings.fit_broadening:
        wanted["VMAC"] = ("vmac", VMAC)
    found, missing = {}, []
    for name, (key, override) in wanted.items():
        value = syn.get(key) if override is None else override
        if value is None:
            missing.append(f"{key} ({name})")
        else:
            found[name] = value if name == "VMAC_PROFILE" else float(value)
            print(f"{name:12s} = {value}  ({'stars.yaml' if override is None else 'override in the settings'})")
    if missing:
        raise SystemExit(f"{cfg.star}: no value for {', '.join(missing)}: add it under `synthesis:` in config/stars.yaml, "
                         "or set it at the top of this script")
    if found["VMAC_PROFILE"] not in FALTBON_PROFILE:
        raise SystemExit(f"{cfg.star}: vmac_profile must be one of {list(FALTBON_PROFILE)}, not {found['VMAC_PROFILE']!r}")
    return found


def main() -> None:
    cfg = load_element_config(SELECTED_STAR, ELEMENT, nlte=NLTE == "T")
    params = stellar_parameters(cfg, FIT_SETTINGS)
    teff, logg, vmic, feh = params["TEFF"], params["LOGG"], params["VMIC"], params["FEH"]
    settings = dataclasses.replace(FIT_SETTINGS, broad_profile=params["VMAC_PROFILE"],
                                   broadening=params.get("VMAC", FIT_SETTINGS.broadening))
    if cfg.spectrum_file is None:
        raise SystemExit(f"{SELECTED_STAR}: no normalized spectrum listed in stars.yaml")
    if LINE_FILE:
        cfg.line_file = resolve_path(LINE_FILE)
    if not cfg.line_file.exists():
        raise SystemExit(f"no line list for {ELEMENT}: {cfg.line_file} does not exist (or set LINE_FILE)")
    if (cfg.raw.get("spectra") or {}).get("snr") is None:
        print(f"note: no `spectra: snr:` for {SELECTED_STAR} in stars.yaml, using SNR = {cfg.snr:g}")
    if not USE_REFERENCE_ABUNDANCES:
        cfg.reference_abundances, cfg.reference_notes, cfg.relative_abundances = {}, {}, {}
    if not (USE_REFERENCE_ABUNDANCES and USE_ISOTOPE_RATIOS):
        cfg.c12_c13 = None
    cfg.reference_scaling = REFERENCE_SCALING
    cfg.abundance_overrides = dict(ABUNDANCE_OVERRIDES)
    cfg.isotope_fractions = isotope_fractions(ISOTOPE_FRACTIONS)
    start = round(start_abundance(cfg, feh, DC_ABUNDANCE), 2)
    print(f"12C/13C: {cfg.c12_c13 if cfg.c12_c13 is not None else 'solar'}")
    if cfg.isotope_fractions:
        print("isotopic fractions: " + "  ".join(f"{k} {v:.4f}" for k, v in cfg.isotope_fractions.items()))
    print(f"individual abundances (everything else scaled solar for [Fe/H] = {feh:+.2f}, {ELEMENT} fitted):")
    for el, a in individual_abundances(cfg, feh).items():
        src = "override" if el in ABUNDANCE_OVERRIDES else cfg.reference_notes.get(el, "")
        print(f"  {el:2s} {a:6.2f}" + (f"  ({src})" if src else ""))
    print(f"starting A({ELEMENT}) = {start:.2f} (departure coefficients, babsma, first guess)")
    mass = MASS if MASS is not None else (parse_atm_name(cfg.yaml_atm.name).get("mass", 1.0) if cfg.yaml_atm else 1.0)

    in_file = load_lines(cfg.line_file, required=("wavelength",))
    lines = [r for r in in_file if r["species"].split()[0] == ELEMENT]
    if not lines:
        raise SystemExit(f"no {ELEMENT} lines in {cfg.line_file}; species in it: {sorted({r['species'] for r in in_file})}")
    manual = [r["line_id"] for r in lines if r["chi2_left"] is not None]
    print(f"{len(lines)} {ELEMENT} line(s), out of {len(in_file)} in {cfg.line_file.name}")
    print(f"{len(manual)} line(s) with a manual fit region (chi2_left / chi2_right in {cfg.line_file.name})")
    flags_file = resolve_path(LINE_FLAGS_FILE) if LINE_FLAGS_FILE else line_flags_path(cfg)
    flags = load_line_flags(flags_file)
    known = {r["line_id"] for r in lines}
    if set(flags) - known:
        raise SystemExit(f"{flags_file} names lines that are not in {cfg.line_file}: {sorted(set(flags) - known)}")
    n_rej = sum(is_reject(v) for v in flags.values())
    print(f"{len(flags) - n_rej} flagged line(s) (marked, still used), from {flags_file}")
    print(f"{n_rej} line(s) rejected by hand, from {flags_file}")
    if ONLY_LINES:
        if set(ONLY_LINES) - known:
            raise SystemExit(f"ONLY_LINES names lines that are not in {cfg.line_file}: {sorted(set(ONLY_LINES) - known)}")
        lines = [r for r in lines if r["line_id"] in ONLY_LINES]
    if ONLY_FLAGGED:
        if not flags:
            raise SystemExit(f"ONLY_FLAGGED is set but {flags_file} flags no lines")
        lines = [r for r in lines if r["line_id"] in flags]
    spectrum = load_spectrum(cfg.spectrum_file, cfg.rv)
    mode = "NLTE" if NLTE == "T" else "LTE"

    outdir = (RUNS_DIR / SELECTED_STAR / f"{ELEMENT.lower()}_abundance" /
              f"{time.strftime('%Y%m%d_%H%M%S')}_"
              f"T{teff:.0f}_g{logg:.2f}_v{vmic:.2f}_z{feh:+.2f}_{mode}{'_flagged' if ONLY_FLAGGED else ''}")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.yaml").write_text(yaml.dump(cfg.raw))
    here = Path(__file__)
    for f in (here, here.with_name("element_line_fit.py"), here.with_name("fe_line_fit.py"), here.with_name("interp_nlte_4.py")):
        shutil.copy2(f, outdir / f.name)
    save_line_flags(outdir / flags_file.name, flags)

    print(f"{SELECTED_STAR}: {len(lines)} {ELEMENT} lines, {mode} for {ELEMENT} (LTE for the rest), "
          f"Teff={teff:.0f} logg={logg:.2f} vmic={vmic:.2f} [Fe/H]={feh:+.2f}\n-> {outdir}")
    rows, info = run_point(cfg, lines, spectrum, teff, logg, vmic, feh, NLTE, mass, start, outdir, settings, flags,
                           note=RUN_NOTE)
    print_element_summary(info)
    for r in rows:
        print(f"    {r['line_id']:15s} A = {r['A']:.3f} +/- {r['sigma']:.3f}  chi2_red {r['chi2_red']:.2f}  "
              f"{r['selection']}  [{r['nlte_levels']}]")
    print(f"Results in {outdir}")


if __name__ == "__main__":
    main()
