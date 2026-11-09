#!/usr/bin/env python3
"""
Library: line-by-line abundances of any element with Turbospectrum, with that element in NLTE and all the others in LTE.
The script to run is find_element_abundance.py (next to this file).

The fit itself is the one of fe_line_fit.py (LineSynth, LineFitter, fit_lines, the quality cuts, the overview plot) and is
imported from there, so changes to the fitting apply to every element. bsyn treats an element in NLTE only if it is listed
in nlte_infofile.dat and LineSynth lists the fitted one alone: everything else, Fe included, is LTE. This file adds what
depends on the element:

  * load_element_config: the star's configuration with the element, its model atom and its line CSV
    (data/selected_lines/<star>/<el>_kept_lines.csv, same columns as the Fe files; only species and wavelength are needed),
  * start_abundance: A(X) for the departure coefficients, babsma and the first guess
    (DC_ABUNDANCE, else the reference file, else scaled solar),
  * make_atmosphere: atmosphere + departure coefficients through interp_nlte_4.py, reporting the abundance actually used
    at each of the eight corners of the grid (the interpolator does not interpolate in abundance),
  * nlte_level_report: which selected lines have model-atom levels in the line list. A line without them is computed in
    LTE even in an NLTE run (Mg I, Cr I, Mn II, Co II ... are poorly covered). Read-only, no level is ever invented,
  * analyze_element: per-ion means, [X/H], [X/Fe],
  * isotope mixtures: fixed fractions for bsyn's ISOTOPES block (default_isotope_fractions, ba_mix, eu_mix, isotope_fractions).

A later fit of isotope ratios: the mixture is fixed per LineSynth (cfg.isotope_fractions). Fitting f_odd or f151 needs the
mixture in the cache key and file stem of LineSynth.__call__ (one babsma, many bsyn runs) and a search over it.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from fe_line_fit import (ATOMIC_NUMBER, CONFIG_DIR, ELEMENTS, HOME, TS_DIR, _LL_HEADER, _LL_LEVELS, AtmosphereSpec,
                         FitSettings, StarConfig, _load_yaml, col, individual_abundances, is_reject, linfit,
                         load_star_config, parse_atm_name, quality_reasons, read_linelist_blocks, selection_kind)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

INTERP_SCRIPT = Path(__file__).resolve().parent / "interp_nlte_4.py"
INTERP_DIR = HOME / "data" / "original" / "NLTE"            # interpol_modeles_nlte (Fortran)
DC_GRID = INTERP_DIR / "dc"                                 # dc/<element>/dc.bin + master.dat
MARCS_SPH = HOME / "data" / "original" / "atm" / "MARCS_st_sph_t02_mod"
MARCS_PPL = HOME / "data" / "original" / "atm" / "MARCS_st_ppl_t01_mod"
LOGG_MAX_SPHERICAL = 3.5                                    # the spherical MARCS grid stops at logg = 3.5

# Turbospectrum's solar abundances (abund_source 'asp2007' = Grevesse, Asplund & Sauval 2007): sunabund_2007 in
# Turbospectrum_NLTE_20.1/source/makeabund.f, for Z = 1..92. Scale of the MARCS models and of [X/H]; -99 = no value.
SOLAR_TS = dict(zip(ELEMENTS, map(float, """
    12.00 10.93  1.05  1.38  2.70  8.39  7.78  8.66  4.56   7.84  6.17  7.53  6.37  7.51  5.36  7.14  5.50  6.18
     5.08  6.31  3.17  4.90  4.00  5.64  5.39  7.45  4.92   6.23  4.21  4.60  2.88  3.58  2.29  3.33  2.56  3.25
     2.60  2.92  2.21  2.58  1.42  1.92 -99.0  1.84  1.12   1.66  0.94  1.77  1.60  2.00  1.00  2.19  1.51  2.24
     1.07  2.17  1.13  1.70  0.58  1.45 -99.0  1.00  0.52   1.11  0.28  1.14  0.51  0.93  0.00  1.08  0.06  0.88
    -0.17  1.11  0.23  1.25  1.38  1.64  1.01  1.13  0.90   2.00  0.65 -99.0 -99.0 -99.0 -99.0 -99.0 -99.0  0.06
    -99.0 -0.52""".split())))
ALPHA_ELEMENTS = ("O", "Ne", "Mg", "Si", "S", "Ar", "Ca", "Ti")   # what ALPHA/Fe scales (ielalpha in makeabund.f)
ROMAN = {1: "I", 2: "II", 3: "III"}

COMPONENT_DW = 0.1                   # [A] line-list components within this of a selected line belong to it (HFS, isotopes) ...
COMPONENT_DEP = 0.01                 # [eV] ... if their excitation potential agrees within this


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def marcs_alpha(feh: float) -> float:
    """[alpha/Fe] of the standard MARCS composition, what the atmosphere interpolation uses."""
    return 0.0 if feh >= 0.0 else (0.4 if feh <= -1.0 else -0.4 * feh)


def scaled_solar(element: str, feh: float) -> float:
    """A(X) that Turbospectrum gives an element that is not listed individually, for a standard MARCS composition."""
    return SOLAR_TS[element] + feh + (marcs_alpha(feh) if element in ALPHA_ELEMENTS else 0.0)


def has_dc_grid(element: str) -> bool:
    return all((DC_GRID / element / name).exists() for name in ("dc.bin", "master.dat"))


def load_element_config(star: str, element: str, nlte: bool = True) -> StarConfig:
    """load_star_config(star), set up for fitting `element`: atomic number, model atom (config/nlte.yaml), its line CSV,
    and the no-element profile built from the line lists themselves. An NLTE run needs an entry in config/nlte.yaml and a
    grid in data/original/NLTE/dc/<element>/."""
    if element not in ATOMIC_NUMBER:
        raise SystemExit(f"unknown element {element!r}")
    cfg = load_star_config(star)
    entry = (_load_yaml(CONFIG_DIR / "nlte.yaml").get("element") or {}).get(element)
    if nlte and not entry:
        raise SystemExit(f"{element} has no entry in config/nlte.yaml (model_atom, model_atom_path), needed for NLTE")
    if nlte and not has_dc_grid(element):
        raise SystemExit(f"no departure coefficients for {element}: {DC_GRID / element} needs dc.bin and master.dat")
    changes = dict(element=element, Z=ATOMIC_NUMBER[element], no_fe_linelists=None, strip_fitted_lines=True,
                   line_file=HOME / "data" / "selected_lines" / star / f"{element.lower()}_kept_lines.csv")
    if entry:
        changes |= dict(model_atom=entry["model_atom"], atom_path=Path(entry["model_atom_path"]))
    return dataclasses.replace(cfg, **changes)


def start_abundance(cfg: StarConfig, feh: float, override: float | None = None) -> float:
    """A(X) of the departure coefficients, of babsma and of the first guess: `override`, else the value of the reference
    file (same reference_scaling as the other elements) or of ABUNDANCE_OVERRIDES, else scaled solar."""
    if override is not None:
        return float(override)
    listed = individual_abundances(dataclasses.replace(cfg, element="H"), feh)   # "H": not the fitted element, keep it in
    return float(listed[cfg.element]) if cfg.element in listed else scaled_solar(cfg.element, feh)


# ---------------------------------------------------------------------------
# Isotopes
# ---------------------------------------------------------------------------

def default_isotope_fractions(element: str) -> dict[int, float]:
    """Mass number -> isotopic fraction of an element as Turbospectrum has it (isotopfrac in makeabund.f)."""
    z = ATOMIC_NUMBER[element]
    text = (TS_DIR / "source" / "makeabund.f").read_text(errors="replace")
    return {int(m[1]): float(m[2]) for m in re.finditer(rf"^\s+isotopfrac\({z},(\d+)\)=([\d.]+)", text, re.M)}


def ba_mix(f_odd: float) -> dict[int, float]:
    """Ba isotopes for f_odd = N(135Ba + 137Ba) / N(Ba), keeping Turbospectrum's ratios 135:137 and 134:136:138
    (130Ba and 132Ba stay as they are)."""
    d = default_isotope_fractions("Ba")
    room = 1.0 - d[130] - d[132]
    if not 0.0 <= f_odd <= room:
        raise ValueError(f"f_odd must be between 0 and {room:.5f}, not {f_odd}")
    odd, even = d[135] + d[137], d[134] + d[136] + d[138]
    return {135: f_odd * d[135] / odd, 137: f_odd * d[137] / odd,
            **{a: (room - f_odd) * d[a] / even for a in (134, 136, 138)}}


def eu_mix(f151: float) -> dict[int, float]:
    """Eu isotopes for f151 = N(151Eu) / N(Eu)."""
    if not 0.0 <= f151 <= 1.0:
        raise ValueError(f"f151 must be between 0 and 1, not {f151}")
    return {151: f151, 153: 1.0 - f151}


def isotope_fractions(mixtures: dict[str, dict[int, float]]) -> dict[str, float]:
    """{"Ba": {135: f, ...}, ...} -> bsyn codes {"56.135": f, ...}. What is given replaces Turbospectrum's fraction of that
    isotope, the other isotopes of the element keep theirs. The fractions of an element must add up to 1 within 0.01."""
    out = {}
    for element, given in mixtures.items():
        merged = default_isotope_fractions(element) | {int(a): float(f) for a, f in given.items()}
        total = sum(merged.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"the isotopic fractions of {element} add up to {total:.4f}, not 1: {merged}")
        out |= {f"{ATOMIC_NUMBER[element]}.{a:03d}": f for a, f in sorted(merged.items())}
    return out


# ---------------------------------------------------------------------------
# Atmosphere and departure coefficients
# ---------------------------------------------------------------------------

def make_atmosphere(teff: float, logg: float, feoh: float, element: str, abund: float, mass: float, vmic: float,
                    outdir: Path, nlte: bool = True) -> tuple[AtmosphereSpec, dict]:
    """Interpolate the atmosphere and the departure coefficients of `element` at A(X) = abund with interp_nlte_4.py, in a
    fresh scratch directory. Returns the AtmosphereSpec and what the interpolator reported (its DC_INFO line: the
    abundance used at each corner, and which corners have nothing within 0.1 dex of A(X)). An LTE run of an element without
    a grid only needs the atmosphere: it goes through the Fe grid (the Fe departure coefficients are not used)."""
    if mass > 0 and logg > LOGG_MAX_SPHERICAL:
        raise SystemExit(f"logg = {logg} is beyond the spherical MARCS grid (max {LOGG_MAX_SPHERICAL}); "
                         f"use MASS = 0.0 for plane-parallel models")
    outdir.mkdir(parents=True, exist_ok=True)
    grid_element = element
    if not nlte and not has_dc_grid(element):
        grid_element, abund = "Fe", SOLAR_TS["Fe"] + feoh
    for f in (INTERP_SCRIPT, INTERP_DIR / "interpol_modeles_nlte", MARCS_SPH, MARCS_PPL, DC_GRID / grid_element):
        if not f.exists():
            raise SystemExit(f"{f} not found: check the paths at the top of element_line_fit.py")
    env = dict(os.environ, INTERP_OPATH=str(outdir), INTERP_PLOT="0", INTERP_INTERACTIVE="0", MPLBACKEND="Agg")
    env["PATH"] = os.pathsep.join([str(INTERP_DIR), str(TS_DIR / "interpolator"), env.get("PATH", "")])
    cmd = [sys.executable, str(INTERP_SCRIPT), f"{mass:.1f}", str(int(round(teff))), f"{logg:.2f}", f"{feoh:.2f}",
           grid_element, f"{abund:.2f}"]
    with tempfile.TemporaryDirectory(prefix="interp_") as tmp:
        tmp = Path(tmp)
        (tmp / "st").mkdir()
        (tmp / "st" / "s").symlink_to(MARCS_SPH)
        (tmp / "st" / "p").symlink_to(MARCS_PPL)
        (tmp / "dc").symlink_to(DC_GRID)
        proc = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True)
    ints, dcs = sorted(outdir.glob("*.int")), sorted(outdir.glob("*_dc.dat"))
    info = next((json.loads(ln[len("DC_INFO "):]) for ln in proc.stdout.splitlines() if ln.startswith("DC_INFO ")), None)
    if proc.returncode != 0 or len(ints) != 1 or len(dcs) != 1 or info is None:
        raise RuntimeError(
            f"interpolation failed for Teff={teff:.0f} logg={logg:.2f} [Fe/H]={feoh:.2f} A({grid_element})={abund:.2f}\n"
            f"command: {' '.join(cmd)}\n--- stdout (tail) ---\n{proc.stdout[-1500:]}\n--- stderr (tail) ---\n{proc.stderr[-1500:]}")
    p = parse_atm_name(ints[0].name)
    atm = AtmosphereSpec(atm=ints[0], dc=dcs[0], feoh=p["feoh"], alpha_fe=p["alpha_fe"], vmic=float(vmic),
                         teff=float(teff), logg=float(logg))
    return atm, info


def describe_dc(info: dict, element: str) -> list[str]:
    """Readable lines about the departure coefficients actually used (empty when there is nothing to say)."""
    if info["element"] != element or info["abundances_per_model"] <= 1:
        return []
    used = info["used"]
    out = [f"departure coefficients: A({element}) requested {info['requested']:.2f}, used at the corners of the grid: "
           f"{min(used):+.2f} .. {max(used):+.2f} (the closest abundance of each model, not interpolated in abundance)"]
    far = list(dict.fromkeys((c, a) for c, a, off in zip(info["corners"], used, info["far"]) if off))
    if far:
        out.append(f"WARNING: no departure coefficients within 0.1 dex of A({element}) = {info['requested']:.2f} at {len(far)} of "
                   f"the {len(set(info['corners']))} models (outside their range, or a gap); the closest abundance was used there:")
        out += [f"    {c}: A({element}) = {a:+.2f}" for c, a in far]
    return out


# ---------------------------------------------------------------------------
# NLTE levels of the selected lines
# ---------------------------------------------------------------------------

def _is_atomic_linelist(path: Path) -> bool:
    """True for a list that starts with an atomic species heavier than H (not hydrogen, not molecules)."""
    with Path(path).open(errors="replace") as f:
        first = next((_LL_HEADER.match(t) for t in f if t.startswith("'")), None)
    return first is not None and 1.5 <= float(first[1]) < 100.0


def nlte_level_report(cfg: StarConfig, lines: list[dict]) -> dict[str, dict]:
    """Read-only: for each selected line, the line-list components of the element with the same ion, within
    COMPONENT_DW and with the same excitation potential (hyperfine and isotope components). bsyn computes a component in
    NLTE only if the line list gives both of its levels as model-atom numbers. line_id -> {levels_used, how, ...}. A missing
    excitation potential in `lines` is filled in from the strongest component."""
    found: dict[str, list] = {r["line_id"]: [] for r in lines}
    by_ion: dict[int, list] = {}
    for r in lines:
        by_ion.setdefault(r["ion"], []).append(r)
    for path in cfg.linelists:
        if not _is_atomic_linelist(path):
            continue
        for header, _, data in read_linelist_blocks(path):
            h = _LL_HEADER.match(header)
            if int(float(h[1])) != cfg.Z or int(h[2]) not in by_ion:
                continue
            for ln in data:
                t = ln.split(None, 3)
                wl, ep, gf = float(t[0]), float(t[1]), float(t[2])
                for r in by_ion[int(h[2])]:
                    if abs(wl - r["wavelength"]) <= COMPONENT_DW:
                        m = _LL_LEVELS.match(ln)
                        found[r["line_id"]].append((gf, ep, int(m["lo"]) if m else 0, int(m["up"]) if m else 0))
    report = {}
    for r in lines:
        comps, note = found[r["line_id"]], ""
        if np.isfinite(r["excitation_potential"]):
            same_ep = [c for c in comps if abs(c[1] - r["excitation_potential"]) <= COMPONENT_DEP]
            if comps and not same_ep:
                note = f" (matched by wavelength only: no EP of {r['excitation_potential']:.3f} eV within {COMPONENT_DW} A)"
            comps = same_ep or comps
        elif comps:
            r["excitation_potential"] = max(comps)[1]
        pairs = sorted({f"{c[2]} {c[3]}" for c in comps if c[2] > 0 and c[3] > 0})
        n, n_lev = len(comps), sum(1 for c in comps if c[2] > 0 and c[3] > 0)
        if not comps:
            used, how = "LTE", "not found in the line lists"
        elif n_lev == n:
            used, how = ", ".join(pairs), f"NLTE: {n}/{n} components with both levels in the line list"
        elif n_lev:
            used, how = ", ".join(pairs), f"partly NLTE: {n_lev}/{n} components with both levels, the others LTE"
        else:
            used, how = "LTE", f"LTE: none of the {n} component(s) has both levels in the line list"
        report[r["line_id"]] = dict(line_id=r["line_id"], species=r["species"], wavelength=r["wavelength"], components=n,
                                    with_levels=n_lev, levels_used=used, how=how + note)
    return report


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def analyze_element(rows: list[dict], s: FitSettings, flags: dict[str, str] | None, element: str, feh: float) -> dict:
    """Per ion: mean abundance of the lines that pass the quality cuts (quality_reasons, as for Fe), optionally sigma-clipped
    around the mean (settings.sigma_clip, with at least 5 lines), scatter, error of the mean (scatter / sqrt(N); the fit
    error of the line when N = 1), [X/H] and [X/Fe] = [X/H] - feh, on Turbospectrum's solar scale. Slopes against EP and
    REW (at least 4 lines) are for information. Keys: A1 s1 e1 n1 n1_clipped xh1 xfe1 ... for ion 1, and so on; `selection`
    and `flag` are aligned with rows, as in analyze()."""
    selection = quality_reasons(rows, s, flags)
    ion, A, sig = col(rows, "ion"), col(rows, "A"), col(rows, "sigma")
    solar = SOLAR_TS[element]
    nan = float("nan")
    ions = sorted({int(i) for i in ion})
    info: dict = dict(element=element, solar=solar, feh=feh, ions=ions, n_total=len(rows))
    for k in ions:
        idx = np.flatnonzero((ion == k) & np.array([sel == "" for sel in selection]))
        keep, res, sd = np.ones(len(idx), bool), np.zeros(len(idx)), np.inf
        if s.sigma_clip and len(idx) >= 5:
            for _ in range(5):
                res, sd = A[idx] - A[idx][keep].mean(), A[idx][keep].std(ddof=1)
                new = np.abs(res) < s.clip_sigma * sd
                if new.sum() < 3 or (new == keep).all():
                    break
                keep = new
        for j, i in enumerate(idx):
            selection[i] = "used" if keep[j] else f"sigma-clipped ({res[j] / sd:+.1f} sigma from the mean)"
        a, n = A[idx][keep], int(keep.sum())
        mean = float(a.mean()) if n else nan
        scatter = float(a.std(ddof=1)) if n >= 2 else nan
        err = scatter / np.sqrt(n) if n >= 2 else (float(sig[idx][keep][0]) if n == 1 else nan)
        info.update({f"A{k}": mean, f"s{k}": scatter, f"e{k}": float(err), f"n{k}": n, f"n{k}_clipped": int(len(idx) - n),
                     f"xh{k}": mean - solar, f"xfe{k}": mean - solar - feh})
        for name in ("ep", "rew"):
            x = col(rows, name)[idx][keep]
            good = np.isfinite(x)
            fit = linfit(x[good], a[good]) if good.sum() >= 4 and np.ptp(x[good]) > 0 else (nan, nan, nan)
            info.update({f"slope_{name}{k}": fit[0], f"icpt_{name}{k}": fit[1], f"slope_{name}{k}_err": fit[2]})
    info["dA"] = info["A2"] - info["A1"] if {1, 2} <= set(ions) else nan
    info["dA_err"] = float(np.hypot(info["e1"], info["e2"])) if {1, 2} <= set(ions) else nan
    ids = [r["line_id"] for r in rows]
    flag = [(flags or {}).get(i, "") for i in ids]
    by_hand = [sel.startswith("rejected: by hand") for sel in selection]
    info.update(n_flagged=sum(bool(f) and not is_reject(f) for f in flag), n_rejected_by_hand=sum(by_hand),
                n_rejected=sum(sel.startswith("rejected") and not h for sel, h in zip(selection, by_hand)),
                used_ids=[i for i, sel in zip(ids, selection) if sel == "used"],
                clipped_ids=[i for i, sel in zip(ids, selection) if sel.startswith("sigma-clipped")],
                flagged_ids=[i for i, f in zip(ids, flag) if f and not is_reject(f)],
                rejected_by_hand_ids=[i for i, h in zip(ids, by_hand) if h],
                max_synth_ids=[r["line_id"] for r in rows if r.get("stop_reason") == "max_synth"],
                selection=selection, flag=flag)
    return info


def main_ion(info: dict) -> int | None:
    """The ion with the most used lines (the first such one), None if no line is used."""
    return max(info["ions"], key=lambda k: (info[f"n{k}"], -k)) if any(info[f"n{k}"] for k in info["ions"]) else None


def element_totals_text(info: dict) -> str:
    """The per-ion totals of a run on one line (info from analyze_element, or `results` of summary.yaml)."""
    el, parts = info["element"], []
    for k in info["ions"]:
        name = f"{el} {ROMAN.get(k, k)}"
        if info.get(f"n{k}"):
            sd = info.get(f"s{k}", float("nan"))
            scatter = f"sd {sd:.3f}, " if np.isfinite(sd) else ""
            parts.append(f"A({name}) = {info[f'A{k}']:.3f} +/- {info[f'e{k}']:.3f} ({scatter}N = {info[f'n{k}']})  "
                         f"[{el}/H] = {info[f'xh{k}']:+.3f}  [{el}/Fe] = {info[f'xfe{k}']:+.3f}")
        else:
            parts.append(f"A({name}) = n/a")
    return "     ".join(parts)


def print_element_summary(info: dict) -> None:
    el = info["element"]
    print(f"  solar A({el}) = {info['solar']:.2f} (Turbospectrum, Grevesse et al. 2007), [Fe/H] = {info['feh']:+.2f}")
    for k in info["ions"]:
        name = f"{el} {ROMAN.get(k, k):3s}"
        if not info[f"n{k}"]:
            print(f"  {name}: no line used")
            continue
        sd = f"scatter {info[f's{k}']:.3f}, " if np.isfinite(info[f"s{k}"]) else ""
        print(f"  {name}: A = {info[f'A{k}']:.3f} +/- {info[f'e{k}']:.3f} ({sd}{info[f'n{k}']} lines, {info[f'n{k}_clipped']} "
              f"sigma-clipped)   [{el}/H] = {info[f'xh{k}']:+.3f}   [{el}/Fe] = {info[f'xfe{k}']:+.3f}")
    if np.isfinite(info["dA"]):
        print(f"  A({el} II) - A({el} I) = {info['dA']:+.3f} +/- {info['dA_err']:.3f}")
    print("  selection of lines (see `selection` in lines.csv):")
    print(f"  used lines:                           {len(info['used_ids'])} / {info['n_total']}")
    print(f"  sigma-clipped:                        {len(info['clipped_ids'])}")
    print(f"  flagged by hand (marked, still used): {info['n_flagged']}")
    print(f"  rejected by hand:                     {info['n_rejected_by_hand']}")
    print(f"  rejected by quality cuts:             {info['n_rejected']}")
    if info["max_synth_ids"]:
        print(f"  NOTE: maximum number of syntheses reached (not converged to a_tol): {', '.join(info['max_synth_ids'])}")


def plot_element_diagnostics(rows: list[dict], info: dict, path: Path, title: str = "") -> None:
    """Optional (MAKE_DIAGNOSTICS): A(X) against EP, REW and wavelength, per ion. Filled marks = used (circles for the
    neutral atom, squares for the ion), x = sigma-clipped, open grey = rejected; dotted lines = the means."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    el = info["element"]
    ion, A, sig = col(rows, "ion"), col(rows, "A"), col(rows, "sigma")
    kind = np.array([selection_kind(sel) for sel in info["selection"]])
    used = kind == "used"
    finite = A[np.isfinite(A) & used] if used.any() else A[np.isfinite(A)]
    if finite.size == 0:
        return
    lo, hi = finite.min() - 0.25, finite.max() + 0.25
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    ion_style = {1: ("tab:blue", "o"), 2: ("tab:red", "s"), 3: ("tab:green", "^")}
    for ax, (name, label) in zip(axes, (("ep", "Excitation potential [eV]"), ("rew", "log(EW/lambda)"),
                                        ("wavelength", "Wavelength [A]"))):
        x = col(rows, name)
        if not np.isfinite(x).any():                         # a minimal line CSV has no equivalent widths
            ax.text(0.5, 0.5, "no equivalent widths in the line CSV", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
        for k in info["ions"]:
            color, marker = ion_style.get(k, ("tab:gray", "D"))
            sel = used & (ion == k)
            ax.errorbar(x[sel], A[sel], yerr=sig[sel], fmt=marker, color=color, ms=5, lw=0.8,
                        label=f"{el} {ROMAN.get(k, k)} used")
            if info.get(f"n{k}"):
                ax.axhline(info[f"A{k}"], color=color, ls=":", lw=1.0,
                           label=f"mean {el} {ROMAN.get(k, k)}: {info[f'A{k}']:.3f}")
        for kd, st in (("sigma-clipped", dict(marker="x", color="0.35", ms=6, label="sigma-clipped")),
                       ("rejected", dict(marker="o", mfc="none", mec="0.6", ms=6, label="rejected"))):
            sel = (kind == kd) & np.isfinite(A) & np.isfinite(x)
            if sel.any():
                ax.plot(x[sel], np.clip(A[sel], lo, hi), ls="none", **st)
        ax.set_xlabel(label)
        ax.set_ylim(lo, hi)
        ax.grid(color="0.92", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(f"A({el})")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(f"{title}\n{element_totals_text(info)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
