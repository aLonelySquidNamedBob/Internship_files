#!/usr/bin/env python3
"""
Library: fit every Fe line of a line list (NLTE Turbospectrum) for ONE model atmosphere.
The script to run is find_stellar_parameters (next to this file).

Needs numpy, scipy and pyyaml (matplotlib is optional and only used for plots). No pandas.
This one file also holds everything that used to be in ts_common.py: paths, yaml loading and
the babsma/bsyn/faltbon wrappers.

Usage:

    from fe_line_fit import fit_lines, analyze
    rows = fit_lines(lines, atm, cfg, spectrum, outdir)   # list of dicts, one per line
    info = analyze(rows)                                   # mean A(Fe I/II), slopes, ...

Per line:
  * the fit region is the line centre +/- FitSettings.half_window (0.4 A), so it contains continuum on
    both sides, or the bounds chi2_left / chi2_right given for that line in the line CSV (blank = default);
    all its pixels are fitted, neighbouring features included (fit_left/fit_right from the CSV are only
    drawn in the plots),
  * A(Fe) is found by a few bsyn runs + parabolic chi^2 minimisation (4-6 syntheses
    when warm-started from a previous result, ~6-8 from scratch),
  * at every trial abundance a small RV shift (fit_rv, else rv_fixed) and an extra broadening are refitted on the stored
    synthetic spectrum. The broadening is done by faltbon (Gaussian = profile 2 or radial-tangential =
    profile 3; free per line within broad_min..broad_max, or fixed for all lines with fit_broadening=False
    and FitSettings.broadening), the
    instrumental profile by faltbon too. The continuum is fixed at 1 unless
    FitSettings.fit_continuum is set,
  * the pixel noise is 1/SNR, with the SNR from `spectra: snr:` in stars.yaml (default 100),
  * sigma(A) comes from the chi^2 curvature (delta chi^2 = 1, rescaled by
    sqrt(reduced chi^2) if the fit is poor).

Lines are processed in parallel; each one works in its own scratch directory.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per worker process

import csv
import dataclasses
import multiprocessing
import re
import shutil
import subprocess
import tempfile
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import minimize_scalar
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

HOME = Path("~/NICO").expanduser()              # project root
CONFIG_DIR = HOME / "config"
RUNS_DIR = HOME / "runs"
SCRATCH_DIR = Path(tempfile.gettempdir())       # short paths: Fortran dislikes very long file names

TS_DIR = HOME / "Turbospectrum_NLTE_20.1"
TS_DATA = TS_DIR / "DATA"                       # symlinked as ./DATA in every scratch directory
BABSMA = TS_DIR / "exec-gf" / "babsma_lu"
BSYN = TS_DIR / "exec-gf" / "bsyn_lu"
FALTBON = TS_DIR / "Utilities" / "faltbon"

SOLAR_FE = 7.50          # only used to convert A(Fe) <-> [Fe/H]
DW = 0.01                # synthesis wavelength step [A]
SPHERICAL_RT = False     # bsyn 'SPHERICAL:' flag. Your old scripts use F, even with spherical MARCS models.
DEFAULT_VMIC = 1.0
C_KMS = 299792.458
FALTBON_PROFILE = {"gauss": 2, "radtan": 3}   # faltbon profile types of the extra broadening
MIN_BROAD_KMS = 0.1      # below this the extra broadening is skipped (kernel narrower than a pixel)
REFERENCE_ABUNDANCE_FILE = CONFIG_DIR / "reference_abundance.yaml"   # published A(X) per star (keys as stars.yaml)

ELEMENTS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
            "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb "
            "Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U").split()
ATOMIC_NUMBER = {el: z for z, el in enumerate(ELEMENTS, 1)}


def resolve_path(p) -> Path:
    """Absolute paths pass through; relative ones are taken relative to ~/NICO."""
    p = Path(str(p)).expanduser()
    return p if p.is_absolute() else HOME / p


def parse_atm_name(name: str) -> dict:
    """teff, logg, mass, geometry, feoh, alpha_fe from a MARCS file name."""
    out: dict = {}
    m = re.match(r"([sp])(\d+)_g([+-]\d+\.\d+)_m(\d+\.\d+)", name)
    if m:
        out.update(geometry=m[1], teff=float(m[2]), logg=float(m[3]), mass=float(m[4]))
    for key, tag in (("feoh", "z"), ("alpha_fe", "a")):
        m = re.search(rf"_{tag}([+-]\d+\.\d+)", name)
        if m:
            out[key] = float(m[1])
    return out


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StarConfig:
    star: str
    raw: dict
    element: str
    Z: int
    model_atom: str
    atom_path: Path
    linelists: list
    spectrum_file: Path | None
    line_file: Path
    rv: float
    snr: float
    inst_fwhm: float
    yaml_atm: Path | None
    yaml_dc: Path | None
    feoh: float | None
    alpha_fe: float | None
    abu_c: float | None            # absolute A(C) = log eps(C), H = 12; None: scaled solar
    vmic: float
    nlte_abundance: float | None
    # abundances of the other elements, see individual_abundances()
    reference_abundances: dict = field(default_factory=dict)  # element -> A(X) from reference_abundance.yaml
    reference_notes: dict = field(default_factory=dict)       # element -> `note` of that entry
    reference_scaling: str = "absolute"                       # "absolute" or "fe"
    abundance_overrides: dict = field(default_factory=dict)   # element -> A(X), or None = scaled solar


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def load_reference_abundances(star: str, path: Path = REFERENCE_ABUNDANCE_FILE) -> tuple[dict, dict]:
    """(element -> A(X), element -> note) of one star from the reference-abundance file. Entries that are null,
    or have a null value, are not measured and left out; a missing file or star gives empty dicts."""
    path = Path(path)
    if not path.exists():
        return {}, {}
    entry = (_load_yaml(path) or {}).get(star) or {}
    values, notes = {}, {}
    for el, v in (entry.get("abundances") or {}).items():
        if el not in ATOMIC_NUMBER:
            raise ValueError(f"{path}: unknown element {el!r} for {star}")
        if v is None or v.get("value") is None:
            continue
        values[el] = float(v["value"])
        if v.get("note"):
            notes[el] = str(v["note"])
    return values, notes


def individual_abundances(cfg: StarConfig, feoh: float) -> dict[str, float]:
    """A(X) (log eps, H = 12) given to Turbospectrum for the elements other than the fitted one, ordered by Z.
    Elements not listed stay at the scaled-solar value of the model (METALLICITY = feoh, ALPHA/Fe on top).
    Lowest to highest priority:
      1. cfg.reference_abundances (reference_abundance.yaml), with cfg.reference_scaling
           "absolute": A(X) as published,
           "fe":       A(X) shifted by feoh - [Fe/H]_reference, i.e. the published [X/Fe] is kept;
         Fe itself is never taken from there: it follows the model metallicity,
      2. abu_c from stars.yaml (C),
      3. cfg.abundance_overrides: a value, or None to put the element back to scaled solar.
    The fitted element (cfg.element) is always left out: LineSynth adds its trial value."""
    ref = dict(cfg.reference_abundances)
    if cfg.reference_scaling == "fe" and ref:
        if "Fe" not in ref:
            raise ValueError(f"{cfg.star}: reference_scaling 'fe' needs a reference A(Fe)")
        shift = feoh - (ref["Fe"] - SOLAR_FE)
        ref = {el: a + shift for el, a in ref.items()}
    elif cfg.reference_scaling not in ("absolute", "fe"):
        raise ValueError(f"reference_scaling must be 'absolute' or 'fe', not {cfg.reference_scaling!r}")
    ref.pop("Fe", None)
    if cfg.abu_c is not None:
        ref["C"] = cfg.abu_c
    for el, a in cfg.abundance_overrides.items():
        if el not in ATOMIC_NUMBER:
            raise ValueError(f"abundance override for unknown element {el!r}")
        if a is None:
            ref.pop(el, None)
        else:
            ref[el] = float(a)
    ref.pop(cfg.element, None)
    return dict(sorted(ref.items(), key=lambda kv: ATOMIC_NUMBER[kv[0]]))


def load_star_config(star: str) -> StarConfig:
    stars = _load_yaml(CONFIG_DIR / "stars.yaml")
    stars = stars.get("stars", stars)
    if star not in stars:
        raise SystemExit(f"{star!r} not in stars.yaml. Available: {list(stars)}")
    s = stars[star]
    syn = s["synthesis"]
    spectra = s.get("spectra") or {}

    nlte_cfg = _load_yaml(CONFIG_DIR / "nlte.yaml")
    ll_cfg = _load_yaml(CONFIG_DIR / "line_lists.yaml")
    elt = syn["NLTE_element"]
    el = nlte_cfg["element"][elt]

    linelists = [resolve_path(p) for p in ll_cfg["master"]]
    for mol in syn.get("molecular_line_lists") or []:
        linelists += [resolve_path(p) for p in ll_cfg["molecular"][mol]]

    normalized = [p for p in (spectra.get("normalized") or []) if p]
    dc_dir = syn.get("departure_coefficients_path")
    dc = syn.get("departure_coefficients")
    atm = syn.get("model_atmosphere")
    parsed = parse_atm_name(Path(atm).name) if atm else {}
    ref_values, ref_notes = load_reference_abundances(star)

    return StarConfig(
        star=star,
        raw=s,
        element=elt,
        Z=int(el["atomic_number"]),
        model_atom=el["model_atom"],
        atom_path=Path(el["model_atom_path"]),
        linelists=linelists,
        spectrum_file=HOME / normalized[0] if normalized else None,
        line_file=(HOME / s["selected_lines"] if s.get("selected_lines")
                   else HOME / "data" / "selected_lines" / star / "fe_kept_lines.csv"),
        rv=float(spectra.get("rv") or 0.0),
        snr=float(spectra.get("snr") or 100.0),
        inst_fwhm=float(spectra.get("instrument_fwhm") or -3.0),   # faltbon convention: <0 means km/s
        yaml_atm=HOME / atm if atm else None,
        yaml_dc=(HOME / dc_dir / dc if dc_dir else HOME / dc) if dc else None,
        feoh=float(syn["metallicity"]) if syn.get("metallicity") is not None else parsed.get("feoh"),
        alpha_fe=float(syn["alpha_fe"]) if syn.get("alpha_fe") is not None else parsed.get("alpha_fe"),
        abu_c=float(syn["abu_c"]) if syn.get("abu_c") is not None else None,    # None: scaled solar
        vmic=float(syn.get("vmic") or DEFAULT_VMIC),
        nlte_abundance=float(syn["NLTE_abundance"]) if syn.get("NLTE_abundance") is not None else None,
        reference_abundances=ref_values,
        reference_notes=ref_notes,
    )


@dataclass(frozen=True)
class AtmosphereSpec:
    atm: Path
    dc: Path
    feoh: float
    alpha_fe: float
    vmic: float
    teff: float | None = None      # informational only
    logg: float | None = None


# ---------------------------------------------------------------------------
# Turbospectrum wrappers
# ---------------------------------------------------------------------------

def run_program(program: Path, input_text: str, log_file: Path, cwd: Path) -> None:
    with log_file.open("a") as fh:
        try:
            subprocess.run([str(program)], input=input_text, text=True, stdout=fh,
                           stderr=subprocess.STDOUT, check=True, cwd=cwd)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"{program.name} exited with {exc.returncode} (log: {log_file})") from exc


class LineSynth:
    """babsma once for [wmin, wmax], then bsyn + faltbon per abundance (cached)."""

    def __init__(self, cfg: StarConfig, atm: AtmosphereSpec, wmin: float, wmax: float,
                 workdir: Path, log: Path):
        self.cfg, self.atm = cfg, atm
        self.wmin, self.wmax = round(wmin, 3), round(wmax, 3)
        self.workdir, self.log = workdir, log
        self.n_runs = 0
        self.abundances = individual_abundances(cfg, atm.feoh)
        self._cache: dict = {}
        self._bcache: dict = {}

        workdir.mkdir(parents=True, exist_ok=True)
        if not (workdir / "DATA").exists():
            (workdir / "DATA").symlink_to(TS_DATA)
        self.opac = workdir / "cont.opac"
        self.nlte_info = workdir / "nlte_infofile.dat"
        self.nlte_info.write_text(
            "#\n# path for model atom files\n"
            f"{cfg.atom_path}/\n"
            "# path for departure files\n"
            f"{atm.dc.parent}/\n"
            f"{cfg.Z} '{cfg.element}' 'nlte' '{cfg.model_atom}' '{atm.dc.name}' 'ascii'\n"
        )
        run_program(BABSMA, self._babsma_input(), log, workdir)

    def _individual_abundances(self, fe: float | None = None) -> str:
        """The 'INDIVIDUAL ABUNDANCES' block (absolute log eps, H = 12) shared by babsma and bsyn: the fitted
        element (the trial value, bsyn only) and every element of individual_abundances() (reference file,
        abu_c, overrides; both, so that the continuous opacity and molecular equilibrium of babsma agree with the
        line synthesis). Elements that are not listed are left at the scaled-solar value."""
        pairs = [(ATOMIC_NUMBER[el], a) for el, a in self.abundances.items()]
        if fe is not None:
            pairs = sorted(pairs + [(self.cfg.Z, fe)])
        if not pairs:
            return ""
        return f"'INDIVIDUAL ABUNDANCES:'   '{len(pairs)}'\n" + "".join(f"{z}  {v:.4f}\n" for z, v in pairs)

    def _babsma_input(self) -> str:
        a = self.atm
        return f"""\
'LAMBDA_MIN:'  '{self.wmin}'
'LAMBDA_MAX:'  '{self.wmax}'
'LAMBDA_STEP:' '{DW}'
'MODELINPUT:' '{a.atm}'
'MARCS-FILE:' '.false.'
'MODELOPAC:' '{self.opac}'
'METALLICITY:'    '{a.feoh}'
'ALPHA/Fe   :'    '{a.alpha_fe}'
'HELIUM     :'    '0.00'
'R-PROCESS  :'    '0.00'
'S-PROCESS  :'    '0.00'
{self._individual_abundances()}'XIFIX:' 'T'
{a.vmic}
"""

    def _bsyn_input(self, abu: float, nlte: str, result: Path) -> str:
        a, c = self.atm, self.cfg
        return f"""\
'NLTE :'          {nlte}
'NLTEINFOFILE:'   '{self.nlte_info}'
'LAMBDA_MIN:'     '{self.wmin}'
'LAMBDA_MAX:'     '{self.wmax}'
'LAMBDA_STEP:'    '{DW}'
'INTENSITY/FLUX:' 'Flux'
'MODELOPAC:' '{self.opac}'
'RESULTFILE :' '{result}'
'METALLICITY:'    '{a.feoh}'
'ALPHA/Fe   :'    '{a.alpha_fe}'
'HELIUM     :'    '0.00'
'R-PROCESS  :'    '0.00'
'S-PROCESS  :'    '0.00'
{self._individual_abundances(abu)}'NFILES   :' '{len(c.linelists)}'
""" + "\n".join(str(p) for p in c.linelists) + f"""
'SPHERICAL:'  '{"T" if SPHERICAL_RT else "F"}'
  30
  300.00
  15
  1.30
"""

    @staticmethod
    def _faltbon_input(spec: Path, out: Path, fwhm: float, profile: int) -> str:
        """fwhm < 0: km/s, > 0: mA. profile 2 = Gaussian (FWHM), 3 = radial-tangential (zeta_RT = 1.433 x FWHM)."""
        return f"""\
{spec}
{out}
{fwhm}  FWHM OF CONVOLU.PROFILE=  MILLIANGSTROM, OR KM/S IF < 0.
{profile} PROFILE TYPE=                (1=EXP, 2=GAUSS, 3=RAD-TAN, 4=ROT)
2 FLUX-SCALE = (1=REL.FLUX,  2=ABS.FLUX))
1 NBIN = (1 = NO IN BIN )
"""

    def __call__(self, abu: float, nlte: str = "T"):
        key = (round(float(abu), 4), nlte)
        if key not in self._cache:
            stem = f"A{key[0]:.4f}_{nlte}"
            spec, cvl = self.workdir / f"{stem}.spec", self.workdir / f"{stem}.cvl"
            run_program(BSYN, self._bsyn_input(key[0], nlte, spec), self.log, self.workdir)
            run_program(FALTBON, self._faltbon_input(spec, cvl, self.cfg.inst_fwhm, 2), self.log, self.workdir)
            self._cache[key] = np.loadtxt(cvl, usecols=(0, 1), unpack=True)
            self.n_runs += 1
        return self._cache[key]

    def broadened(self, abu: float, nlte: str, broad: float, profile: str):
        """The synthetic spectrum (after the instrumental profile) with the extra broadening on top, done by
        faltbon: `broad` is its FWHM argument in km/s (Gaussian FWHM, or 1/1.433 of zeta_RT for "radtan")."""
        wl, fl = self(abu, nlte)
        if broad < MIN_BROAD_KMS:
            return wl, fl
        key = (round(float(abu), 4), nlte, profile, round(float(broad), 3))
        if key not in self._bcache:
            stem = f"A{key[0]:.4f}_{nlte}"
            out = self.workdir / f"{stem}_{profile}_{key[3]:.3f}.bro"
            run_program(FALTBON, self._faltbon_input(self.workdir / f"{stem}.cvl", out, -key[3],
                                                     FALTBON_PROFILE[profile]), self.log, self.workdir)
            self._bcache[key] = np.loadtxt(out, usecols=(0, 1), unpack=True)
            out.unlink()
        return self._bcache[key]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def linfit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least squares y = a + b x. Returns slope, intercept, standard error of the slope."""
    r = linregress(x, y)
    return float(r.slope), float(r.intercept), float(r.stderr)


def col(rows: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows], float)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class FitSettings:
    n_jobs: int = 8                # parallel lines (you have 8 physical cores)
    half_window: float = 0.4       # default fit region = line centre +/- this [A] (per line overridden by the
                                   # chi2_left / chi2_right columns of the line CSV)
    synth_pad: float = 1.0         # synthesis range extends this far [A] beyond the fit window (RV shift and
                                   # broadening kernel need margin)
    a_step_first: float = 0.30     # initial A(Fe) bracket half-width, no previous result [dex]
    a_step_warm: float = 0.15      # ... when warm-started from a previous result
    a_tol: float = 0.004           # stop when the parabola minimum moves less than this [dex]
    a_half_range: float = 1.5      # A(Fe) is confined to a_guess +/- this
    max_synth: int = 20            # hard cap on bsyn runs per line
    fit_rv: bool = True            # True: fit a small RV shift per line (absorbs line-list wavelength errors)
    rv_max: float = 3.0            # km/s, search limit when fitted
    rv_fixed: float = 0.0          # km/s, the shift applied to every line when fit_rv is False (on top of the
                                   # star's `rv` from stars.yaml)
    # which pixels enter chi^2 (the plots and bestfit.csv always show the whole +/- half_window window);
    # all pixels of the region are used, nothing is masked
    fit_region: str = "window"     # "window": +/- half_window around the line centre
                                   # "line":   fit_left..fit_right from the line CSV
                                   # "core":   +/- core_half_width around the line centre
    core_half_width: float = 0.04  # [A]
    # extra broadening (macroturbulence etc.) on top of the instrumental profile, done by faltbon; the number
    # is faltbon's FWHM argument in km/s
    broad_profile: str = "radtan"  # "radtan": radial-tangential macroturbulence (faltbon profile 3, zeta_RT =
                                   # 1.433 x the number); "gauss": Gaussian (faltbon profile 2, the number is
                                   # the FWHM). With a Gaussian the observed cores of HD196944 come out
                                   # deeper than the model and the shoulders higher (a W in the residuals);
                                   # rad-tan removes that.
    fit_broadening: bool = True    # True: fitted per line within [broad_min, broad_max]
    broadening: float = 5.5        # km/s, the value used for every line when fit_broadening is False (same
                                   # meaning as the fitted number; median of the HD196944 fits is about 5.7)
    broad_min: float = 0.0
    broad_max: float = 12.0        # values at the limit mean the fit is trying to absorb something else (look
                                   # at the line). Lower it if you use a broader instrumental profile.
    # continuum
    fit_continuum: bool = False    # False: continuum fixed at 1 (trust the normalisation)
    cont_order: int = 0            # if fitted - 0: scale only, 1: scale + slope
    cont_prior: float = 0.01       # if fitted - sigma of a Gaussian prior on the local continuum (scale 1,
                                   # slope 0); <= 0 leaves it free. A free scale absorbs profile mismatches
                                   # when a window has little continuum.
    assign_nlte_levels: bool = True  # NLTE: give the selected lines missing model-atom level numbers (all Fe II
                                   # lines, a few Fe I) by energy, see "NLTE level identification" below
    no_fe_profile: bool = True     # also synthesise every line region with A(Fe) = no_fe_abundance (one extra bsyn run
                                   # per line) and show it, with the same broadening / RV shift / continuum as the
                                   # best fit, as the "no Fe" profile: whatever absorbs there is not iron (blends)
    no_fe_abundance: float = -20.0   # log eps(Fe) of that profile. Note that bsyn recomputes the electron pressure
                                   # with Fe removed, so other lines change slightly (see lines.csv: nofe_*)
    keep_synth: bool = True        # keep the synthetic spectra of every trial abundance in lines/<line>/synth/
                                   # (A<abund>_<T|F>.spec: bsyn output; .cvl: after the instrumental profile)
                                   # and the chi^2 at each of them in lines/<line>/chi2_curve.csv
    keep_files: bool = False       # keep all Turbospectrum scratch files of every line (incl. ~100 MB of log)
    make_plots: bool = True
    # line selection used by analyze()
    chi2_factor: float = 3.0       # drop lines with chi2_red > max(chi2_factor * median(chi2_red), 1); relative
                                   # so a too-high SNR in stars.yaml does not empty the sample, floored at 1 so
                                   # a too-low SNR does not throw out good strong lines (their chi2 is higher
                                   # than that of weak lines)
    sigma_clip: bool = True        # False: no sigma clipping of the Fe I lines (all lines that pass the quality
                                   # cuts are used; Fe II is never clipped)
    clip_sigma: float = 2.5        # clipping threshold (sigma of the residuals of the A = c + a*EP + b*REW fit)


# ---------------------------------------------------------------------------
# Input files
# ---------------------------------------------------------------------------

_NUMERIC = ("wavelength", "min_flux", "voigt_fit_depth", "voigt_fit_sigma", "voigt_fit_gamma",
            "voigt_fit_shift", "voigt_fit_area", "fit_left", "fit_right", "excitation_potential")


def load_lines(path: Path) -> list[dict]:
    """The selected-lines CSV as a list of dicts (numbers converted, plus `ion` and `line_id`). The optional
    columns chi2_left / chi2_right (absolute wavelengths, A) set the fit region of a line by hand; both blank
    or missing = None = the default region."""
    rows = []
    with Path(path).open(newline="") as f:
        for raw in csv.DictReader(f):
            r = {k.strip(): (v or "").strip() for k, v in raw.items() if k}
            for k in _NUMERIC:
                r[k] = float(r[k])
            r["ion"] = int(r["species"].split()[1])
            r["line_id"] = f"Fe{r['ion']}_{r['wavelength']:.4f}"
            lo, hi = r.get("chi2_left", ""), r.get("chi2_right", "")
            if bool(lo) != bool(hi):
                raise ValueError(f"{r['line_id']}: give both chi2_left and chi2_right, or neither ({path})")
            r["chi2_left"], r["chi2_right"] = (float(lo), float(hi)) if lo else (None, None)
            if lo and not float(lo) < r["wavelength"] < float(hi):
                raise ValueError(f"{r['line_id']}: chi2_left < line wavelength < chi2_right is required ({path})")
            rows.append(r)
    return rows


def line_region(row: dict, s: "FitSettings") -> tuple[float, float, float, float, str]:
    """(window_lo, window_hi, fit_lo, fit_hi, source) of a line. The window is what is observed, plotted and
    synthesised (centre +/- half_window, widened if a manual region reaches further); the fit bounds are the
    pixels that enter chi^2: the manual chi2_left / chi2_right if set, else per settings.fit_region."""
    w = row["wavelength"]
    lo, hi = w - s.half_window, w + s.half_window
    if row.get("chi2_left") is not None:
        return min(lo, row["chi2_left"]), max(hi, row["chi2_right"]), row["chi2_left"], row["chi2_right"], "manual"
    if s.fit_region == "line":
        return lo, hi, row["fit_left"], row["fit_right"], "fit_region=line"
    if s.fit_region == "core":
        return lo, hi, w - s.core_half_width, w + s.core_half_width, "fit_region=core"
    return lo, hi, lo, hi, "default"


def line_flags_path(cfg: StarConfig) -> Path:
    """The star's manual line flags (line_id: reason), next to its selected-lines CSV."""
    return cfg.line_file.with_name("fe_line_flags.yaml")


def load_line_flags(path: Path) -> dict[str, str]:
    path = Path(path)
    if not path.exists():
        return {}
    return {str(k): str(v) for k, v in (yaml.safe_load(path.read_text()) or {}).items()}


def save_line_flags(path: Path, flags: dict[str, str]) -> None:
    header = ("# Lines left out of the Fe abundance summary by find_stellar_parameters: line_id: reason.\n"
              "# Written by review_fits.py; can also be edited by hand.\n")
    body = yaml.safe_dump(dict(sorted(flags.items())), sort_keys=False, allow_unicode=True) if flags else ""
    Path(path).write_text(header + body)


def _is_number(t: str) -> bool:
    try:
        float(t)
        return True
    except ValueError:
        return False


def load_spectrum(path: Path, rv: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Normalised spectrum from a text/CSV file: a wavelength and a flux column, header optional.
    Columns are picked by header name (wave/lambda/wl; a normalised-flux column such as `normed_flux` is
    preferred over a plain `flux` column, which may hold raw counts), otherwise the first two."""
    head, head_idx = "", 0
    with Path(path).open() as f:
        for i, line in enumerate(f):
            s = line.strip()
            if s and not s.startswith("#"):
                head, head_idx = s, i
                break
    delim = "," if "," in head else (";" if ";" in head else None)     # None: any whitespace
    toks = [t.strip() for t in (head.split(delim) if delim else head.split())]
    has_header = not (len(toks) >= 2 and _is_number(toks[0]) and _is_number(toks[1]))
    names = [t.lower() for t in toks] if has_header else []

    def pick(keys, default):
        return next((i for i, n in enumerate(names) if any(k in n for k in keys)), default)

    wl_i, fl_i = pick(("wave", "lambda", "wl"), 0), pick(("norm",), pick(("flux",), 1))
    wl, fl = np.loadtxt(path, delimiter=delim, comments="#", skiprows=head_idx + (1 if has_header else 0),
                        usecols=(wl_i, fl_i), unpack=True)
    print(f"spectrum {Path(path).name}: {len(wl)} pixels, using columns "
          f"{names[wl_i] if names else wl_i} (wavelength) and {names[fl_i] if names else fl_i} (flux)")
    order = np.argsort(wl)
    return wl[order] / (1.0 + rv / C_KMS), fl[order]


# ---------------------------------------------------------------------------
# Fitting one line
# ---------------------------------------------------------------------------

@dataclass
class Data:
    mid: float
    half: float
    wl: np.ndarray
    fl: np.ndarray
    sig: np.ndarray


def model_and_chi2(wl_syn, fl_syn, d: Data, dv: float, order: int | None, prior: float):
    """Synthetic (already broadened) -> RV shift -> best continuum scale -> chi^2.
    order None: continuum fixed at 1. `prior` > 0 adds a Gaussian prior (scale = 1 +/- prior, higher terms =
    0 +/- prior) to the continuum."""
    m = np.interp(d.wl / (1.0 + dv / C_KMS), wl_syn, fl_syn)
    if order is None:
        return float(np.sum(((d.fl - m) / d.sig) ** 2)), m, np.array([1.0])
    x = (d.wl - d.mid) / d.half
    basis = np.vstack([m * x**k for k in range(order + 1)]).T
    lhs, rhs = basis / d.sig[:, None], d.fl / d.sig
    target = np.eye(order + 1)[0]
    if prior > 0:
        lhs, rhs = np.vstack([lhs, np.eye(order + 1) / prior]), np.concatenate([rhs, target / prior])
    coef = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    model = basis @ coef
    chi2 = np.sum(((d.fl - model) / d.sig) ** 2)
    if prior > 0:
        chi2 += np.sum(((coef - target) / prior) ** 2)
    return float(chi2), model, coef


def continuum(d: Data, coef: np.ndarray) -> np.ndarray:
    """The continuum polynomial of model_and_chi2 on d's pixels."""
    x = (d.wl - d.mid) / d.half
    return sum(c * x**k for k, c in enumerate(coef))


class LineFitter:
    """chi^2(A(Fe)) with the nuisance parameters (broadening, RV shift, optionally continuum) profiled out at
    each abundance: an outer 1-D search over the broadening (one faltbon call per value) around an inner 1-D
    search over the RV shift (cheap, on the broadened spectrum)."""

    def __init__(self, synth, data: Data, nlte: str, s: FitSettings, fixed_broad: float | None,
                 region: np.ndarray | None = None):
        if s.broad_profile not in FALTBON_PROFILE:
            raise ValueError(f"broad_profile must be one of {list(FALTBON_PROFILE)}, not {s.broad_profile!r}")
        self.synth, self.nlte, self.s = synth, nlte, s
        self.d_full = data                                  # all pixels; self.d holds the ones in the chi^2
        self.region = np.ones(len(data.wl), bool) if region is None else region
        self.d = Data(data.mid, data.half, data.wl[self.region], data.fl[self.region], data.sig[self.region])
        self.fixed_broad = fixed_broad
        self.order = s.cont_order if s.fit_continuum else None
        self.n_free = int(s.fit_rv) + int(fixed_broad is None)
        self.evals: dict[float, dict] = {}

    def _best_dv(self, abu: float, broad: float):
        """(dv, chi2, continuum coefficients) at this abundance and broadening."""
        wl_syn, fl_syn = self.synth.broadened(abu, self.nlte, broad, self.s.broad_profile)

        def chi2(dv):
            return model_and_chi2(wl_syn, fl_syn, self.d, dv, self.order, self.s.cont_prior)

        dv = self.s.rv_fixed
        if self.s.fit_rv:
            dv = float(minimize_scalar(lambda v: chi2(v)[0], bounds=(-self.s.rv_max, self.s.rv_max),
                                       method="bounded", options={"xatol": 1e-3}).x)
        c, _, coef = chi2(dv)
        return dv, c, coef

    def at(self, abu: float) -> dict:
        abu = round(float(abu), 4)
        if abu in self.evals:
            return self.evals[abu]
        if self.fixed_broad is None:
            broad = float(minimize_scalar(lambda b: self._best_dv(abu, round(float(b), 3))[1],
                                          bounds=(self.s.broad_min, self.s.broad_max), method="bounded",
                                          options={"xatol": 0.02}).x)
            broad = round(broad, 3)
        else:
            broad = float(self.fixed_broad)
        dv, chi2, coef = self._best_dv(abu, broad)
        self.evals[abu] = {"chi2": chi2, "dv": dv, "broad": broad, "cont": float(coef[0]), "coef": coef}
        return self.evals[abu]

    def model_full(self, abu: float, ev: dict) -> np.ndarray:
        """The model at this abundance on all pixels of the window, for plots and bestfit.csv."""
        wl_syn, fl_syn = self.synth.broadened(abu, self.nlte, ev["broad"], self.s.broad_profile)
        m = model_and_chi2(wl_syn, fl_syn, self.d_full, ev["dv"], None, 0.0)[1]
        return m * continuum(self.d_full, ev["coef"])

    def run(self, a_guess: float, warm: bool) -> dict:
        s = self.s
        lo, hi = a_guess - s.a_half_range, a_guess + s.a_half_range
        step = s.a_step_warm if warm else s.a_step_first

        if len(self.d.wl) < 5:
            raise RuntimeError(f"only {len(self.d.wl)} pixels in the fit region")
        for a in (a_guess - step, a_guess, a_guess + step):
            self.at(np.clip(a, lo, hi))

        while len(self.evals) < s.max_synth:
            xs = np.array(sorted(self.evals))
            ys = np.array([self.evals[x]["chi2"] for x in xs])
            i = int(np.argmin(ys))
            if i == 0:
                a_new = xs[0] - 2 * step
            elif i == len(xs) - 1:
                a_new = xs[-1] + 2 * step
            else:
                c2, c1, _ = np.polyfit(xs[i - 1:i + 2], ys[i - 1:i + 2], 2)
                if c2 <= 0:                       # not convex: keep walking downhill
                    a_new = xs[i - 1] - step if ys[i - 1] < ys[i + 1] else xs[i + 1] + step
                else:
                    a_new = -c1 / (2 * c2)
                    if abs(a_new - xs[i]) < s.a_tol:
                        break
            a_new = round(float(np.clip(a_new, lo, hi)), 4)
            if a_new in self.evals:               # ran into the search boundary
                break
            self.at(a_new)

        xs = np.array(sorted(self.evals))
        ys = np.array([self.evals[x]["chi2"] for x in xs])
        i = int(np.argmin(ys))
        bracketed = 0 < i < len(xs) - 1
        curvature = np.nan
        a_best = float(xs[i])
        if bracketed:
            c2, c1, _ = np.polyfit(xs[i - 1:i + 2], ys[i - 1:i + 2], 2)
            if c2 > 0:
                a_best, curvature = float(-c1 / (2 * c2)), float(c2)
            else:
                bracketed = False

        n = len(self.d.wl)
        n_cont = s.cont_order + 1 if self.order is not None else 0
        dof = max(n - n_cont - self.n_free - 1, 1)
        a_near = float(xs[np.argmin(np.abs(xs - a_best))])
        nearest = self.evals[a_near]
        chi2_red = float(nearest["chi2"] / dof)
        sigma = (1.0 / np.sqrt(curvature)) * np.sqrt(max(chi2_red, 1.0)) if bracketed else np.nan
        at_broad_limit = self.fixed_broad is None and (nearest["broad"] >= s.broad_max - 0.05 or
                                                       nearest["broad"] <= s.broad_min + 0.05)
        model_nofe, nofe_status = np.full(len(self.d_full.wl), np.nan), "off"
        if s.no_fe_profile:
            try:
                model_nofe, nofe_status = self.model_full(s.no_fe_abundance, nearest), "ok"
            except Exception as exc:                          # noqa: BLE001 - the fit itself is fine
                nofe_status = f"failed: {type(exc).__name__}: {exc}"[:120]
        return dict(A=a_best, sigma=float(sigma), chi2_red=chi2_red, dv_kms=nearest["dv"],
                    model_nofe=model_nofe, no_fe_status=nofe_status,
                    broad_kms=nearest["broad"], broad_at_limit=bool(at_broad_limit), cont=nearest["cont"],
                    n_synth=len(self.evals), at_boundary=not bracketed,
                    model=self.model_full(a_near, nearest), used=self.region.copy())


def _plot_fit(path: Path, res: dict, data: Data, model: np.ndarray, row: dict, used: np.ndarray,
              nofe: np.ndarray | None = None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.axvspan(row["fit_left"], row["fit_right"], color="0.9", zorder=0)     # line window from the CSV
    ax.axvline(data.mid, color="0.6", lw=0.8, zorder=0)
    for b in (res["chi2_left"], res["chi2_right"]):                          # the fit region
        ax.axvline(b, color="tab:blue", lw=0.8, ls="--", zorder=0)
    ax.plot(data.wl[used], data.fl[used], "k.", ms=3)
    ax.plot(data.wl[~used], data.fl[~used], ".", color="0.7", ms=3, label="not in the fit")
    ax.plot(data.wl, model, "r-")
    if nofe is not None and np.isfinite(nofe).any():
        ax.plot(data.wl, nofe, "-", color="tab:cyan", lw=1.1, label="no Fe")
    if (~used).any() or (nofe is not None and np.isfinite(nofe).any()):
        ax.legend(fontsize=7, loc="lower left")
    ax.set_title(f"{res['line_id']}  A(Fe)={res['A']:.2f}+/-{res['sigma']:.2f}  chi2r={res['chi2_red']:.1f}  "
                 f"broad={res['broad_kms']:.1f} km/s", fontsize=9)
    ax.set_xlabel("Wavelength [A]")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fit_one_line(job: dict) -> dict:
    """Worker: one line, own scratch directory. Never raises; failures go into `status`."""
    row, s = job["line"], job["settings"]
    w = float(row["wavelength"])
    res = dict(line_id=job["line_id"], species=row["species"], ion=int(row["ion"]), wavelength=w,
               ep=float(row["excitation_potential"]), ew_mA=float(row["voigt_fit_area"]),
               rew=float(np.log10(row["voigt_fit_area"] / 1000.0 / w)),
               A=np.nan, sigma=np.nan, chi2_red=np.nan, dv_kms=np.nan, broad_kms=np.nan, broad_at_limit=False,
               cont=np.nan, npts=int(len(job["wl"])), n_synth=0, at_boundary=False,
               chi2_left=round(job["fit_lo"], 4), chi2_right=round(job["fit_hi"], 4), region_source=job["region_source"],
               nofe_depth_centre=np.nan, nofe_depth_max=np.nan, no_fe_status="off",
               other_lines_in_window=job["neighbours"], nlte_levels=job["nlte_levels"], status="ok")
    outdir = Path(job["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="ts_", dir=SCRATCH_DIR))
    fitter = None
    try:
        if len(job["wl"]) < 8:
            res["status"] = "no_data"
            return res
        win_lo, win_hi = job["win_lo"], job["win_hi"]
        data = Data(w, max(w - win_lo, win_hi - w), job["wl"], job["fl"], np.full(len(job["wl"]), 1.0 / job["snr"]))
        synth = LineSynth(job["cfg"], job["atm"], win_lo - s.synth_pad, win_hi + s.synth_pad,
                          workdir, workdir / "log.txt")
        region = (data.wl >= job["fit_lo"]) & (data.wl <= job["fit_hi"])
        fitter = LineFitter(synth, data, job["nlte"], s, job["fixed_broad"], region)
        out = fitter.run(job["a_guess"], job["warm"])
        model, used, nofe = out.pop("model"), out.pop("used"), out.pop("model_nofe")
        res.update(out)
        if np.isfinite(nofe).any():                           # absorption of everything that is not iron
            res["nofe_depth_centre"] = round(float(1.0 - nofe[np.abs(data.wl - w) <= 0.03].min()), 4)
            res["nofe_depth_max"] = round(float(1.0 - nofe[used].min()), 4)
        np.savetxt(outdir / "bestfit.csv", np.column_stack([data.wl, data.fl, model, used.astype(int), nofe]),
                   delimiter=",", header="wavelength,flux_obs,flux_model,used_in_fit,flux_noFe", comments="",
                   fmt=["%.4f", "%.6f", "%.6f", "%d", "%.6f"])
        if s.make_plots:
            _plot_fit(outdir / "fit.png", res, data, model, row, used, nofe)
    except Exception as exc:                              # noqa: BLE001 - keep the batch going
        res["status"] = f"failed: {type(exc).__name__}: {exc}"
        # The full Turbospectrum log is ~15 MB per line (bsyn warns about every Fe line of the whole line
        # list that has no level in the model atom), so only the tail is kept, and only when the fit fails.
        if (workdir / "log.txt").exists():
            with (workdir / "log.txt").open(errors="replace") as f:
                tail = deque(f, maxlen=300)
            (outdir / "log_tail.txt").write_text("".join(tail))
    finally:
        if s.keep_synth:
            if fitter is not None and fitter.evals:
                write_csv(outdir / "chi2_curve.csv",
                          [dict(A=a, chi2=e["chi2"], dv_kms=e["dv"], broad_kms=e["broad"], cont=e["cont"],
                                synth=f"synth/A{a:.4f}_{job['nlte']}.cvl")
                           for a, e in sorted(fitter.evals.items())])
            spectra = sorted(workdir.glob("A*.spec")) + sorted(workdir.glob("A*.cvl"))
            if spectra:
                (outdir / "synth").mkdir(exist_ok=True)
                for f in spectra:
                    shutil.copy2(f, outdir / "synth" / f.name)
        if s.keep_files:
            shutil.copytree(workdir, outdir / "work", symlinks=True, dirs_exist_ok=True)
        shutil.rmtree(workdir, ignore_errors=True)
    return res


# ---------------------------------------------------------------------------
# NLTE level identification of the selected lines
# ---------------------------------------------------------------------------
#
# bsyn applies departure coefficients to a line only if the line list gives both of its levels as model-atom
# level numbers; if either is 0 the whole line is computed in LTE. In the GES NLTE line list every Fe II line
# has "0 0" (the Fe II levels of atom.fe607a are unlabelled, energy-binned super-levels, so label matching
# found nothing), and a few Fe I lines miss one level (e.g. 5225.53: its a5D1 lower level). For the SELECTED
# lines the missing numbers are filled in by energy, in a temporary copy of the line list:
#   Fe I : the model-atom Fe I level closest in energy (fine structure is resolved), if within FE1_MAX_DE
#          and no second level lies within FE1_AMBIGUOUS_DE of it (a few levels are near-degenerate),
#   Fe II: the closest Fe II super-level (energies in the model atom count from the Fe I ground state).
# The result for every selected line is written to nlte_levels.csv and to the `nlte_levels` column.

CM_PER_EV = 8065.544
FE1_IONISATION_CM = 63737.704          # NIST
FE1_MAX_DE = 50.0                      # cm^-1
FE1_AMBIGUOUS_DE = 10.0                # cm^-1: two Fe I levels this close cannot be told apart by energy

_LL_HEADER = re.compile(r"^'\s*([\d.]+)\s*'\s+(\d+)\s+(\d+)\s*$")
_LL_LEVELS = re.compile(r"^(?P<head>.*'\s+)(?P<lo>\d+)(?P<mid>\s+)(?P<up>\d+)(?P<tail>\s+'[^']*'\s+'[^']*'.*)$", re.S)


def air_to_vac(wl_air: float) -> float:
    """Air -> vacuum wavelength [A] (IAU standard, Morton 2000)."""
    s2 = (1e4 / wl_air) ** 2
    return wl_air * (1 + 8.34254e-5 + 2.406147e-2 / (130 - s2) + 1.5998e-4 / (38.9 - s2))


def read_model_atom(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Energies [cm^-1 above the Fe I ground state], statistical weights and ionisation stages of the
    model-atom levels; level number n is at index n-1."""
    with Path(path).open(errors="replace") as f:
        rows = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("*")]
    nlev = int(rows[2].split()[0])
    t = [ln.split() for ln in rows[3:3 + nlev]]
    return (np.array([float(x[0]) for x in t]), np.array([float(x[1]) for x in t]),
            np.array([int(x[-1]) for x in t]))


def _match_level(e_target: float, ion: int, atom) -> tuple[int, float, bool]:
    """(level number, E_atom - E_target, ambiguous) of the closest model-atom level of this ionisation stage."""
    E, _, stage = atom
    idx = np.flatnonzero(stage == ion)
    e_rel = E[idx] - (FE1_IONISATION_CM if ion == 2 else 0.0)
    d = np.abs(e_rel - e_target)
    order = np.argsort(d)
    j = int(order[0])
    ambiguous = len(order) > 1 and d[order[1]] - d[j] < FE1_AMBIGUOUS_DE
    return int(idx[j] + 1), float(e_rel[j] - e_target), bool(ambiguous)


def assign_nlte_levels(cfg: StarConfig, lines: list[dict], workdir: Path) -> tuple[list[Path], dict[str, dict]]:
    """Line lists to use (patched copies in workdir where something changed) and, per selected line_id,
    what was found and assigned."""
    atom = read_model_atom(cfg.atom_path / cfg.model_atom)
    wanted = {1: [r for r in lines if r["ion"] == 1], 2: [r for r in lines if r["ion"] == 2]}
    report: dict[str, dict] = {}
    paths = []
    for path in cfg.linelists:
        out, changed, block = [], False, None
        with Path(path).open(errors="replace") as f:
            for ln in f:
                h = _LL_HEADER.match(ln)
                if h:
                    block = int(h[2]) if abs(float(h[1]) - cfg.Z) < 1e-6 else None
                    out.append(ln)
                    continue
                if block in wanted and not ln.startswith("'"):
                    t = ln.split(None, 2)
                    wl, ep = float(t[0]), float(t[1])
                    hit = next((r for r in wanted[block] if abs(r["wavelength"] - wl) < 0.002
                                and abs(r["excitation_potential"] - ep) < 0.01 and r["line_id"] not in report), None)
                    m = _LL_LEVELS.match(ln) if hit else None
                    if m:
                        lo, up = int(m["lo"]), int(m["up"])
                        e_lo = ep * CM_PER_EV
                        e_up = e_lo + 1e8 / air_to_vac(wl)
                        (n_lo, d_lo, a_lo), (n_up, d_up, a_up) = (_match_level(e_lo, block, atom),
                                                                  _match_level(e_up, block, atom))
                        rep = dict(line_id=hit["line_id"], levels_in_linelist=f"{lo} {up}",
                                   closest_lower=n_lo, dE_lower=round(d_lo, 1),
                                   closest_upper=n_up, dE_upper=round(d_up, 1))
                        new_lo, new_up = lo, up
                        if lo == 0 or up == 0:
                            ok = block == 2 or ((lo or (abs(d_lo) < FE1_MAX_DE and not a_lo)) and
                                                (up or (abs(d_up) < FE1_MAX_DE and not a_up)))
                            if ok:
                                new_lo, new_up = lo or n_lo, up or n_up
                                ln = f"{m['head']}{new_lo}{m['mid']}{new_up}{m['tail']}"
                                changed = True
                        how = ("line list" if lo and up else
                               ("assigned by energy" + (" (Fe II super-levels)" if block == 2 else ""))
                               if new_lo and new_up else "LTE: no unambiguous model-atom level within tolerance")
                        rep.update(levels_used=f"{new_lo} {new_up}" if new_lo and new_up else "LTE", how=how)
                        report[hit["line_id"]] = rep
                out.append(ln)
        if changed:
            dst = workdir / Path(path).name
            dst.write_text("".join(out))
            paths.append(dst)
        else:
            paths.append(Path(path))
    for r in lines:
        report.setdefault(r["line_id"], dict(line_id=r["line_id"], levels_used="LTE", how="not found in the line lists"))
    return paths, report


def fit_lines(lines: list[dict], atm: AtmosphereSpec, cfg: StarConfig,
              spectrum: tuple[np.ndarray, np.ndarray], outdir: Path, *,
              a_guess: dict[str, float] | None = None, broadening: float | None = None,
              settings: FitSettings | None = None, nlte: str = "T") -> list[dict]:
    """Fit all lines in parallel; returns one dict per line, sorted by wavelength (also written to
    lines.csv). `a_guess` maps line_id -> starting A(Fe) (warm start); `broadening` (km/s) fixes the
    extra broadening for every line and overrides the settings; None follows settings.fit_broadening /
    settings.broadening."""
    s = settings or FitSettings()
    if broadening is None and not s.fit_broadening:
        broadening = s.broadening                 # fixed for every line
    if broadening is not None and not broadening >= 0:
        raise ValueError(f"broadening must be a value in km/s >= 0, not {broadening!r}")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    obs_wl, obs_fl = spectrum
    a_default = SOLAR_FE + atm.feoh
    a_guess = a_guess or {}

    ll_dir = Path(tempfile.mkdtemp(prefix="ll_", dir=SCRATCH_DIR))
    levels: dict[str, dict] = {}
    if nlte == "T" and s.assign_nlte_levels:
        paths, levels = assign_nlte_levels(cfg, lines, ll_dir)
        cfg = dataclasses.replace(cfg, linelists=paths)
        write_csv(outdir / "nlte_levels.csv", [levels[r["line_id"]] for r in lines])

    jobs = []
    for row in lines:
        win_lo, win_hi, fit_lo, fit_hi, source = line_region(row, s)
        m = (obs_wl >= win_lo) & (obs_wl <= win_hi)
        # other selected lines inside the fit region: they respond to A(Fe) too, so these fits are coupled
        neighbours = " ".join(r["line_id"] for r in lines
                              if r is not row and fit_lo <= r["wavelength"] <= fit_hi)
        lev = levels.get(row["line_id"])
        nlte_levels = ("LTE run" if nlte != "T" else
                       f"{lev['levels_used']} ({lev['how']})" if lev else "not checked (line list as is)")
        jobs.append(dict(line=row, line_id=row["line_id"], neighbours=neighbours, nlte_levels=nlte_levels,
                         atm=atm, cfg=cfg,
                         wl=obs_wl[m], fl=obs_fl[m], win_lo=win_lo, win_hi=win_hi, fit_lo=fit_lo, fit_hi=fit_hi,
                         region_source=source, snr=cfg.snr, a_guess=a_guess.get(row["line_id"], a_default),
                         warm=row["line_id"] in a_guess, fixed_broad=broadening, nlte=nlte,
                         outdir=outdir / "lines" / row["line_id"], settings=s))

    t0 = time.time()
    results = []
    # fork: workers inherit everything, so the calling script is not re-imported (Python >= 3.14
    # defaults to forkserver on Linux)
    try:
        with ProcessPoolExecutor(max_workers=s.n_jobs, mp_context=multiprocessing.get_context("fork")) as ex:
            futures = [ex.submit(fit_one_line, j) for j in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                results.append(r)
                if r["status"] != "ok":
                    print(f"    [{i}/{len(jobs)}] {r['line_id']}: {r['status']}", flush=True)
                elif i % 10 == 0 or i == len(jobs):
                    print(f"    [{i}/{len(jobs)}] lines done ({time.time() - t0:.0f} s)", flush=True)
    finally:
        shutil.rmtree(ll_dir, ignore_errors=True)

    results.sort(key=lambda r: r["wavelength"])
    write_csv(outdir / "lines.csv", results)
    return results


# ---------------------------------------------------------------------------
# Diagnostics: excitation / ionisation balance and microturbulence
# ---------------------------------------------------------------------------

def quality_reasons(rows: list[dict], s: FitSettings, flags: dict[str, str] | None = None) -> list[str]:
    """Why each line is left out of the summary before sigma clipping ("" = candidate).
    A manual flag wins over everything else; then failed fits, unbracketed chi^2 minima and the chi^2 cut."""
    flags = flags or {}
    chi = col(rows, "chi2_red")
    fitted = np.array([r["status"] == "ok" and not r["at_boundary"] and np.isfinite(r["A"]) and np.isfinite(r["sigma"])
                       for r in rows])
    unflagged = np.array([r["line_id"] not in flags for r in rows])
    base = fitted & unflagged
    cut = max(s.chi2_factor * np.median(chi[base]), 1.0) if base.any() else np.inf
    out = []
    for r, c in zip(rows, chi):
        if r["line_id"] in flags:
            out.append(f"flagged: {flags[r['line_id']]}")
        elif r["status"] != "ok":
            out.append(f"rejected: {r['status']}")
        elif r["at_boundary"] or not (np.isfinite(r["A"]) and np.isfinite(r["sigma"])):
            out.append("rejected: chi2 minimum not bracketed (A at the search limit)")
        elif c > cut:
            out.append(f"rejected: chi2_red {c:.2f} > cut {cut:.2f}")
        else:
            out.append("")
    return out


def analyze(rows: list[dict], s: FitSettings | None = None, flags: dict[str, str] | None = None) -> dict:
    """Mean abundances and slopes. Fe I lines are sigma-clipped on the residuals of a joint
    A = c + a*EP + b*REW fit; Fe II lines are not clipped (there are too few). `flags` maps line_id -> reason
    for lines you excluded by hand. info["selection"] holds, aligned with rows, "used", "sigma-clipped (...)",
    "flagged: ..." or "rejected: ..." for every line."""
    s = s or FitSettings()
    selection = quality_reasons(rows, s, flags)
    ok = np.array([sel == "" for sel in selection])
    ion, ep, rew, A = col(rows, "ion"), col(rows, "ep"), col(rows, "rew"), col(rows, "A")
    m1, m2 = ok & (ion == 1), ok & (ion == 2)
    idx1 = np.flatnonzero(m1)
    ep1, rew1, A1 = ep[m1], rew[m1], A[m1]

    keep = np.ones(len(A1), bool)
    res, sd = np.zeros(len(A1)), np.inf
    if s.sigma_clip and len(A1) >= 6:
        X = np.column_stack([np.ones(len(A1)), ep1, rew1])
        for _ in range(5):
            coef = np.linalg.lstsq(X[keep], A1[keep], rcond=None)[0]
            res = A1 - X @ coef
            sd = res[keep].std(ddof=3) if keep.sum() > 3 else np.inf
            new = np.abs(res) < s.clip_sigma * sd
            if new.sum() < 5 or (new == keep).all():
                break
            keep = new
    for j, i in enumerate(idx1):
        selection[i] = "used" if keep[j] else f"sigma-clipped ({res[j] / sd:+.1f} sigma from the A-EP-REW fit)"
    for i in np.flatnonzero(m2):
        selection[i] = "used"
    ep1, rew1, A1k = ep1[keep], rew1[keep], A1[keep]
    A2 = A[m2]

    ids = [r["line_id"] for r in rows]
    info: dict = dict(n_fe1=len(A1k), n_fe1_clipped=int(len(A1) - len(A1k)), n_fe2=len(A2),
                      n_flagged=sum(sel.startswith("flagged") for sel in selection),
                      n_rejected=sum(sel.startswith("rejected") for sel in selection),
                      used_ids=[i for i, sel in zip(ids, selection) if sel == "used"],
                      clipped_ids=[i for i, sel in zip(ids, selection) if sel.startswith("sigma-clipped")],
                      flagged_ids=[i for i, sel in zip(ids, selection) if sel.startswith("flagged")],
                      selection=selection)
    nan = float("nan")
    if len(A1k) >= 4:
        sl_ep, ic_ep, er_ep = linfit(ep1, A1k)
        sl_rew, ic_rew, er_rew = linfit(rew1, A1k)
        info.update(slope_ep=sl_ep, slope_ep_err=er_ep, icpt_ep=ic_ep,
                    slope_rew=sl_rew, slope_rew_err=er_rew, icpt_rew=ic_rew,
                    A1=float(A1k.mean()), s1=float(A1k.std(ddof=1)), e1=float(A1k.std(ddof=1) / np.sqrt(len(A1k))))
    else:
        info.update(slope_ep=nan, slope_ep_err=nan, icpt_ep=nan, slope_rew=nan, slope_rew_err=nan,
                    icpt_rew=nan, A1=nan, s1=nan, e1=nan)
    if len(A2) >= 2:
        info.update(A2=float(A2.mean()), s2=float(A2.std(ddof=1)), e2=float(A2.std(ddof=1) / np.sqrt(len(A2))))
    else:
        info.update(A2=nan, s2=nan, e2=nan)
    info["dA"] = info["A2"] - info["A1"]
    info["dA_err"] = float(np.hypot(info["e1"], info["e2"]))
    info["feh"] = info["A1"] - SOLAR_FE
    return info


def print_summary(info: dict) -> None:
    print(f"  Fe I : A = {info['A1']:.3f} +/- {info['e1']:.3f} (scatter {info['s1']:.3f}, "
          f"{info['n_fe1']} lines, {info['n_fe1_clipped']} sigma-clipped)")
    print(f"  Fe II: A = {info['A2']:.3f} +/- {info['e2']:.3f} (scatter {info['s2']:.3f}, {info['n_fe2']} lines)")
    print(f"  A(Fe II) - A(Fe I) = {info['dA']:+.3f} +/- {info['dA_err']:.3f}")
    print(f"  slope vs EP  = {info['slope_ep']:+.4f} +/- {info['slope_ep_err']:.4f} dex/eV")
    print(f"  slope vs REW = {info['slope_rew']:+.4f} +/- {info['slope_rew_err']:.4f} dex/dex")
    print(f"  sigma-clipped: {', '.join(info['clipped_ids']) or '-'}")
    print(f"  flagged by hand: {', '.join(info['flagged_ids']) or '-'}")
    print(f"  rejected by the quality cuts: {info['n_rejected']} (see `selection` in lines.csv)")


def totals_text(info: dict) -> str:
    """The Fe I / Fe II totals of a run on one line (info from analyze(), or `results` of summary.yaml). Means of
    the lines marked "used"; +/- is scatter / sqrt(N), (sd) the line-to-line scatter."""
    def one(name, k, n):
        a, e, sd, num = info.get(f"A{k}"), info.get(f"e{k}"), info.get(f"s{k}"), info.get(f"n_fe{k}")
        if a is None or not np.isfinite(a):
            return f"A({name}) = n/a"
        return f"A({name}) = {a:.3f} +/- {e:.3f} (sd {sd:.3f}, N = {num})"
    t = f"{one('Fe I', 1, 1)}     {one('Fe II', 2, 2)}"
    if np.isfinite(info.get("dA", np.nan)):
        t += f"     Fe II - Fe I = {info['dA']:+.3f} +/- {info['dA_err']:.3f}     [Fe/H] = {info['feh']:+.3f}"
    return t


def plot_diagnostics(rows: list[dict], info: dict, path: Path, title: str = "") -> None:
    """A(Fe) against EP, REW and wavelength. Filled = used, x = sigma-clipped, open square = flagged by hand,
    open circle = rejected by the quality cuts. The y range follows the used lines; points beyond it are
    drawn on the edge as triangles."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ion, A, sig = col(rows, "ion"), col(rows, "A"), col(rows, "sigma")
    kind = np.array([sel.split(" ")[0].rstrip(":") for sel in info["selection"]])   # used/sigma-clipped/...
    used = kind == "used"
    if used.any():
        lo, hi = A[used].min() - 0.25, A[used].max() + 0.25
    else:
        lo, hi = np.nanmin(A) - 0.1, np.nanmax(A) + 0.1
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    panels = [("ep", "Excitation potential [eV]", "ep"), ("rew", "log(EW/lambda)", "rew"),
              ("wavelength", "Wavelength [A]", None)]
    styles = {"sigma-clipped": dict(marker="x", color="0.35", ls="none", ms=6, label="sigma-clipped"),
              "flagged": dict(marker="s", mfc="none", mec="tab:orange", ls="none", ms=7, label="flagged by hand"),
              "rejected": dict(marker="o", mfc="none", mec="0.6", ls="none", ms=6, label="rejected (quality)")}
    for ax, (name, label, key) in zip(axes, panels):
        x = col(rows, name)
        for k, color in ((1, "tab:blue"), (2, "tab:red")):
            sel = used & (ion == k)
            ax.errorbar(x[sel], A[sel], yerr=sig[sel], fmt="o", color=color, ms=4, lw=0.8,
                        label=f"Fe {'I' * k} used")
        for kd, st in styles.items():
            sel = (kind == kd) & np.isfinite(A)
            inside = sel & (A >= lo) & (A <= hi)
            ax.plot(x[inside], A[inside], **st)
            for edge, below in ((lo, True), (hi, False)):
                off = sel & ((A < lo) if below else (A > hi))
                ax.plot(x[off], np.full(off.sum(), edge), marker="v" if below else "^", ls="none",
                        color=st.get("color", st.get("mec")), ms=6)
        if key and np.isfinite(info[f"slope_{key}"]):
            xx = np.array([x[used & (ion == 1)].min(), x[used & (ion == 1)].max()])
            ax.plot(xx, info[f"icpt_{key}"] + info[f"slope_{key}"] * xx, "b-", lw=0.8)
            ax.set_title(f"slope {info[f'slope_{key}']:+.4f} +/- {info[f'slope_{key}_err']:.4f}", fontsize=9)
        for k, color in ((1, "tab:blue"), (2, "tab:red")):                      # the totals
            m, e = info.get(f"A{k}"), info.get(f"e{k}")
            if m is not None and np.isfinite(m):
                ax.axhspan(m - e, m + e, color=color, alpha=0.10, lw=0)
                ax.axhline(m, color=color, ls=":", lw=1.0, label=f"mean Fe {'I' * k}: {m:.3f} +/- {e:.3f}")
        ax.set_ylim(lo - 0.02, hi + 0.02)
        ax.set_xlabel(label)
        ax.set_ylabel("A(Fe)")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(f"{title}\n{totals_text(info)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


_SELECTION_COLORS = {"used": "black", "sigma-clipped": "0.45", "flagged": "tab:orange", "rejected": "tab:red"}


def selection_kind(selection: str) -> str:
    """"used", "sigma-clipped", "flagged" or "rejected" from a `selection` string ("" if unknown)."""
    return selection.split(" ")[0].rstrip(":") if selection else ""


def read_bestfit(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """wavelength, observed, model, used-in-fit mask, no-Fe model from a bestfit.csv (older files lack the last
    two columns: all pixels used, no-Fe model = nan)."""
    a = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    used = a[:, 3].astype(bool) if a.shape[1] > 3 else np.ones(len(a), bool)
    nofe = a[:, 4] if a.shape[1] > 4 else np.full(len(a), np.nan)
    return a[:, 0], a[:, 1], a[:, 2], used, nofe


def plot_overview(rows: list[dict], run_dir: Path, path: Path, windows: dict[str, tuple] | None = None,
                  title: str = "", marks: dict[str, str] | None = None, ncols: int = 8,
                  totals: str = "") -> None:
    """All line fits of a run in one image (from lines/<line_id>/bestfit.csv), sorted by wavelength and
    numbered. Panel colour = selection (black used, grey sigma-clipped, orange flagged, red rejected); the grey
    trace at the bottom of each panel is obs - fit around a zero line. `windows` (line_id -> (left, right))
    shades fit_left..fit_right; `marks` (line_id -> text) adds a line of text to a panel."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = sorted(rows, key=lambda r: float(r["wavelength"]))
    windows, marks = windows or {}, marks or {}
    nrows = -(-len(rows) // ncols)
    head = 1.55                                           # inches for the figure title above the first row
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.5 * nrows + head), squeeze=False)

    def f(r, k):
        if k == "broad_kms" and k not in r:                  # runs made before the rename
            k = "fwhm_kms"
        try:
            return float(r.get(k, "nan"))
        except (TypeError, ValueError):
            return float("nan")

    for n, (ax, r) in enumerate(zip(axes.flat, rows), 1):
        lid = r["line_id"]
        sel = str(r.get("selection", ""))
        color = _SELECTION_COLORS.get(selection_kind(sel), "black")
        fit = Path(run_dir) / "lines" / lid / "bestfit.csv"
        if fit.exists():
            wl, obs, mod, used, nofe = read_bestfit(fit)
            lo, hi = min(obs.min(), mod.min()), max(obs.max(), mod.max())
            base = lo - 0.12 * (hi - lo) - 0.02
            if lid in windows:
                ax.axvspan(*windows[lid], color="0.92", zorder=0)
            ax.axvline(f(r, "wavelength"), color="0.7", lw=0.6, zorder=0)
            for b in (f(r, "chi2_left"), f(r, "chi2_right")):
                if np.isfinite(b):
                    ax.axvline(b, color="tab:blue", lw=0.6, ls="--", zorder=0)
            ax.axhline(base, color="0.7", lw=0.5)
            ax.plot(wl, base + np.where(used, obs - mod, np.nan), color="0.5", lw=0.6)
            ax.plot(wl[used], obs[used], "k.", ms=1.5)
            ax.plot(wl[~used], obs[~used], ".", color="0.75", ms=1.5)
            ax.plot(wl, mod, "r-", lw=0.8)
            if np.isfinite(nofe).any():
                ax.plot(wl, nofe, "-", color="tab:cyan", lw=0.8)
            ax.set_xlim(wl.min(), wl.max())
            ax.set_ylim(base - 0.04 * (hi - lo) - 0.01, hi + 0.04 * (hi - lo) + 0.005)
        else:
            ax.text(0.5, 0.5, f"no fit\n{r.get('status', '')}", ha="center", va="center", fontsize=7,
                    transform=ax.transAxes)
        if str(r.get("broad_at_limit", "")) == "True":
            color_b = "tab:purple"
            ax.text(0.98, 0.04, "broadening at limit", transform=ax.transAxes, fontsize=6.5, color=color_b,
                    ha="right", va="bottom")
        short = {"used": "used", "sigma-clipped": "sigma-clipped", "flagged": "flagged",
                 "rejected": sel.replace("rejected: ", "rej: ")[:34]}.get(selection_kind(sel), sel[:34])
        ax.set_title(f"{n}. {lid}  A={f(r, 'A'):.2f}+/-{f(r, 'sigma'):.2f}\n"
                     f"chi2r {f(r, 'chi2_red'):.2f}  b {f(r, 'broad_kms'):.1f}  c {f(r, 'cont'):.3f}  {short}",
                     fontsize=7, color=color, loc="left")
        if lid in marks:
            ax.text(0.02, 0.04, marks[lid], transform=ax.transAxes, fontsize=6.5, color="tab:blue",
                    va="bottom")
        for side in ax.spines.values():
            side.set_edgecolor(color)
            side.set_linewidth(1.6 if color != "black" else 0.8)
        ax.tick_params(labelsize=5.5, length=2, pad=1)
        ax.ticklabel_format(useOffset=False, axis="x")
    for ax in list(axes.flat)[len(rows):]:
        ax.axis("off")
    fig.suptitle(f"{title}\n{totals}\npanel colour: black used, grey sigma-clipped, orange flagged by hand, red rejected "
                 "by the quality cuts;  grey band: fit_left..fit_right;  dashed: fit region;  light grey points: outside it;  cyan: no-Fe profile;  "
                 "bottom trace: obs - fit", fontsize=10,
                 y=1 - 0.25 / (2.5 * nrows + head), va="top")
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.02, top=1 - head / (2.5 * nrows + head),
                        wspace=0.18, hspace=0.55)
    fig.savefig(path, dpi=120)
    plt.close(fig)
