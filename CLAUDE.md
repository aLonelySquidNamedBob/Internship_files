# NICO: NLTE stellar-abundance work with Turbospectrum

Research project (owner: Nico, working with T. Merle) that derives stellar parameters and abundances of
metal-poor / CEMP-type stars by fitting synthetic spectra, computed with **Turbospectrum NLTE 20.1**
(Fortran, `babsma` -> `bsyn` -> `faltbon`), to observed normalised spectra. Fe is the NLTE element
(model atom `atom.fe607a`, MARCS atmospheres interpolated together with departure coefficients).
Everything is plain Python scripts that you edit and run; there is no package, build system or test suite.

Git: `main`, remote on GitHub (`aLonelySquidNamedBob`). Only code and `config/` are tracked (20 files). `data/`,
`DATA/`, `ll`, `line_lists/`, `TiO_ll/`, `runs/`, `plots/`, `archive/`, `test/test1/*`, `*.png`, `log.txt` are
git-ignored. Many past commits are PRs from Claude branches, so keep commit messages short and descriptive.

## Layout (project root `/home/student/NICO`, also `HOME` in the scripts)

| Path | What |
|---|---|
| `config/stars.yaml` | One entry per star: spectra (raw/normalized, `rv`, `snr`, `instrument_fwhm`), `synthesis` (model atmosphere `.int`, departure-coefficient file, `NLTE_element`, `NLTE_abundance`, the converged `teff`, `logg`, `feoh`, `vmic`, `vmac`, `vmac_profile` (`gauss` or `radtan`), `molecular_line_lists`), `notes`. Stars: HD115444, HD196944, HD2454, HE22081239, TIC396792499, SUN. Paths are relative to `~/NICO`. |
| `config/reference_abundance.yaml` | Published A(X) per star (same keys as stars.yaml): `abundances: {El: {value, error, nlte, note}}` (null = not measured) and `isotope_ratios: C12_C13`. Source: Merle et al. table, a starting point. |
| `config/nlte.yaml` | NLTE elements: atomic number, model atom name/path, global DC grid (Fe, Ba). |
| `config/line_lists.yaml` | `master` list (atoms + `DATA/Hlinedata`, always used) and `molecular` groups (CC, CN, CH, NH, OH, MgH, SiH, TiO, ZrO, CaH, FeH, VO), each a list of `ll/molecules/*_GESv5.bsyn` files. |
| `config/yaml_test.py` | Scratch; broken (expects a top-level `stars` key that does not exist). |
| `synthesis_scripts/claude_abundance/` | **The current, maintained pipeline** (see below). |
| `synthesis_scripts/molecular_list_test.py` | Which molecular groups matter for a star (current, uses fe_line_fit helpers). |
| `synthesis_scripts/abundance/`, `temperature_scripts/`, `sun_template.py` | Older stand-alone scripts that each hard-code the babsma/bsyn/faltbon calls. `abundance_scan_yaml.py` (single-line A(Fe) scan, config-driven) is still useful; `orchestrator.py` is an unfinished sketch; `abundance_scan.py`, `sun_template.py` (Ba, old test dir) are legacy. |
| `plotting scripts/` | `molecular_lls.py`, `plot_spectra_test.py`: ad hoc plots. |
| `data/original/atm/` | MARCS grids: `MARCS_st_sph_t02_mod` (spherical, logg <= 3.5), `MARCS_st_ppl_t01_mod` (plane-parallel), `sun_marcs.mod`. |
| `data/original/NLTE/` | `interp_nlte_3.py` (+ `_2`, original) and Fortran `interpol_modeles_nlte`, `dc/<El>/` departure-coefficient grid, `MODEL_ATOM/`. `dc/Fe/` is the full grid; `dc/Ba/` holds the zipped full grid (23.9 GB unpacked, do not unzip: the disk cannot take it) plus a small subset (`dc.bin`, `master.dat`, explained in `dc/Ba/README.md`). |
| `data/interpolated/<star>/` | Interpolated `.int` atmosphere + `*_<El>_<A>_dc.dat` per star. |
| `data/observed_spectra/<star>/`, `data/normalized_spectra/` | Raw coadds and the normalised CSV/SSV spectra that are fitted. |
| `data/selected_lines/<star>/` | `fe_kept_lines.csv` (Fe I/II line list with EP, `chi2_left/right` fit-region overrides) and `fe_line_flags.yaml` (`line_id: reason`; "reject..." = excluded). Every other species: one file, `<star>_kept_lines.csv` (same format; each run takes the rows of its element), and per element `<el>_line_flags.yaml`. |
| `Turbospectrum_NLTE_20.1/` | Code + binaries: `exec-gf/{babsma_lu,bsyn_lu}`, `Utilities/faltbon`, `interpolator/`. Don't edit. |
| `DATA/` | Turbospectrum data tables (Hlinedata, `nlte_infofile.dat`, ...). Scripts link it as `./DATA` in scratch dirs. |
| `ll` -> `/home/student/GREG/test_tbs20/ll` | Symlink to atomic/molecular line lists (`ll/molecules/*.bsyn`, `nlte_ges_linelist_jmg6may2025_I_II`, GES lists). `line_lists/` and `TiO_ll/` are local copies / extra TiO lists (`ll`'s 47TiO is truncated, so line_lists.yaml points to `TiO_ll/`). |
| `runs/<star>/<kind>/<timestamp>_.../` | All outputs. Kinds: `fe_abundance`, `<el>_abundance` (other elements), `synthesis`, `molecular_tests`, `balmer_wings`, `balmer_wings_hlines_only`, `abundance_tests`, `temperature_tests`. |
| `log.txt` | Turbospectrum stdout/stderr from the older scripts (13 MB, overwritten). |

## The main pipeline: `synthesis_scripts/claude_abundance/`

All scripts: edit the **Settings block at the top** (`SELECTED_STAR`, parameters), then `python <script>` from anywhere
(paths use `~/NICO`). Python 3.14; needs numpy, scipy, pyyaml, matplotlib. No CLI flags except `review_fits.py`.

- **`fe_line_fit.py`**: library, ~1560 lines. Paths/constants (`HOME`, `RUNS_DIR`, `TS_DIR`, `SOLAR_FE = 7.45`,
  `DW`), `StarConfig`/`load_star_config`, `individual_abundances` (reference abundances, optionally shifted to keep
  [X/Fe] with `REFERENCE_SCALING="fe"`, plus overrides), `LineSynth` (writes babsma/bsyn/faltbon inputs, runs them),
  `FitSettings` dataclass (every fit option, documented in place), `LineFitter` (chi^2 on a few bsyn runs +
  parabola, per-line RV shift, broadening and continuum), `fit_lines` (ProcessPool, one scratch dir per line),
  `analyze` (means of A(Fe I)/A(Fe II), slopes vs EP/REW, quality cuts, sigma clipping, flags), plotting
  (`plot_diagnostics`, `plot_overview`), NLTE-level assignment helpers.
- **`find_stellar_parameters`** (no `.py` extension): runs ONE parameter set (Teff, logg, vmic, [Fe/H], vmac,
  NLTE T/F, mass 1.0 spherical / 0.0 plane-parallel): interpolates atmosphere + DCs via `interp_nlte_3.py`
  (DC computed for A(Fe) = 7.45 + FEH), fits every Fe line, writes
  `runs/<star>/fe_abundance/<time>_T<teff>_g<logg>_v<vmic>_z<feh>_<NLTE|LTE>/` with `lines.csv`, `summary.yaml`,
  `diagnostics.png`, `overview.png`, `lines/<line>/`, `atm/`, `config.yaml`. Nothing is iterated automatically:
  Teff/logg/vmic are tuned by hand by re-running (excitation balance, ionisation balance, no slope with REW).
- **`find_element_abundance.py`** + **`element_line_fit.py`**: the same line-by-line fit for any other element: that
  element in NLTE, everything else (Fe included) in LTE, because `LineSynth` lists only the fitted element in
  `nlte_infofile.dat`. The fit itself is imported from `fe_line_fit.py` (keep the fitting there, so every element gets it);
  the library adds the element config, the starting A(X), per-ion results with [X/H] and [X/Fe] on Turbospectrum's
  solar scale, a read-only report of NLTE levels and fixed isotope fractions (`ba_mix`, `eu_mix`). `MAKE_DIAGNOSTICS` is off by default.
- **`interp_nlte_4.py`** (tracked here, unlike `_3` in `data/`): `interp_nlte_3.py` for any element with a grid. For grids
  with several abundances per model it chooses the closest record per corner itself and gives the Fortran a master file
  with only those. Fe still goes through the old path, with identical output.
- **`synthesize_region.py`**: one synthesis of a free range (e.g. CH G band, C2 Swan, CN) over the observed spectrum,
  nothing fitted: chosen A(X) or [X/Fe], 12C/13C, molecule groups, other abundances; stellar parameters and broadening
  from stars.yaml. Draws an A -/+ offset band, the model without the element, extra ratios/abundances, residuals, and
  prints chi^2_red / rms per model. Writes `runs/<star>/synthesis/<time>_<El>_<lo>-<hi>/`.
- **`make_dc_subset.py`**: cuts a small grid out of a zipped one by streaming it, without unzipping. Edit `POINTS`/`MARGIN` to cover new parameters.
- **`review_fits.py`**: matplotlib GUI to keep/flag/reject each line of a run (writes `review.yaml` and the star's
  `<el>_line_flags.yaml`) and to set per-line chi^2 limits in the line CSV. `--overview` and `--region` are headless modes;
  `--element Ba` picks the newest run of that element.
- **`balmer_wings.py`**, **`balmer_wings_hlines_only.py`**: synthesise H-alpha / H-beta (full line lists vs H lines
  only) for the newest `fe_abundance` run, Teff +/- 100 K and a CUSTOM literature parameter set; Teff check from Balmer
  wings. They import `find_stellar_parameters` through `SourceFileLoader`.
- **`../molecular_list_test.py`**: one synthesis without molecules plus one per molecular group; recommends groups
  whose added absorption exceeds `IMPACT_THRESHOLD` for `molecular_line_lists` in stars.yaml.

Typical workflow: choose star -> `molecular_list_test.py` to choose molecules -> `find_stellar_parameters` (iterate
parameters) -> `review_fits.py` (reject blends) -> rerun -> `balmer_wings*.py` to cross-check Teff. Other elements come after
that: put the converged `teff`, `logg`, `feoh`, `vmic`, `vmac` and `vmac_profile` in stars.yaml, export `<star>_kept_lines.csv` (all species but Fe), run
`find_element_abundance.py` (it reads them from stars.yaml; its settings can override them), then `review_fits.py --element <El>`.

## Conventions and gotchas

- Abundances are A(X) = log N(X)/N(H) + 12; solar Fe on the MARCS/Turbospectrum scale is **7.45**
  (note stars.yaml SUN lists `NLTE_abundance: 7.46`).
- Atmosphere file names encode the parameters, e.g. `s5539_g+2.4_m1.0_t02_st_z-1.95_a+0.40_c+0.00_...int`
  (`s`/`p` = spherical/plane-parallel, g = logg, m = mass, z = [Fe/H], a = [alpha/Fe]); `parse_atm_name` reads it.
  [alpha/Fe] is +0.40 for the halo stars (HD2454 uses +0.08). The spherical MARCS grid ends at logg 3.5.
- Convolution widths follow faltbon: negative number = km/s, positive = mA. Instrumental FWHM default -3 km/s;
  extra broadening is `gauss` (FWHM) or `radtan` (zeta_RT = 1.433 x number, so vmac 5.5 means ~7.9 km/s).
- Turbospectrum quirks: babsma needs ~1 A wavelength margin beyond bsyn's range; `DATA/nlte_infofile.dat` must
  start with a lone `#` line; Fortran dislikes long paths, so scratch dirs live in the system temp dir;
  `SPHERICAL_RT = False` even with spherical MARCS (deliberate, matches the old scripts).
- `-20` as an abundance removes that element's lines (used to isolate Fe lines in the abundance scans).
- Do not invent NLTE levels (`assign_nlte_levels=False`); lines without levels are handled by the line list.
- A line is NLTE only if the line list gives both its model-atom levels; "0 0" lines stay LTE in an NLTE run. Coverage in the
  GES NLTE list varies a lot: Ba II / Eu II isotope blocks, Na, Y are complete, Mg I has 9 of 244 lines (`nlte_levels.csv` of a run shows it).
- Departure coefficients are not interpolated in abundance: each grid corner uses the record closest to A(X). The Ba grid stops near
  [Ba/Fe] = +1 and has gaps, so the CEMP-s stars get warnings; a run saves the abundance used per corner in `summary.yaml`.
- Don't hand-edit anything in `runs/`; each run stores a copy of the scripts and `config.yaml` used.
- The `stars.yaml` loader accepts both a top-level `stars:` key and bare star keys (currently bare).
- Hard-coded absolute paths exist in `config/nlte.yaml`, `config/line_lists.yaml` and the older scripts
  (`/home/student/NICO/...`); only `/home/student` works.
- Compute is heavy: line fits run with `n_jobs=10`, molecular tests with 12 workers. Don't launch full runs
  casually; ask first, and prefer one star/one small window for checks.
- Working tree normally has uncommitted edits to the claude_abundance scripts; check `git diff` before assuming the
  committed version.
