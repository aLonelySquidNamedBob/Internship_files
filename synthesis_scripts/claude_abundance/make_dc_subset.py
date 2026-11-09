#!/usr/bin/env python3
"""
Cut a small departure-coefficient grid out of a zipped full grid, without unzipping it.

The full grids are big (Ba: 11.5 GB zipped, 23.9 GB unpacked), but an interpolation only ever reads the eight records
around one star. This script reads the zip as a stream: the grid goes through memory a little at a time and only the
wanted records are kept. The zip and the auxData file are only read. It writes, in data/original/NLTE/dc/<element>/ :

    dc.bin       the header and the wanted records of the binary grid
    master.dat   the list of those records, as in the auxData file but with pointers into the new dc.bin
    README.md    what was done, for which parameters, how to extend or replace it

These two are the names interp_nlte_4.py looks for, so nothing else has to change. Run it as a plain script: edit the
settings below, then `python make_dc_subset.py`. Set DRY_RUN = True to only see what it would keep.

A record is kept if its model atmosphere is inside the box around one of POINTS: the bracket of grid points around the
parameters (as interp_nlte_4.py builds it) plus MARGIN grid steps on every side, for all the abundances of that model.

Format of the binary grid (read by interpol_modeles_nlte, form='unformatted', access='stream'): a 1000-byte header, then
for each record: 500-byte model id, n_dep and n_lev (4-byte integers), tau (n_dep x 8 bytes), departure coefficients
(n_dep x n_lev x 8 bytes). A pointer in the auxData file is the 1-based byte position of a record.
"""

from __future__ import annotations

import os
import shutil
import struct
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))     # find fe_line_fit.py next to this script
from fe_line_fit import HOME

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

ELEMENT = "Ba"
DC_DIR = HOME / "data" / "original" / "NLTE" / "dc" / ELEMENT
ZIP_FILE = None                      # None -> the only *.zip in DC_DIR
MASTER_FILE = None                   # None -> the only auxData*.dat in DC_DIR
MARGIN = 1                           # grid steps added on every side of each bracket (Teff 100-250 K, logg 0.5, [Fe/H] 0.25-1)
POINTS = {                           # label: (MARCS mass: 1.0 spherical / 0.0 plane-parallel, Teff, logg, [Fe/H])
    "HD115444": (1.0, 4540, 1.00, -3.08),
    "HD196944": (1.0, 5213, 1.86, -2.29),
    "HD2454": (0.0, 6330, 3.95, -0.43),
    "HE22081239": (1.0, 5000, 1.50, -2.70),
    "TIC396792499": (1.0, 4670, 1.20, -2.21),
    "SUN": (0.0, 5778, 4.44, 0.00),
}
DRY_RUN = False                      # True: only read the auxData file and print what would be kept

# the MARCS grid as in interp_nlte_4.py (masses 0.0 and 1.0)
TEFFS = np.array([2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4250,
                  4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500, 6750, 7000, 7250, 7500, 7750, 8000])
LOGGS = np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5])
FEHS = np.array([-5.00, -4.00, -3.00, -2.50, -2.00, -1.50, -1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00])

HEADER_BYTES = 1000                  # of the binary grid
ID_BYTES = 500                       # of a record
CHUNK = 32 * 1024 * 1024             # skipped at a time


def alpha_of(feh: float) -> float:
    """[alpha/Fe] of the standard MARCS composition (find_aofe in interp_nlte_4.py)."""
    return 0.0 if feh >= 0.0 else (0.4 if feh <= -1.0 else -0.4 * feh)


def model_id(mass: float, teff: int, logg: float, feh: float) -> str:
    """The name of a model atmosphere in the auxData file (build_fn in interp_nlte_4.py)."""
    geo, vmic = ("p", "1") if mass == 0.0 else ("s", "2")
    a = alpha_of(feh)
    return (f"{geo}{teff}_g{logg:+3.1f}_m{mass}_t0{vmic}_st_z{feh:+4.2f}_a{a:+4.2f}_c+0.00_n+0.00_o{a:+4.2f}"
            f"_r+0.00_s+0.00")


def axis_box(grid: np.ndarray, x: float, margin: int) -> list:
    """The grid values of the bracket around x (one value if x is on the grid), plus `margin` steps on each side."""
    if not grid[0] <= x <= grid[-1]:
        raise SystemExit(f"{x} is outside the MARCS grid {grid[0]} .. {grid[-1]}")
    hit = np.flatnonzero(np.isclose(grid, x))
    lo, hi = (int(hit[0]), int(hit[0])) if hit.size else (int(np.searchsorted(grid, x)) - 1, int(np.searchsorted(grid, x)))
    return grid[max(lo - margin, 0):min(hi + margin, len(grid) - 1) + 1].tolist()


def wanted_models(points: dict, margin: int) -> set[str]:
    out = set()
    for mass, teff, logg, feh in points.values():
        for t in axis_box(TEFFS, teff, margin):
            for g in axis_box(LOGGS, logg, margin):
                for z in axis_box(FEHS, feh, margin):
                    out.add(model_id(mass, int(t), float(g), float(z)))
    return out


def read_master(path: Path) -> list[tuple[str, list[str], int]]:
    """(model id, the fields between the id and the pointer, pointer) of every record, in the order of the file."""
    records = []
    with path.open() as f:
        next(f)
        for line in f:
            if line.startswith("'"):
                _, model, rest = line.split("'", 2)
                fields = rest.split()
                records.append((model, fields[:-1], int(fields[-1])))
    return records


def only_file(folder: Path, pattern: str) -> Path:
    hits = sorted(folder.glob(pattern))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one {pattern} in {folder}, found {[p.name for p in hits]}: set it in the settings")
    return hits[0]


def check_targets(dc_bin: Path, master: Path) -> None:
    """Only a previous subset may be overwritten, never a full grid (or a link to one)."""
    for p in (dc_bin, master):
        if p.is_symlink():
            raise SystemExit(f"{p} is a symbolic link, probably to a full grid: not overwriting it")
    if dc_bin.exists() or master.exists():
        if not master.exists():
            raise SystemExit(f"{dc_bin} exists without {master.name}: not overwriting it")
        with master.open() as f:
            if "SUBSET" not in f.readline():
                raise SystemExit(f"{master} is not a subset made by this script: not overwriting it")


def read_exact(src, n: int) -> bytes:
    data = src.read(n)
    if len(data) != n:
        raise SystemExit(f"unexpected end of the grid: wanted {n} bytes, got {len(data)}")
    return data


def extract(zip_path: Path, pointers: dict[int, str], out: Path) -> tuple[dict[int, int], dict]:
    """Copy the header and the records at `pointers` (pointer -> model id) from the stream of the zip to `out`. Returns
    old pointer -> new pointer, and some numbers about the records. Each record is checked (id, sizes, finite numbers)."""
    new_ptr: dict[int, int] = {}
    sizes = {"n_dep": set(), "n_lev": set()}
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if len(members) != 1:
            raise SystemExit(f"{zip_path} should hold one file, not {[m.filename for m in members]}")
        total = members[0].file_size
        with zf.open(members[0]) as src, out.open("wb") as dst:
            dst.write(read_exact(src, HEADER_BYTES))
            pos, shown = HEADER_BYTES, 0              # pos: bytes of the grid consumed so far
            for ptr in sorted(pointers):
                skip = ptr - 1 - pos                  # pointers are 1-based
                if skip < 0:
                    raise SystemExit(f"record at {ptr} overlaps the previous one: the record format is not as expected")
                while skip:
                    n = min(skip, CHUNK)
                    read_exact(src, n)
                    skip -= n
                    pos += n
                    if pos - shown >= 2_000_000_000:
                        shown = pos
                        rate = pos / 1e6 / max(time.time() - t0, 1e-3)
                        print(f"    {pos / 1e9:5.1f} / {total / 1e9:.1f} GB read, {len(new_ptr)} / {len(pointers)} records, "
                              f"{rate:.0f} MB/s", flush=True)
                head = read_exact(src, ID_BYTES + 8)
                name = head[:ID_BYTES].decode(errors="replace").strip()
                if name != pointers[ptr]:
                    raise SystemExit(f"the record at {ptr} is {name!r}, not {pointers[ptr]!r}: the format is not as expected")
                n_dep, n_lev = struct.unpack("<ii", head[ID_BYTES:])
                if not (0 < n_dep < 1000 and 0 < n_lev < 10000):
                    raise SystemExit(f"the record at {ptr} has n_dep = {n_dep}, n_lev = {n_lev}: the format is not as expected")
                body = read_exact(src, 8 * n_dep * (1 + n_lev))
                if not np.isfinite(np.frombuffer(body, dtype="<f8")).all():
                    raise SystemExit(f"the record at {ptr} ({name}) holds non-finite numbers")
                new_ptr[ptr] = dst.tell() + 1
                dst.write(head)
                dst.write(body)
                pos += len(head) + len(body)
                sizes["n_dep"].add(n_dep)
                sizes["n_lev"].add(n_lev)
    return new_ptr, dict(n_dep=sorted(sizes["n_dep"]), n_lev=sorted(sizes["n_lev"]), seconds=time.time() - t0, grid_bytes=total)


def write_master(path: Path, records: list, new_ptr: dict[int, int], source: str) -> None:
    """The records, in the order of the auxData file. The word SUBSET in the first line tells interp_nlte_4.py and this
    script what it is."""
    with path.open("w") as f:
        f.write(f"# SUBSET of {source}, written by make_dc_subset.py: atmos ID, Teff [K], log(g) [cgs], [Fe/H], "
                "[alpha/Fe], mass, Vturb [km/s], A(X), pointer\n")
        for model, fields, ptr in records:
            f.write(f"'{model}'    " + "    ".join(fields) + f"    {new_ptr[ptr]}\n")


def write_readme(path: Path, zip_path: Path, aux_path: Path, models: list, records: list, info: dict, size: int) -> None:
    by_axis = "\n".join(
        f"| {label} | {'spherical' if m else 'plane-parallel'} | {t} | {g:.2f} | {z:+.2f} |"
        f" {axis_box(TEFFS, t, MARGIN)[0]}-{axis_box(TEFFS, t, MARGIN)[-1]} |"
        f" {axis_box(LOGGS, g, MARGIN)[0]:.1f} to {axis_box(LOGGS, g, MARGIN)[-1]:.1f} |"
        f" {axis_box(FEHS, z, MARGIN)[0]:+.2f} to {axis_box(FEHS, z, MARGIN)[-1]:+.2f} |"
        for label, (m, t, g, z) in POINTS.items())
    abund = np.array([float(fields[-1]) for _, fields, _ in records])
    path.write_text(f"""\
# {ELEMENT} departure coefficients: a subset of the full grid

Written by `synthesis_scripts/claude_abundance/make_dc_subset.py` on {time.strftime('%Y-%m-%d %H:%M')}.

The full {ELEMENT} grid unpacks to {info['grid_bytes'] / 1e9:.1f} GB, too much for this disk, so it was **not unzipped**.
An interpolation only reads the eight records around a star, so the script read the zip as a stream (through memory, a
little at a time) and saved only the records of the model atmospheres around the stars below.

| File | What it is |
|---|---|
| `{zip_path.name}` | the full grid as downloaded; not changed |
| `{aux_path.name}` | its master list; not changed |
| `dc.bin` | the subset of the binary grid, {size / 1e6:.0f} MB: the 1000-byte header + {len(records)} records of {len(models)} model atmospheres |
| `master.dat` | the list of those records, with pointers into the new `dc.bin` |
| `README.md` | this file (rewritten by the script each time it runs) |

`interp_nlte_4.py` reads `dc.bin` + `master.dat` here like it reads a full grid. The subset is **not** tracked by git
(`data/` is ignored), the script that makes it is.

## What it covers

Each star's parameters, the bracket of MARCS grid points around them, plus MARGIN = {MARGIN} grid step(s) on every side. All
the abundances available for those models are kept: A({ELEMENT}) from {abund.min():+.2f} to {abund.max():+.2f} overall.

| Star | Geometry | Teff | logg | [Fe/H] | Teff box | logg box | [Fe/H] box |
|---|---|---|---|---|---|---|---|
{by_axis}

Move the parameters inside a box and nothing has to be rebuilt. Outside, `interp_nlte_4.py` stops and says so.

## Things to know

- **Abundances are not interpolated.** Each of the eight corners of the interpolation uses the one record whose A({ELEMENT}) is
  closest to the requested value (`interp_nlte_4.py` chooses it). A run prints and saves (`summary.yaml`) the abundance used at
  each corner.
- **The grid stops around [{ELEMENT}/Fe] = +1 and has gaps.** For the strongly enhanced stars the requested A({ELEMENT}) is above the
  top of the grid at some corners, and some models lack a stretch of abundances. The closest abundance is used
  there and the run prints a warning when it is more than 0.1 dex away.
- Every record was checked while it was copied: its name against the master list, its array sizes (n_dep = {info['n_dep']},
  n_lev = {info['n_lev']}), and that all its numbers are finite.

## Extending it, or replacing it by the full grid

- More stars or wider boxes: edit `POINTS` / `MARGIN` in `make_dc_subset.py` and run it again (a few minutes). It replaces
  `dc.bin`, `master.dat` and this file, and refuses to overwrite anything that is not a previous subset.
- With enough disk space for the full grid: delete `dc.bin` and `master.dat`, unzip the grid here, then link it the
  way `dc/Fe/` is: `ln -s NLTEgrid_....bin dc.bin` and `ln -s auxData_....dat master.dat`.
""")


def main() -> None:
    zip_path = Path(ZIP_FILE) if ZIP_FILE else only_file(DC_DIR, "*.zip")
    aux_path = Path(MASTER_FILE) if MASTER_FILE else only_file(DC_DIR, "auxData*.dat")
    dc_bin, master = DC_DIR / "dc.bin", DC_DIR / "master.dat"

    models = wanted_models(POINTS, MARGIN)
    all_records = read_master(aux_path)
    known = {model for model, _, _ in all_records}
    missing = sorted(models - known)
    records = [r for r in all_records if r[0] in models]
    found = sorted(models & known)
    print(f"{ELEMENT}: {len(POINTS)} points, margin {MARGIN}: {len(models)} model atmospheres wanted, "
          f"{len(found)} in the grid ({len(missing)} are not), {len(records)} records of {len(all_records)}")
    pointers = {ptr: model for model, _, ptr in records}
    if len(pointers) != len(records):
        raise SystemExit("two records of the master file have the same pointer")
    estimate = len(records) * 60_000 + HEADER_BYTES           # a record is ~51 kB for Ba, more for bigger atoms
    free = shutil.disk_usage(DC_DIR).free
    print(f"output: roughly {estimate / 1e6:.0f} MB at most ({free / 1e9:.1f} GB free in {DC_DIR})")
    if DRY_RUN:
        print("DRY_RUN: nothing read from the zip, nothing written")
        return
    if free < estimate + 2_000_000_000:
        raise SystemExit("not enough free disk space (keeping 2 GB spare)")
    check_targets(dc_bin, master)

    tmp_bin, tmp_master = DC_DIR / "dc.bin.tmp", DC_DIR / "master.dat.tmp"
    print(f"streaming {zip_path.name} (nothing but the {len(records)} records is written) ...", flush=True)
    try:
        new_ptr, info = extract(zip_path, pointers, tmp_bin)
        write_master(tmp_master, records, new_ptr, zip_path.name)
        size = tmp_bin.stat().st_size
        os.replace(tmp_bin, dc_bin)
        os.replace(tmp_master, master)
    finally:
        tmp_bin.unlink(missing_ok=True)
        tmp_master.unlink(missing_ok=True)
    write_readme(DC_DIR / "README.md", zip_path, aux_path, found, records, info, size)
    print(f"done in {info['seconds'] / 60:.1f} min: {dc_bin} ({size / 1e6:.0f} MB), {master.name}, README.md "
          f"(n_dep {info['n_dep']}, n_lev {info['n_lev']})")


if __name__ == "__main__":
    main()
