#!/usr/bin/env python3
"""Plot every synthetic spectrum of the most recent TIC abundance_tests run over the observed spectrum."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import yaml

home = Path("~/NICO").expanduser()
C_KMS = 299792.458
edge = 0.5      # Å cut from each end of the synthetic spectra (faltbon distorts the edges)

# One colour per synthetic spectrum, always used in this order
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# ---------------------------------------------------------------------------
# Find the run: the one containing the newest convolved spectrum
# ---------------------------------------------------------------------------

newest = max(home.glob("runs/TIC*/abundance_tests/*/cvl/*.cvl"), key=lambda p: p.stat().st_mtime)
run = newest.parent.parent
line = float(run.name.split("_Fe_")[1])

# File names start with the abundance: "<A(Fe)>_<wmin>-<wmax>_nlte-<T|F>_<atmosphere>_hermes.cvl"
cvls = sorted((run / "cvl").glob("*.cvl"), key=lambda p: float(p.name.split("_")[0]))
assert len(cvls) <= len(COLORS), f"{len(cvls)} spectra but only {len(COLORS)} colours"

# ---------------------------------------------------------------------------
# Observed spectrum: the one listed in the star config saved with the run
# ---------------------------------------------------------------------------

with (run / "config.yaml").open() as f:
    config = yaml.safe_load(f)

obs_wl, obs_flux = np.loadtxt(home / config["spectra"]["normalized"][0], delimiter=",", skiprows=1, unpack=True)
obs_wl = obs_wl / (1 + (config["spectra"].get("rv") or 0.0) / C_KMS)     # shift to rest frame

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 5), layout="constrained")

for cvl, color in zip(cvls, COLORS):
    wl, flux = np.loadtxt(cvl, usecols=(0, 1), unpack=True)
    keep = (wl > wl[0] + edge) & (wl < wl[-1] - edge)
    abu, _, nlte = cvl.name.split("_")[:3]
    label = f"A(Fe) = {abu} ({'NLTE' if nlte == 'nlte-T' else 'LTE'})"
    ax.plot(wl[keep], flux[keep] * 1.002, color=color, lw=1.5, label=label, zorder=2)

lo, hi = line - 1, line + 1
in_window = (obs_wl > lo) & (obs_wl < hi)
ax.scatter(obs_wl[in_window], obs_flux[in_window], color="#0b0b0b", label="observed", zorder=1, s=2)
ax.axvline(line, color="#8a8984", ls="--", lw=0.8, label=f"{line} Å", zorder=0)

ax.set_xlim(lo, hi)
ax.set_xlabel("Wavelength (Å)")
ax.set_ylabel("Normalised flux")
ax.set_title(f"{config['name']}: {run.name}")
ax.grid(color="#d9d8d3", lw=0.5)
ax.spines[["top", "right"]].set_visible(False)
fig.legend(frameon=False, loc="outside right upper")

out = run / "abundance_scan.png"
fig.savefig(out, dpi=200)
print(f"run:  {run}\nplot: {out}")
plt.show()
