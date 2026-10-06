#!/usr/bin/env python3
"""
Click through the line fits of one find_stellar_parameters run and keep or reject each one.

    python review_fits.py                         # the latest run of SELECTED_STAR
    python review_fits.py <run dir>               # a specific run (e.g. a _flagged rerun)
    python review_fits.py --overview [<run dir>]  # no window: (re)write <run dir>/overview.png, all fits in one
                                                  # picture, with the review decisions and current flags marked
                                                  # (for ssh sessions without X; flag lines by editing
                                                  # data/selected_lines/<star>/fe_line_flags.yaml)

Buttons, or keys:
    left / right   previous / next line
    k              keep                    -> the line is removed from the star's flag file
    r              reject with the reason in the text box
    1 2 3          reject with one of the preset REASONS
    u              undo the decision for this line (restores the flag file entry as it was)
    q              quit
Click in the text box to type a reason (Enter to finish); keys go to the text box while it is active.

Every decision is saved at once to
    <run dir>/review.yaml                              what was decided in this run, with reason and time
    data/selected_lines/<star>/fe_line_flags.yaml      rejected lines are added, kept lines removed;
                                                       find_stellar_parameters leaves the flagged lines out
"Keep" only lifts a manual flag: lines rejected by the automatic cuts (chi2, sigma clipping) stay rejected.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))     # find fe_line_fit.py next to this script
from fe_line_fit import RUNS_DIR, load_line_flags, load_lines, plot_overview, read_bestfit, save_line_flags, totals_text

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

SELECTED_STAR = "HD196944"
RUN_DIR = None                       # None -> newest run in runs/<star>/fe_abundance/ (a command-line argument wins)
REASONS = ["rerun: poor fit", "rerun: doubtful continuum normalisation", "rerun: blended"]   # keys 1, 2, 3


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def latest_run(star: str) -> Path:
    runs = sorted((RUNS_DIR / star / "fe_abundance").glob("*/summary.yaml"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit(f"no finished runs in {RUNS_DIR / star / 'fe_abundance'}")
    return runs[-1].parent


def read_rows(run: Path) -> list[dict]:
    with (run / "lines.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def num(r: dict, key: str) -> float:
    if key == "broad_kms" and key not in r:              # runs made before the rename
        key = "fwhm_kms"
    try:
        return float(r.get(key, "nan"))
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# The viewer
# ---------------------------------------------------------------------------

class Reviewer:
    def __init__(self, run: Path):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, TextBox

        self.plt = plt
        self.run = run
        summary = yaml.safe_load((run / "summary.yaml").read_text())
        self.star = summary["star"]
        self.rows = sorted(read_rows(run), key=lambda r: num(r, "wavelength"))
        line_file = Path(summary["line_file"])
        self.windows = {r["line_id"]: (r["fit_left"], r["fit_right"]) for r in load_lines(line_file)}
        self.flags_path = line_file.with_name("fe_line_flags.yaml")
        self.flags0 = load_line_flags(self.flags_path)          # as they were before this session (for undo)
        self.review_path = run / "review.yaml"
        self.review: dict[str, dict] = (yaml.safe_load(self.review_path.read_text()) or {}) \
            if self.review_path.exists() else {}
        self.i = next((k for k, r in enumerate(self.rows) if r["line_id"] not in self.review), 0)

        for key in [k for k in plt.rcParams if k.startswith("keymap.")]:  # free our keys from matplotlib's
            plt.rcParams[key] = [c for c in plt.rcParams[key]
                                 if c not in ("k", "r", "u", "q", "1", "2", "3", "left", "right")]

        self.fig = plt.figure(figsize=(12, 7))
        self.fig.canvas.manager.set_window_title(f"review_fits: {run.name}")
        self.ax = self.fig.add_axes([0.07, 0.42, 0.90, 0.46])
        self.axr = self.fig.add_axes([0.07, 0.24, 0.90, 0.15], sharex=self.ax)
        self.status = self.fig.text(0.07, 0.905, "", fontsize=12, weight="bold")
        self.header = self.fig.text(0.07, 0.94, "", fontsize=10)

        self.box = TextBox(self.fig.add_axes([0.20, 0.13, 0.55, 0.05]), "reject reason ", initial=REASONS[0])
        self.box.on_submit(lambda _: self.box.stop_typing())
        self.buttons = []
        for x, label, cb in ((0.07, "< prev", lambda _: self.move(-1)), (0.25, "keep (k)", lambda _: self.keep()),
                             (0.43, "reject (r)", lambda _: self.reject(self.box.text)),
                             (0.61, "undo (u)", lambda _: self.undo()), (0.79, "next >", lambda _: self.move(1))):
            b = Button(self.fig.add_axes([x, 0.04, 0.16, 0.06]), label)
            b.on_clicked(cb)
            self.buttons.append(b)
        self.fig.text(0.20, 0.105, "keys 1-3 reject with: " +
                      "   ".join(f"{n + 1}: {r}" for n, r in enumerate(REASONS)), fontsize=8, color="0.3")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()

    # --- decisions ---------------------------------------------------------------------------------

    def save(self) -> None:
        self.review_path.write_text(yaml.safe_dump(self.review, sort_keys=True, allow_unicode=True))
        flags = load_line_flags(self.flags_path)
        for lid, d in self.review.items():
            if d["decision"] == "reject":
                flags[lid] = d["reason"]
            else:
                flags.pop(lid, None)
        save_line_flags(self.flags_path, flags)

    def decide(self, decision: str, reason: str = "") -> None:
        lid = self.rows[self.i]["line_id"]
        self.review[lid] = dict(decision=decision, reason=reason, time=time.strftime("%Y-%m-%d %H:%M:%S"))
        self.save()
        self.move(1)

    def keep(self) -> None:
        self.decide("keep")

    def reject(self, reason: str) -> None:
        self.decide("reject", reason.strip() or REASONS[0])

    def undo(self) -> None:
        lid = self.rows[self.i]["line_id"]
        if self.review.pop(lid, None) is None:
            return
        self.review_path.write_text(yaml.safe_dump(self.review, sort_keys=True, allow_unicode=True))
        flags = load_line_flags(self.flags_path)
        if lid in self.flags0:
            flags[lid] = self.flags0[lid]
        else:
            flags.pop(lid, None)
        save_line_flags(self.flags_path, flags)
        self.draw()

    def move(self, step: int) -> None:
        self.i = (self.i + step) % len(self.rows)
        self.draw()

    def on_key(self, event) -> None:
        if self.box.capturekeystrokes:                         # typing a reason
            return
        actions = {"left": lambda: self.move(-1), "right": lambda: self.move(1), "k": self.keep,
                   "r": lambda: self.reject(self.box.text), "u": self.undo,
                   "q": lambda: self.plt.close(self.fig)}
        actions.update({str(n + 1): (lambda s=s: self.reject(s)) for n, s in enumerate(REASONS)})
        if event.key in actions:
            actions[event.key]()

    # --- drawing -----------------------------------------------------------------------------------

    def draw(self) -> None:
        r = self.rows[self.i]
        lid = r["line_id"]
        self.ax.clear()
        self.axr.clear()
        fit = self.run / "lines" / lid / "bestfit.csv"
        if fit.exists():
            wl, obs, mod, used = read_bestfit(fit)
            left, right = self.windows.get(lid, (np.nan, np.nan))
            for a in (self.ax, self.axr):
                a.axvspan(left, right, color="0.92", zorder=0)
                a.axvline(num(r, "wavelength"), color="0.6", lw=0.8, zorder=0)
                for bnd in (num(r, "chi2_left"), num(r, "chi2_right")):
                    if np.isfinite(bnd):
                        a.axvline(bnd, color="tab:blue", lw=0.8, ls="--", zorder=0)
            self.ax.plot(wl[used], obs[used], "k.", ms=3, label="observed")
            if (~used).any():
                self.ax.plot(wl[~used], obs[~used], ".", color="0.7", ms=3, label="not in the fit")
            self.ax.plot(wl, mod, "r-", lw=1.2, label="best fit")
            self.ax.legend(fontsize=8, loc="lower left")
            self.axr.plot(wl[used], (obs - mod)[used], "k.", ms=2)
            self.axr.plot(wl[~used], (obs - mod)[~used], ".", color="0.7", ms=2)
            self.axr.axhline(0, color="r", lw=0.8)
            lo, hi = min(obs.min(), mod.min()), max(obs.max(), mod.max())
            self.ax.set_ylim(lo - 0.05 * (hi - lo) - 0.005, hi + 0.05 * (hi - lo) + 0.005)
            self.ax.set_xlim(wl.min(), wl.max())
        else:
            self.ax.text(0.5, 0.5, f"no fit ({r.get('status', '')})", ha="center", transform=self.ax.transAxes)
        self.ax.set_ylabel("normalised flux")
        self.axr.set_ylabel("obs - fit")
        self.axr.set_xlabel("Wavelength [A]  (grey band: fit_left..fit_right of the line CSV)")
        self.ax.tick_params(labelbottom=False)

        others = f"   other selected lines in window: {r['other_lines_in_window']}" \
            if r.get("other_lines_in_window") else ""
        self.header.set_text(
            f"[{self.i + 1}/{len(self.rows)}]  {lid}   EP {num(r, 'ep'):.2f} eV   EW {num(r, 'ew_mA'):.0f} mA   "
            f"A(Fe) = {num(r, 'A'):.3f} +/- {num(r, 'sigma'):.3f}   chi2_red {num(r, 'chi2_red'):.2f}   "
            f"broad {num(r, 'broad_kms'):.1f} km/s   RV {num(r, 'dv_kms'):+.2f} km/s   cont {num(r, 'cont'):.3f}\n"
            f"run: {r.get('selection', '?')}   |   NLTE levels: {r.get('nlte_levels', '?')}{others}   |   "
            f"fit region {num(r, 'chi2_left'):.2f}..{num(r, 'chi2_right'):.2f} ({r.get('region_source', 'older run')})")
        d = self.review.get(lid)
        if d is None:
            flag = load_line_flags(self.flags_path).get(lid)
            text, color = (f"not reviewed (flagged: {flag})", "tab:orange") if flag else ("not reviewed", "0.4")
        elif d["decision"] == "keep":
            text, color = "KEPT", "tab:green"
        else:
            text, color = f"REJECTED: {d['reason']}", "tab:red"
            self.box.set_val(d["reason"])
        n_done = sum(1 for x in self.rows if x["line_id"] in self.review)
        self.status.set_text(f"{text}      ({n_done}/{len(self.rows)} reviewed)")
        self.status.set_color(color)
        self.fig.canvas.draw_idle()


def write_overview(run: Path) -> Path:
    """overview.png of a run, marking review decisions and flags set since the run."""
    summary = yaml.safe_load((run / "summary.yaml").read_text())
    line_file = Path(summary["line_file"])
    rows = read_rows(run)
    review = (yaml.safe_load((run / "review.yaml").read_text()) or {}) if (run / "review.yaml").exists() else {}
    flags = load_line_flags(line_file.with_name("fe_line_flags.yaml"))
    marks = {}
    for r in rows:
        lid = r["line_id"]
        if lid in review:
            d = review[lid]
            marks[lid] = "review: KEEP" if d["decision"] == "keep" else f"review: REJECT ({d['reason']})"
        elif lid in flags and not r.get("selection", "").startswith("flagged"):
            marks[lid] = f"flagged since this run: {flags[lid]}"
        elif r.get("selection", "").startswith("flagged") and lid not in flags:
            marks[lid] = "flag removed since this run"
    mode = "NLTE" if summary.get("nlte") else "LTE"
    path = run / "overview.png"
    plot_overview(rows, run, path, windows={r["line_id"]: (r["fit_left"], r["fit_right"]) for r in load_lines(line_file)},
                  title=f"{summary['star']} {mode}: Teff={summary['teff']:.0f} logg={summary['logg']:.2f} "
                        f"vmic={summary['vmic']:.2f} [Fe/H]={summary['feh_atmosphere']:+.2f}   ({run.name})"
                        + (f"\nnote: {summary['note']}" if summary.get("note") else ""),
                  marks=marks, totals=totals_text(summary["results"]))
    return path


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--overview"]
    run = Path(args[0]).expanduser() if args else (Path(RUN_DIR) if RUN_DIR else latest_run(SELECTED_STAR))
    if not (run / "lines.csv").exists() or not (run / "summary.yaml").exists():
        raise SystemExit(f"{run} is not a finished find_stellar_parameters run")
    if "--overview" in sys.argv:
        print(f"wrote {write_overview(run)}")
        return
    print(f"reviewing {run}")
    rv = Reviewer(run)
    rv.plt.show()
    kept = sum(d["decision"] == "keep" for d in rv.review.values())
    print(f"{len(rv.review)} of {len(rv.rows)} lines reviewed ({kept} kept, {len(rv.review) - kept} rejected)\n"
          f"decisions: {rv.review_path}\nflags:     {rv.flags_path}")


if __name__ == "__main__":
    main()
