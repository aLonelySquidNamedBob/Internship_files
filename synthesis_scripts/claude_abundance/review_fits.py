#!/usr/bin/env python3
"""
Click through the line fits of one find_stellar_parameters run and keep, flag or reject each one.

    python review_fits.py                         # the latest run of SELECTED_STAR
    python review_fits.py <run dir>               # a specific run (e.g. a _flagged rerun)
    python review_fits.py --overview [<run dir>]  # no window: (re)write <run dir>/overview.png, all fits in one
                                                  # picture, with the review decisions and current flags marked
                                                  # (for ssh sessions without X; flag lines by editing
                                                  # data/selected_lines/<star>/fe_line_flags.yaml)
    python review_fits.py --region <line_id> <left> <right>   # no window: set the chi^2 limits (fit region, A)
    python review_fits.py --region <line_id> clear            # of a line in the line CSV / back to the default

Buttons, or keys:
    left / right   previous / next line
    k              keep                    -> the line's entry is removed from the star's flag file
    f              flag with the reason in the text box   -> marked only, still used in the means and slopes
    1 2 3          flag with one of the preset REASONS
    r              reject with the reason in the text box -> written as "reject: <reason>", left out
    u              undo the decision for this line (restores the flag file entry as it was)
    drag on plot   choose new chi^2 limits (fit region) for this line (blue band)
    m              save the chosen limits to chi2_left / chi2_right in the line CSV (used from the next run on)
    x              clear the limits of this line in the line CSV (back to the default window)
    q              quit
Click in the text box to type a reason (Enter to finish); keys go to the text box while it is active.

Every decision is saved at once to
    <run dir>/review.yaml                              what was decided in this run, with reason and time
    data/selected_lines/<star>/fe_line_flags.yaml      flagged / rejected lines are added, kept lines removed;
                                                       find_stellar_parameters marks the flagged lines and leaves
                                                       the ones whose reason starts with "reject" out
"Keep" only lifts a manual entry: lines rejected by the automatic cuts (chi2, sigma clipping) stay rejected.
Fit regions are written straight into the star's selected-lines CSV (chi2_left / chi2_right) and take effect in
the next find_stellar_parameters run; the plot shows the run's region (dashed) and the one in the CSV (blue text).
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))     # find fe_line_fit.py next to this script
from fe_line_fit import (RUNS_DIR, is_reject, load_line_flags, load_lines, load_star_config, plot_overview,
                         read_bestfit, save_line_flags, set_line_region, totals_text)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

SELECTED_STAR = "TIC396792499"
RUN_DIR = None                       # None -> newest run in runs/<star>/fe_abundance/ (a command-line argument wins)
REASONS = ["rerun: poor fit", "rerun: doubtful continuum normalisation", "rerun: blended"]   # keys 1-3 (flag)


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
        from matplotlib.widgets import Button, SpanSelector, TextBox

        self.plt = plt
        self.run = run
        summary = yaml.safe_load((run / "summary.yaml").read_text())
        self.star = summary["star"]
        self.rows = sorted(read_rows(run), key=lambda r: num(r, "wavelength"))
        line_file = Path(summary["line_file"])
        self.line_file = line_file
        self.windows = {r["line_id"]: (r["fit_left"], r["fit_right"]) for r in load_lines(line_file)}
        self.pending: tuple[float, float] | None = None          # region dragged on the plot, not saved yet
        self.flags_path = line_file.with_name("fe_line_flags.yaml")
        self.flags0 = load_line_flags(self.flags_path)          # as they were before this session (for undo)
        self.review_path = run / "review.yaml"
        self.review: dict[str, dict] = (yaml.safe_load(self.review_path.read_text()) or {}) \
            if self.review_path.exists() else {}
        self.i = next((k for k, r in enumerate(self.rows) if r["line_id"] not in self.review), 0)

        for key in [k for k in plt.rcParams if k.startswith("keymap.")]:  # free our keys from matplotlib's
            plt.rcParams[key] = [c for c in plt.rcParams[key]
                                 if c not in ("k", "f", "r", "u", "q", "m", "x", "1", "2", "3", "left", "right")]

        self.fig = plt.figure(figsize=(12, 7))
        self.fig.canvas.manager.set_window_title(f"review_fits: {run.name}")
        self.ax = self.fig.add_axes([0.07, 0.42, 0.90, 0.46])
        self.axr = self.fig.add_axes([0.07, 0.24, 0.90, 0.15], sharex=self.ax)
        self.status = self.fig.text(0.07, 0.905, "", fontsize=12, weight="bold")
        self.region_text = self.fig.text(0.55, 0.905, "", fontsize=10, color="tab:blue")
        self.header = self.fig.text(0.07, 0.94, "", fontsize=10)

        self.box = TextBox(self.fig.add_axes([0.20, 0.13, 0.55, 0.05]), "reason ", initial=REASONS[0])
        self.box.on_submit(lambda _: self.box.stop_typing())
        self.buttons = []
        for x, label, cb in ((0.07, "< prev", lambda _: self.move(-1)), (0.22, "keep (k)", lambda _: self.keep()),
                             (0.37, "flag (f)", lambda _: self.flag(self.box.text)),
                             (0.52, "reject (r)", lambda _: self.reject(self.box.text)),
                             (0.67, "undo (u)", lambda _: self.undo()), (0.82, "next >", lambda _: self.move(1))):
            b = Button(self.fig.add_axes([x, 0.02, 0.13, 0.05]), label)
            b.on_clicked(cb)
            self.buttons.append(b)
        self.fig.text(0.20, 0.108, "keys 1-3 flag with: " +
                      "   ".join(f"{n + 1}: {r}" for n, r in enumerate(REASONS)), fontsize=8, color="0.3")
        self.fig.text(0.20, 0.083, "drag on the plot: new chi^2 limits   m: save them to the line CSV   "
                      "x: clear them (default window)", fontsize=8, color="tab:blue")
        self.span = SpanSelector(self.ax, self.on_span, "horizontal", useblit=True, interactive=True,
                                 props=dict(alpha=0.15, facecolor="tab:blue"))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()

    # --- decisions ---------------------------------------------------------------------------------

    def save(self) -> None:
        self.review_path.write_text(yaml.safe_dump(self.review, sort_keys=True, allow_unicode=True))
        flags = load_line_flags(self.flags_path)
        for lid, d in self.review.items():
            if d["decision"] == "reject":
                flags[lid] = d["reason"] if is_reject(d["reason"]) else f"reject: {d['reason']}"
            elif d["decision"] == "flag":
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

    def flag(self, reason: str) -> None:
        self.decide("flag", reason.strip() or REASONS[0])

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
        self.pending = None
        self.draw()

    # --- fit regions (chi2_left / chi2_right in the line CSV) ------------------------------------

    def on_span(self, xmin: float, xmax: float) -> None:
        if xmax - xmin > 0.02:
            self.pending = (round(float(xmin), 3), round(float(xmax), 3))
            self.region_text.set_text(f"chosen region {xmin:.3f}..{xmax:.3f}  (press m to save)")
            self.fig.canvas.draw_idle()

    def csv_region(self, lid: str) -> tuple[float, float] | None:
        row = next((r for r in load_lines(self.line_file) if r["line_id"] == lid), None)
        return (row["chi2_left"], row["chi2_right"]) if row and row["chi2_left"] is not None else None

    def save_region(self) -> None:
        lid = self.rows[self.i]["line_id"]
        if self.pending is None:
            self.region_text.set_text("drag on the plot first to choose a region")
        else:
            try:
                set_line_region(self.line_file, lid, *self.pending)
            except ValueError as exc:
                self.region_text.set_text(str(exc))
            else:
                self.pending = None
                self.draw()
                return
        self.fig.canvas.draw_idle()

    def clear_region(self) -> None:
        set_line_region(self.line_file, self.rows[self.i]["line_id"], None, None)
        self.pending = None
        self.draw()

    def on_key(self, event) -> None:
        if self.box.capturekeystrokes:                         # typing a reason
            return
        actions = {"left": lambda: self.move(-1), "right": lambda: self.move(1), "k": self.keep,
                   "f": lambda: self.flag(self.box.text), "r": lambda: self.reject(self.box.text), "u": self.undo,
                   "m": self.save_region, "x": self.clear_region,
                   "q": lambda: self.plt.close(self.fig)}
        actions.update({str(n + 1): (lambda s=s: self.flag(s)) for n, s in enumerate(REASONS)})
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
            wl, obs, mod, used, nofe = read_bestfit(fit)
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
            if np.isfinite(nofe).any():
                self.ax.plot(wl, nofe, "-", color="tab:cyan", lw=1.1, label="no Fe")
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
            f"A(Fe) = {num(r, 'A'):.3f} +/- {num(r, 'sigma'):.3f}   misfit {num(r, 'misfit'):.3f}   chi2_red {num(r, 'chi2_red'):.2f}   "
            f"broad {num(r, 'broad_kms'):.1f} km/s   RV {num(r, 'dv_kms'):+.2f} km/s   cont {num(r, 'cont'):.3f}\n"
            f"run: {r.get('selection', '?')}   |   NLTE levels: {r.get('nlte_levels', '?')}{others}   |   "
            f"fit region {num(r, 'chi2_left'):.2f}..{num(r, 'chi2_right'):.2f} ({r.get('region_source', 'older run')})")
        d = self.review.get(lid)
        if d is None:
            flag = load_line_flags(self.flags_path).get(lid)
            text, color = ((f"not reviewed ({'rejected' if is_reject(flag) else 'flagged'}: {flag})",
                            "tab:red" if is_reject(flag) else "tab:orange") if flag else ("not reviewed", "0.4"))
        elif d["decision"] == "keep":
            text, color = "KEPT", "tab:green"
        elif d["decision"] == "flag":
            text, color = f"FLAGGED (still used): {d['reason']}", "tab:orange"
            self.box.set_val(d["reason"])
        else:
            text, color = f"REJECTED: {d['reason']}", "tab:red"
            self.box.set_val(d["reason"])
        reg = self.csv_region(lid)
        self.region_text.set_text(f"region in line CSV: {reg[0]:.3f}..{reg[1]:.3f} (next run)" if reg
                                  else "region in line CSV: default window")
        if reg:
            for a in (self.ax, self.axr):
                a.axvspan(*reg, color="tab:blue", alpha=0.08, zorder=0)
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
        sel = r.get("selection", "")
        run_note = r.get("flag") or (sel[len("flagged: "):] if sel.startswith("flagged: ") else
                                     sel[len("rejected: by hand ("):-1] if sel.startswith("rejected: by hand (") else "")
        if lid in review:
            d = review[lid]
            marks[lid] = f"review: {d['decision'].upper()}" + (f" ({d['reason']})" if d.get("reason") else "")
        elif flags.get(lid, "") != run_note:
            marks[lid] = (f"flag file changed since this run: {flags[lid]}" if lid in flags
                          else "flag removed since this run")
    mode = "NLTE" if summary.get("nlte") else "LTE"
    path = run / "overview.png"
    plot_overview(rows, run, path, windows={r["line_id"]: (r["fit_left"], r["fit_right"]) for r in load_lines(line_file)},
                  title=f"{summary['star']} {mode}: Teff={summary['teff']:.0f} logg={summary['logg']:.2f} "
                        f"vmic={summary['vmic']:.2f} [Fe/H]={summary['feh_atmosphere']:+.2f}   ({run.name})"
                        + (f"\nnote: {summary['note']}" if summary.get("note") else ""),
                  marks=marks, totals=totals_text(summary["results"]), fit_settings=summary.get("fit_settings"))
    return path


def set_region_cli(argv: list[str]) -> None:
    """--region <line_id> <left> <right> | --region <line_id> clear"""
    i = argv.index("--region")
    args = argv[i + 1:]
    if len(args) not in (2, 3) or (len(args) == 2 and args[1] != "clear"):
        raise SystemExit("usage: review_fits.py --region <line_id> <left> <right>   or   --region <line_id> clear")
    cfg = load_star_config(SELECTED_STAR)
    lid = args[0]
    if args[1] == "clear":
        set_line_region(cfg.line_file, lid, None, None)
        print(f"{lid}: fit region cleared (default window) in {cfg.line_file}")
    else:
        lo, hi = sorted((float(args[1]), float(args[2])))
        set_line_region(cfg.line_file, lid, lo, hi)
        print(f"{lid}: fit region {lo:.3f}..{hi:.3f} written to {cfg.line_file} (used from the next run on)")


def main() -> None:
    if "--region" in sys.argv:
        set_region_cli(sys.argv)
        return
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
