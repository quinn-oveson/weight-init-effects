import argparse
import csv
import os
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
ARM_COLOR = {"cold": "#2a78d6", "warm_all": "#eb6834", "warm_all_lr_ramp": "#7d4fc4",
             "frozen": "#1baf7a"}
ARM_ORDER = ["cold", "warm_all", "warm_all_lr_ramp", "frozen"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRID,
    "axes.labelcolor": INK, "axes.titlesize": 10, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False,
})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


def load(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def lr_colors(lrs):
    # Learning rate is ordered magnitude, so a sequential single-hue ramp, not categorical.
    return {lr: plt.cm.Blues(t) for lr, t in zip(sorted(lrs), np.linspace(0.35, 0.95, len(lrs)))}


def best_lr(rows, arm, nz, col="best_val_mse_clean"):
    # Min over steps within each seed, then mean over seeds, so trajectory shape cannot rank LRs.
    sel = [r for r in rows if r["arm"] == arm and float(r["noise"]) == nz and r.get(col)]
    if not sel:
        return None
    last = max(int(r["cycle"]) for r in sel)
    per = defaultdict(lambda: defaultdict(list))
    for r in sel:
        if int(r["cycle"]) == last:
            per[float(r["lr"])][int(r["seed"])].append(float(r[col]))
    acc = {lr: np.mean([min(v) for v in d.values()]) for lr, d in per.items() if d}
    return min(acc, key=acc.get) if acc else None


def crop_lead(ax, xs, series_list, frac=0.98):
    # Relative to each curve's own plateau, since a fixed NRMSE threshold is never reached.
    hi = 0.0
    for arr in series_list:
        m = np.asarray(arr, dtype=float).mean(0)
        m = np.where(np.isfinite(m), m, -np.inf)
        if not np.isfinite(m).any():
            continue
        over = np.nonzero(m >= frac * m.max())[0]
        hi = max(hi, xs[over[0]] if len(over) else xs[-1])
    ax.set_xlim(0, min(xs[-1], hi * 1.4) if hi > 0 else xs[-1])


def by(rows, *keys):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def spread(ax, xs, series, color, label):
    # Mean line with the inter-seed range as a band; seeds are never averaged away silently.
    arr = np.array(series, dtype=float)
    ax.plot(xs, arr.mean(0), color=color, lw=2, label=label)
    if arr.shape[0] > 1:
        ax.fill_between(xs, arr.min(0), arr.max(0), color=color, alpha=0.18, lw=0)


# ---------------------------------------------------------------- E1

def e1_gap(summary, out):
    # Best-stopped validation error at the FIRST transition, by learning rate.
    rows = [r for r in summary if int(r["cycle"]) == 1 and r["arm"] in ("cold", "warm_all")]
    if not rows:
        return "e1_gap: no cycle-1 rows"
    noises = sorted({float(r["noise"]) for r in rows})
    fig, axes = plt.subplots(1, len(noises), figsize=(5.2 * len(noises), 3.8),
                             constrained_layout=True, squeeze=False)
    for ax, nz in zip(axes[0], noises):
        for arm in ("cold", "warm_all"):
            sel = [r for r in rows if r["arm"] == arm and float(r["noise"]) == nz]
            g = by(sel, "lr")
            lrs = sorted(float(k[0]) for k in g)
            means = [np.mean([float(r["best_val_mse_clean"]) for r in g[(f"{lr:g}",)]])
                     if (f"{lr:g}",) in g else np.nan for lr in lrs]
            for lr in lrs:
                pts = [float(r["best_val_mse_clean"]) for r in sel if float(r["lr"]) == lr]
                ax.scatter([lr] * len(pts), pts, color=ARM_COLOR[arm], s=14, alpha=0.5, zorder=3)
            ax.plot(lrs, means, color=ARM_COLOR[arm], lw=2, marker="o", ms=5, label=arm, zorder=4)
        ax.set_xscale("log")
        ax.set_xlabel("learning rate")
        ax.set_ylabel("best-stopped val MSE (clean)")
        ax.set_title(f"noise = {nz:g}")
        ax.legend()
        style(ax)
    fig.suptitle("E1 — best-stopped validation MSE vs learning rate, cycle 1", color=INK)
    fig.savefig(f"{out}/e1_gap.png", dpi=160)
    plt.close(fig)
    return None


def e1_gap_vs_budget(training, summary, out):
    # The gap is a function of where you stop; show the whole function, not one slice.
    rows = [r for r in training if int(r["cycle"]) == 1 and r["arm"] in ("cold", "warm_all")]
    if not rows:
        return "e1_gap_vs_budget: no cycle-1 rows"
    noises = sorted({float(r["noise"]) for r in rows})
    lrs = sorted({float(r["lr"]) for r in rows})
    cmap = lr_colors(lrs)
    fig, axes = plt.subplots(1, len(noises), figsize=(5.6 * len(noises), 4.0),
                             constrained_layout=True, squeeze=False)
    for ax, nz in zip(axes[0], noises):
        for lr in lrs:
            cur = {}
            for arm in ("cold", "warm_all"):
                sel = [r for r in rows if r["arm"] == arm and float(r["noise"]) == nz
                       and float(r["lr"]) == lr]
                acc = defaultdict(list)
                for r in sel:
                    acc[int(r["step"])].append(float(r["val_mse_clean"]))
                cur[arm] = {s: np.mean(v) for s, v in acc.items()}
            common = sorted(set(cur["cold"]) & set(cur["warm_all"]))
            if not common:
                continue
            common = [s for s in common if s >= 10]
            gap = [cur["warm_all"][s] - cur["cold"][s] for s in common]
            ax.plot(common, gap, color=cmap[lr], lw=1.8, label=f"lr {lr:g}")
        ax.axhline(0, color=MUTED, lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("step budget (where you stop)")
        ax.set_ylabel("warm_all − cold  val MSE")
        ax.set_title(f"noise = {nz:g}")
        ax.legend(title="", fontsize=8)
        style(ax)
    fig.suptitle("E1 — warm_all minus cold validation MSE vs step budget, cycle 1", color=INK)
    fig.savefig(f"{out}/e1_gap_vs_budget.png", dpi=160)
    plt.close(fig)
    return None


def best_step_map(summary):
    # (arm, lr, noise, seed, cycle) -> the checkpoint that is actually handed forward.
    return {(r["arm"], float(r["lr"]), float(r["noise"]), int(r["seed"]), int(r["cycle"])):
            int(r["step_of_best"]) for r in summary}


def e1_test_skill(training, summary, out):
    # Held-out skill at the best-stopped checkpoint; higher is better, unlike the MSE panels.
    bs = best_step_map(summary)
    rows = []
    for r in training:
        if int(r["cycle"]) != 1 or r["arm"] not in ("cold", "warm_all"):
            continue
        key = (r["arm"], float(r["lr"]), float(r["noise"]), int(r["seed"]), 1)
        if key in bs and int(r["step"]) == bs[key] and r.get("valid_time_lyap"):
            rows.append(r)
    if not rows:
        return "e1_test_skill: no cycle-1 rows at step_of_best"
    noises = sorted({float(r["noise"]) for r in rows})
    fig, axes = plt.subplots(1, len(noises), figsize=(5.2 * len(noises), 3.8),
                             constrained_layout=True, squeeze=False)
    for ax, nz in zip(axes[0], noises):
        for arm in ("cold", "warm_all"):
            sel = [r for r in rows if r["arm"] == arm and float(r["noise"]) == nz]
            if not sel:
                continue
            acc = defaultdict(list)
            for r in sel:
                acc[float(r["lr"])].append(float(r["valid_time_lyap"]))
            lrs = sorted(acc)
            for lr in lrs:
                ax.scatter([lr] * len(acc[lr]), acc[lr], color=ARM_COLOR[arm], s=14,
                           alpha=0.5, zorder=3)
            ax.plot(lrs, [np.mean(acc[lr]) for lr in lrs], color=ARM_COLOR[arm], lw=2,
                    marker="o", ms=5, label=arm, zorder=4)
        ax.set_xscale("log")
        ax.set_xlabel("learning rate")
        ax.set_ylabel("valid prediction time (Lyapunov times)")
        ax.set_title(f"noise = {nz:g}")
        ax.legend()
        style(ax)
    fig.suptitle("E1 — held-out valid prediction time vs learning rate at the best step, "
                 "cycle 1", color=INK)
    fig.savefig(f"{out}/e1_test_skill.png", dpi=160)
    plt.close(fig)
    return None


def inherited_level(training, summary, arm, lr, nz, cycle):
    # What the arm starts the cycle from: the step-0 eval, else the previous cycle's best.
    z = [float(r["val_mse_clean"]) for r in training
         if r["arm"] == arm and int(r["cycle"]) == cycle and int(r["step"]) == 0
         and float(r["noise"]) == nz and abs(float(r["lr"]) - lr) < 1e-12]
    if z:
        return float(np.mean(z))
    prev = [float(r["best_val_mse_clean"]) for r in summary
            if r["arm"] == arm and int(r["cycle"]) == cycle - 1 and float(r["noise"]) == nz
            and abs(float(r["lr"]) - lr) < 1e-12 and r.get("best_val_mse_clean")]
    return float(np.mean(prev)) if prev else None


def e1_convergence(training, summary, out):
    # Validation vs cumulative steps within cycle 1, each arm's optimum marked.
    rows = [r for r in training if int(r["cycle"]) == 1]
    if not rows:
        return "e1_convergence: no cycle-1 rows"
    noises = sorted({float(r["noise"]) for r in rows})
    fig, axes = plt.subplots(1, len(noises), figsize=(5.4 * len(noises), 3.8),
                             constrained_layout=True, squeeze=False)
    warm = {a: a != "cold" for a in ARM_ORDER}
    for ax, nz in zip(axes[0], noises):
        used = {}
        for arm in ARM_ORDER:
            lr0 = best_lr(rows, arm, nz, "val_mse_clean")
            if lr0 is None:
                continue
            used[arm] = lr0
            sel = [r for r in rows if r["arm"] == arm and float(r["noise"]) == nz
                   and abs(float(r["lr"]) - lr0) < 1e-12]
            if not sel:
                continue
            acc = defaultdict(list)
            for r in sel:
                acc[int(r["step"])].append(float(r["val_mse_clean"]))
            steps = [s for s in sorted(acc) if s > 0]   # log axis cannot render step 0
            if len(steps) < 2:
                continue
            series = np.array([[acc[s][i] for s in steps] for i in range(len(acc[steps[0]]))])
            spread(ax, steps, series, ARM_COLOR[arm], arm)
            m = series.mean(0)
            j = int(np.argmin(m))
            ax.scatter([steps[j]], [m[j]], color=ARM_COLOR[arm], s=55, zorder=5,
                       edgecolor="white", linewidth=1.2)
            # The first eval is already one Adam step in, which at high lr destroys it.
            base = inherited_level(rows, summary, arm, lr0, nz, 1)
            if base is not None and warm[arm]:
                ax.axhline(base, color=ARM_COLOR[arm], lw=1, ls=":", zorder=2)
                ax.annotate(f"{arm} inherits", (steps[0], base), fontsize=7,
                            color=ARM_COLOR[arm], va="bottom", xytext=(0, 2),
                            textcoords="offset points")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("gradient steps within cycle 1")
        ax.set_ylabel("val MSE (clean)")
        ax.set_title(f"noise = {nz:g}   " + ", ".join(f"{a} lr {v:g}" for a, v in used.items()),
                     fontsize=8)
        ax.legend()
        style(ax)
    fig.suptitle("E1 — validation MSE vs gradient step within cycle 1; dots mark each minimum",
                 color=INK)
    fig.savefig(f"{out}/e1_convergence.png", dpi=160)
    plt.close(fig)
    return None


# ---------------------------------------------------------------- E2

def e2_gap_by_cycle(summary, out):
    # The Ash & Adams plot: does the penalty accumulate across successive warm starts?
    noises = sorted({float(r["noise"]) for r in summary})
    fig, axes = plt.subplots(1, len(noises), figsize=(5.4 * len(noises), 3.8),
                             constrained_layout=True, squeeze=False)
    for ax, nz in zip(axes[0], noises):
        used = {}
        for arm in ARM_ORDER:
            lr0 = best_lr(summary, arm, nz)
            if lr0 is None:
                continue
            used[arm] = lr0
            sel = [r for r in summary if r["arm"] == arm and float(r["noise"]) == nz
                   and abs(float(r["lr"]) - lr0) < 1e-12]
            if not sel:
                continue
            acc = defaultdict(list)
            for r in sel:
                acc[int(r["cycle"])].append(float(r["best_val_mse_clean"]))
            cyc = sorted(acc)
            series = np.array([[acc[c][i] for c in cyc] for i in range(len(acc[cyc[0]]))])
            spread(ax, cyc, series, ARM_COLOR[arm], arm)
        ax.set_xlabel("cycle")
        ax.set_ylabel("best-stopped val MSE (clean)")
        ax.set_title(f"noise = {nz:g}   " + ", ".join(f"{a} lr {v:g}" for a, v in used.items()),
                     fontsize=8)
        ax.legend()
        style(ax)
    fig.suptitle("E2 — best-stopped validation MSE vs cycle", color=INK)
    fig.savefig(f"{out}/e2_gap_by_cycle.png", dpi=160)
    plt.close(fig)
    return None


GAP_CAP = 1.5  # NRMSE gap is bounded by ~1; beyond that is divergence.
Y_CAP = 1.35   # NRMSE saturates at ~1; diverged rollouts leave the axis rather than set it.


def pick_curve_step(curves_by_step, target):
    # Prefer an exact match, else snap to the nearest logged step as aggregate.gap_vs_lead does.
    steps = sorted(curves_by_step)
    if not steps or target is None:
        return None
    return target if target in curves_by_step else min(steps, key=lambda st: abs(st - target))


def e2_forecast_curves(curves, summary, out, cycles=(0, 3, 6, 9)):
    # Forecast error vs lead for the handoff model, not the end-of-budget one.
    nz = sorted({float(r["noise"]) for r in curves})[-1]
    have = sorted({int(r["cycle"]) for r in curves})
    cycles = [c for c in cycles if c in have] or have[:4]
    bs = best_step_map(summary)
    exact = any(r.get("curve_basis") == "best" for r in curves)
    fig, axes = plt.subplots(1, len(cycles), figsize=(3.6 * len(cycles), 3.6),
                             constrained_layout=True, sharey=True, squeeze=False)
    used, drawn, all_panels = {}, [], []
    for ax, cyc in zip(axes[0], cycles):
        panel = []
        for arm in ARM_ORDER:
            lr0 = best_lr(summary, arm, nz)
            if lr0 is None:
                continue
            used[arm] = lr0
            sel = [r for r in curves if r["arm"] == arm and int(r["cycle"]) == cyc
                   and float(r["noise"]) == nz and abs(float(r["lr"]) - lr0) < 1e-12
                   and (not exact or r.get("curve_basis") == "best")]
            if not sel:
                continue
            per_seed = defaultdict(lambda: defaultdict(dict))
            for r in sel:
                per_seed[int(r["seed"])][int(r["step"])][float(r["lead_hours"])] = \
                    float(r["nrmse"])
            rows = []
            for sd in sorted(per_seed):
                st = pick_curve_step(per_seed[sd], bs.get((arm, lr0, nz, sd, cyc)))
                if st is not None:
                    rows.append(per_seed[sd][st])
            if not rows:
                continue
            xs = sorted(set.intersection(*(set(d) for d in rows)))
            # Clipped, not dropped, so diverged lines exit the axis instead of rescaling it.
            series = np.clip(np.nan_to_num(
                np.array([[d[x] for x in xs] for d in rows], dtype=float),
                nan=Y_CAP * 10, posinf=Y_CAP * 10, neginf=Y_CAP * 10), 0, Y_CAP * 10)
            spread(ax, xs, series, ARM_COLOR[arm], arm)
            panel.append(series)
            drawn = xs
        all_panels.extend(panel)
        ax.axhline(1.0, color=MUTED, lw=1, ls=":")
        ax.set_ylim(0, Y_CAP)
        ax.set_xlabel("lead time (hours)")
        ax.set_title(f"cycle {cyc}")
        style(ax)
        top = ax.twiny()
        top.set_xlim(np.array(ax.get_xlim()) / 120.0 * 1.671)
        top.set_xlabel("Lyapunov times", fontsize=8)
        top.tick_params(labelsize=8)
        top.grid(False)
    if all_panels and drawn:
        crop_lead(axes[0][0], drawn, [np.clip(a, 0, Y_CAP) for a in all_panels])
        xlim = axes[0][0].get_xlim()
        for ax in axes[0]:
            ax.set_xlim(xlim)
        for tw in fig.axes:
            if tw.get_xlabel() == "Lyapunov times":
                tw.set_xlim(np.array(xlim) / 120.0 * 1.671)
    axes[0][0].set_ylabel("normalized RMSE")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"E2 — normalized RMSE vs forecast lead at the handoff checkpoint "
                 f"(noise {nz:g}; " + ", ".join(f"{a} lr {v:g}" for a, v in used.items())
                 + f"); {'exact' if exact else 'nearest logged'} step, "
                 f"dotted line = saturation", color=INK)
    fig.savefig(f"{out}/e2_forecast_curves.png", dpi=160)
    plt.close(fig)
    return None


def e2_gap_vs_lead(gap_rows, summary, out):
    # warm_all minus cold NRMSE vs lead, at cold's tuned LR since the gap needs one shared LR.
    if not gap_rows:
        return "e2_gap_vs_lead: gap_vs_lead.csv missing (run aggregate.py)"
    nz = sorted({float(r["noise"]) for r in gap_rows})[-1]
    lr0 = best_lr(summary, "cold", nz) or min(float(r["lr"]) for r in gap_rows)
    bases = [b for b in ("best", "final") if any(r["basis"] == b for r in gap_rows)]
    cycles = sorted({int(r["cycle"]) for r in gap_rows})
    cmap = {c: plt.cm.Blues(t) for c, t in zip(cycles, np.linspace(0.3, 0.95, len(cycles)))}
    fig, axes = plt.subplots(1, len(bases), figsize=(5.6 * len(bases), 4.0),
                             constrained_layout=True, sharey=True, squeeze=False)
    finite = []
    for ax, basis in zip(axes[0], bases):
        for cyc in cycles:
            sel = [r for r in gap_rows if r["basis"] == basis and int(r["cycle"]) == cyc
                   and float(r["noise"]) == nz and abs(float(r["lr"]) - lr0) < 1e-12]
            if not sel:
                continue
            acc = defaultdict(list)
            for r in sel:
                acc[float(r["lead_hours"])].append(float(r["gap"]))
            xs = sorted(acc)
            ys = np.array([np.mean(acc[x]) for x in xs], dtype=float)
            finite.extend(ys[np.isfinite(ys) & (np.abs(ys) <= GAP_CAP)])
            ys = np.clip(np.nan_to_num(ys, nan=GAP_CAP * 10, posinf=GAP_CAP * 10,
                                       neginf=-GAP_CAP * 10), -GAP_CAP * 10, GAP_CAP * 10)
            ax.plot(xs, ys, color=cmap[cyc], lw=1.8, label=f"cycle {cyc}")
        ax.axhline(0, color=MUTED, lw=1, ls=":")
        ax.set_xlabel("lead time (hours)")
        ax.set_title(f"basis: {basis}-stopped" if basis == "best" else "basis: end of budget")
        style(ax)
    # NRMSE is bounded near 1, so a gap outside GAP_CAP is divergence, not signal.
    if finite:
        pad = 0.1 * max(np.ptp(finite), 1e-6)
        axes[0][0].set_ylim(min(finite) - pad, max(finite) + pad)
    axes[0][0].set_ylabel("warm_all − cold  NRMSE")
    axes[0][-1].legend(fontsize=8, ncol=2)
    fig.suptitle(f"E2 — warm_all minus cold NRMSE vs forecast lead "
                 f"(noise {nz:g}, lr {lr0:g})", color=INK)
    fig.savefig(f"{out}/e2_gap_vs_lead.png", dpi=160)
    plt.close(fig)
    return None


def e2_quality_and_cost(summary, out):
    # Steps spent per cycle are equal by construction, so cost means cumulative step_of_best.
    noises = sorted({float(r["noise"]) for r in summary})
    if not noises:
        return "e2_quality_and_cost: no rows"
    fig, axes = plt.subplots(len(noises), 2, figsize=(11.2, 3.9 * len(noises)),
                             constrained_layout=True, squeeze=False)
    for row, nz in zip(axes, noises):
        ax_q, ax_c = row
        used, totals = {}, {}
        for arm in ARM_ORDER:
            lr0 = best_lr(summary, arm, nz)
            if lr0 is None:
                continue
            sel = [r for r in summary if r["arm"] == arm and float(r["noise"]) == nz
                   and abs(float(r["lr"]) - lr0) < 1e-12]
            if not sel:
                continue
            used[arm] = lr0
            cyc = sorted({int(r["cycle"]) for r in sel})
            per_seed = defaultdict(dict)
            for r in sel:
                per_seed[int(r["seed"])][int(r["cycle"])] = r
            # Only seeds with every cycle present, so the cumulative sum is never short.
            seeds = sorted(s for s, d in per_seed.items() if set(d) >= set(cyc))
            if not seeds:
                continue
            q = np.array([[float(per_seed[s][c]["best_val_mse_clean"]) for c in cyc]
                          for s in seeds])
            st = np.cumsum(np.array([[int(per_seed[s][c]["step_of_best"]) for c in cyc]
                                     for s in seeds]), axis=1)
            spread(ax_q, cyc, q, ARM_COLOR[arm], arm)
            spread(ax_c, cyc, st, ARM_COLOR[arm], arm)
            totals[arm] = st.mean(0)[-1]

        ax_q.set_ylabel("best-stopped val MSE (clean)")
        ax_c.set_ylabel("cumulative steps to best model")
        for ax in (ax_q, ax_c):
            ax.set_xlabel("cycle")
            style(ax)
        lrs = ", ".join(f"{a} lr {v:g}" for a, v in used.items())
        ax_q.set_title(f"noise = {nz:g}   ({lrs})", fontsize=8)
        saved = ""
        if {"cold", "warm_all"} <= set(totals) and totals["warm_all"] > 0:
            saved = f"   cold/warm_all = {totals['cold'] / totals['warm_all']:.2f}x"
        ax_c.set_title(f"noise = {nz:g}{saved}", fontsize=8)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("E2 — best-stopped validation MSE and cumulative steps to best, vs cycle",
                 color=INK)
    fig.savefig(f"{out}/e2_quality_and_cost.png", dpi=160)
    plt.close(fig)
    return None


def e2_diagnostics(summary, out):
    # Plasticity diagnostics; descriptive, not the thesis.
    nz = sorted({float(r["noise"]) for r in summary})[-1]
    fields = [("effective_rank", "effective rank"), ("dormant_frac", "dormant channel fraction"),
              ("weight_norm", "weight norm")]
    fig, axes = plt.subplots(1, len(fields), figsize=(3.5 * len(fields), 3.4),
                             constrained_layout=True, squeeze=False)
    for ax, (col, lab) in zip(axes[0], fields):
        for arm in ARM_ORDER:
            lr0 = best_lr(summary, arm, nz)
            if lr0 is None:
                continue
            sel = [r for r in summary if r["arm"] == arm and float(r["noise"]) == nz
                   and abs(float(r["lr"]) - lr0) < 1e-12 and r.get(col) not in (None, "")]
            if not sel:
                continue
            acc = defaultdict(list)
            for r in sel:
                acc[int(r["cycle"])].append(float(r[col]))
            cyc = sorted(acc)
            series = np.array([[acc[c][i] for c in cyc] for i in range(len(acc[cyc[0]]))])
            spread(ax, cyc, series, ARM_COLOR[arm], arm)
        ax.set_xlabel("cycle")
        ax.set_ylabel(lab)
        style(ax)
    axes[0][0].legend(fontsize=8)
    n_bound = sum(1 for r in summary if r.get("best_at_boundary") == "1")
    fig.suptitle(f"E2 — effective rank, dormant fraction and weight norm vs cycle "
                 f"(noise {nz:g}, per-arm tuned lr; {n_bound} best-at-boundary)", color=INK)
    fig.savefig(f"{out}/e2_diagnostics.png", dpi=160)
    plt.close(fig)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["e1", "e2"], required=True,
                   help="e1 = first transition only; analyze and write it up BEFORE e2")
    p.add_argument("--results_dir", default=None)
    p.add_argument("--out", default="figures/e1e2")
    args = p.parse_args()

    import sweep_config as C
    rd = args.results_dir or C.RESULTS_DIR
    os.makedirs(args.out, exist_ok=True)

    summary = load(f"{rd}/all_summary.csv")
    training = load(f"{rd}/all_training.csv")
    if not summary:
        raise SystemExit(f"no all_summary.csv in {rd} — run aggregate.py first")

    notes = []
    if args.stage == "e1":
        notes += [e1_gap(summary, args.out),
                  e1_test_skill(training, summary, args.out),
                  e1_gap_vs_budget(training, summary, args.out),
                  e1_convergence(training, summary, args.out)]
    else:
        curves = load(f"{rd}/all_curves.csv")
        gaps = load(f"{rd}/gap_vs_lead.csv")
        notes += [e2_gap_by_cycle(summary, args.out),
                  e2_forecast_curves(curves, summary, args.out) if curves else "e2_forecast_curves: no curves",
                  e2_gap_vs_lead(gaps, summary, args.out),
                  e2_quality_and_cost(summary, args.out),
                  e2_diagnostics(summary, args.out)]

    cells = len({(r["arm"], r["lr"], r["noise"], r["seed"]) for r in summary})
    print(f"stage {args.stage}: {cells}/{len(C.CELLS)} cells present -> {args.out}/")
    for n in [x for x in notes if x]:
        print(f"  SKIPPED {n}")
    if args.stage == "e1":
        print("  reminder: write up E1 before running --stage e2")


if __name__ == "__main__":
    main()
