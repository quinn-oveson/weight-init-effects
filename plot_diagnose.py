"""Reads saturation_summary.csv and answers whether more data helps, per width."""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np

from plot_e1e2 import INK, MUTED, plt, style


def load(path):
    if not os.path.exists(path):
        raise SystemExit(f"no {path} -- run: python diagnose_saturation.py --collect")
    with open(path, newline="") as fh:
        return [r for r in csv.DictReader(fh) if r.get("best_val_mse_clean")]


def tuned(rows, col):
    # One value per (noise, hidden, n_train, seed) at that cell's best LR, keyed for lookup.
    per = defaultdict(dict)
    for r in rows:
        k = (float(r["noise"]), int(r["hidden"]), int(r["n_train"]), int(r["seed"]))
        lr, v, sel = float(r["lr"]), float(r[col]), float(r["best_val_mse_clean"])
        if k not in per or sel < per[k][0]:
            per[k] = (sel, v, lr)
    return {k: v[1] for k, v in per.items()}


def series(vals, nz, h, ns):
    # Mean and inter-seed range at each n_train, skipping points no seed reached.
    xs, mean, lo, hi = [], [], [], []
    for n in ns:
        v = [vals[k] for k in vals if k[0] == nz and k[1] == h and k[2] == n]
        if v:
            xs.append(n)
            mean.append(np.mean(v))
            lo.append(np.min(v))
            hi.append(np.max(v))
    return xs, np.array(mean), np.array(lo), np.array(hi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/diagnose")
    p.add_argument("--out", default="figures/diagnose")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = load(os.path.join(args.results_dir, "saturation_summary.csv"))
    noises = sorted({float(r["noise"]) for r in rows})
    hiddens = sorted({int(r["hidden"]) for r in rows})
    ns = sorted({int(r["n_train"]) for r in rows})
    cmap = {h: plt.cm.viridis(t) for h, t in zip(hiddens, np.linspace(0.15, 0.85, len(hiddens)))}

    cols = [("best_val_mse_clean", "best-stopped val MSE (clean)"),
            ("best_test_mse_clean", "best-stopped test MSE (clean)"),
            ("best_train_mse_clean", "best-stopped train MSE (clean)"),
            ("step_of_best", "step of best")]
    fig, axes = plt.subplots(len(noises), len(cols),
                             figsize=(4.0 * len(cols), 3.5 * len(noises)),
                             constrained_layout=True, squeeze=False)
    for row, nz in zip(axes, noises):
        for ax, (col, lab) in zip(row, cols):
            vals = tuned(rows, col)
            for h in hiddens:
                xs, m, lo, hi = series(vals, nz, h, ns)
                if not xs:
                    continue
                ax.plot(xs, m, color=cmap[h], lw=2, marker="o", ms=4, label=f"hidden {h}")
                if np.any(hi > lo):
                    ax.fill_between(xs, lo, hi, color=cmap[h], alpha=0.18, lw=0)
            ax.set_xscale("log")
            if col != "step_of_best":
                ax.set_yscale("log")
            if col == "best_train_mse_clean" and nz > 0:
                ax.axhline(2.0 * nz ** 2, color=MUTED, lw=1, ls=":")
            ax.set_xlabel("training samples")
            ax.set_ylabel(lab)
            ax.set_title(f"noise = {nz:g}", fontsize=9)
            style(ax)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Saturation diagnostic — best-stopped error vs training-set size, by width",
                 color=INK)
    fig.savefig(f"{args.out}/diagnose_saturation.png", dpi=160)
    plt.close(fig)

    val, tr = tuned(rows, "best_val_mse_clean"), tuned(rows, "train_vs_noise_floor")
    print(f"{'noise':>6} {'hidden':>7} {'n=min':>10} {'n=max':>10} {'ratio':>7}  "
          f"{'train/floor':>11}   verdict")
    for nz in noises:
        for h in hiddens:
            xs, m, _, _ = series(val, nz, h, ns)
            if len(xs) < 2:
                continue
            ratio = m[-1] / m[0]
            f = [tr[k] for k in tr if k[0] == nz and k[1] == h and np.isfinite(tr[k])]
            verdict = "FLAT (no data response)" if ratio > 0.9 else "falls with data"
            print(f"{nz:>6g} {h:>7} {m[0]:>10.6f} {m[-1]:>10.6f} {ratio:>7.2f}  "
                  f"{np.mean(f) if f else float('nan'):>11.2f}   {verdict}")
    print(f"\nn_train swept: {ns}")
    print("ratio ~1 at every width => saturation; falls at 64/128 but flat at 512 "
          "=> over-parameterisation.")
    print("train/floor < 1 => fit below the observation noise, i.e. memorising.")
    print(f"\nfigure -> {args.out}/diagnose_saturation.png")


if __name__ == "__main__":
    main()
