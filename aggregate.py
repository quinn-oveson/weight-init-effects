import argparse
import csv
import glob
import os
import sys
from collections import Counter, defaultdict

import sweep_config as C

KINDS = ("training", "curves", "summary")


def model_key(row):
    # A "model type" is one configuration; seeds are the rows within its table.
    return (row["arm"], float(row["lr"]), float(row["noise"]))


def model_name(key):
    arm, lr, noise = key
    return f"{arm}_lr{lr:g}_noise{noise:g}"


def read_tasks(results_dir, kind):
    rows, files = [], sorted(glob.glob(os.path.join(results_dir, f"task*_{kind}.csv")))
    for f in files:
        with open(f, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows, files


def write_table(path, rows, cols):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def check_completeness(summary_rows):
    # Every (arm, lr, noise, seed) cell should contribute exactly N_CYCLES summary rows.
    seen = Counter((r["arm"], float(r["lr"]), float(r["noise"]), int(r["seed"]))
                   for r in summary_rows)
    expected = {(c["arm"], c["lr"], c["noise"], c["seed"]) for c in C.CELLS}
    missing = sorted(expected - set(seen))
    partial = sorted(k for k, n in seen.items() if n != C.N_CYCLES)
    extra = sorted(set(seen) - expected)
    return missing, partial, extra


def check_invariants(summary_rows):
    problems = []

    # cold and warm_all must differ ONLY in initialization: same steps, same batch order.
    cells = defaultdict(dict)
    for r in summary_rows:
        if r["arm"] in ("cold", "warm_all"):
            key = (float(r["noise"]), float(r["lr"]), int(r["seed"]), int(r["cycle"]))
            cells[key][r["arm"]] = (int(r["cycle_steps"]), r["batch_order_hash"])
    for k, d in cells.items():
        if len(d) == 2:
            if d["cold"][0] != d["warm_all"][0]:
                problems.append(f"step-count mismatch at {k}: {d['cold'][0]} vs {d['warm_all'][0]}")
            if d["cold"][1] != d["warm_all"][1]:
                problems.append(f"batch-order mismatch at {k} — more than init differs")

    unstable = [(r["arm"], r["seed"], r["cycle"]) for r in summary_rows if r["stable"] != "1"]
    if unstable:
        problems.append(f"{len(unstable)} unstable rollouts, e.g. {unstable[:3]}")

    gaps = [float(r["final_train_val_gap"]) for r in summary_rows]
    if gaps:
        problems.append(f"INFO train_val_gap range [{min(gaps):+.5f}, {max(gaps):+.5f}] "
                        f"-- if never positive, the model is not overfitting and the "
                        f"phenomenon cannot appear")
    return problems


def check_curves(curve_rows, sample=200):
    # NRMSE should increase with lead time; flag configurations where it does not.
    by_run = defaultdict(list)
    for r in curve_rows:
        key = (r["arm"], r["lr"], r["noise"], r["seed"], r["cycle"], r["step"])
        by_run[key].append((int(r["lead_step"]), float(r["nrmse"])))
    bad = []
    for key, pts in list(by_run.items())[:sample]:
        pts.sort()
        v = [p[1] for p in pts]
        if v[-1] < v[0]:
            bad.append(key)
    return bad


def gap_vs_lead(curve_rows, out_path):
    # Headline analysis: warm_all minus cold NRMSE vs forecast lead, per seed (never averaged).
    # Flat in lead => generic init penalty; growing => amplified by chaotic error compounding.
    by_arm = {}
    for r in curve_rows:
        if r["arm"] not in ("cold", "warm_all"):
            continue
        # Compare each arm at its own final step of the cycle.
        key = (float(r["noise"]), float(r["lr"]), int(r["seed"]), int(r["cycle"]),
               int(r["lead_step"]))
        slot = by_arm.setdefault(key, {})
        prev = slot.get(r["arm"])
        if prev is None or int(r["step"]) >= prev[0]:
            slot[r["arm"]] = (int(r["step"]), float(r["nrmse"]))

    rows = []
    for (noise, lr, seed, cycle, lead), d in sorted(by_arm.items()):
        if len(d) == 2:
            rows.append(dict(noise=noise, lr=lr, seed=seed, cycle=cycle, lead_step=lead,
                             lead_mtu=lead * 0.05, lead_hours=lead * 6.0,
                             lead_lyapunov=lead * 0.05 * 1.671,
                             cold_nrmse=round(d["cold"][1], 8),
                             warm_all_nrmse=round(d["warm_all"][1], 8),
                             gap=round(d["warm_all"][1] - d["cold"][1], 8)))
    if rows:
        write_table(out_path, rows, list(rows[0]))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default=C.RESULTS_DIR)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()
    out_dir = args.out_dir or os.path.join(args.results_dir, "by_model")

    ok = True
    for kind in KINDS:
        rows, files = read_tasks(args.results_dir, kind)
        if not rows:
            print(f"[{kind}] no task files found in {args.results_dir}")
            ok = False
            continue
        cols = list(rows[0].keys())

        groups = defaultdict(list)
        for r in rows:
            groups[model_key(r)].append(r)

        for key, grp in sorted(groups.items()):
            grp.sort(key=lambda r: (int(r["seed"]), int(r["cycle"]),
                                    int(r.get("step", 0) or 0),
                                    int(r.get("lead_step", 0) or 0)))
            write_table(os.path.join(out_dir, f"{model_name(key)}_{kind}.csv"), grp, cols)

        write_table(os.path.join(args.results_dir, f"all_{kind}.csv"), rows, cols)
        print(f"[{kind}] {len(files)} task files -> {len(rows)} rows, "
              f"{len(groups)} model types")

        if kind == "summary":
            missing, partial, extra = check_completeness(rows)
            if missing:
                print(f"  MISSING {len(missing)} cells, e.g. {missing[:3]}")
                ok = False
            if partial:
                print(f"  PARTIAL {len(partial)} cells (wrong cycle count), e.g. {partial[:3]}")
                ok = False
            if extra:
                print(f"  UNEXPECTED cells: {extra[:3]}")
                ok = False
            if not (missing or partial or extra):
                print(f"  complete: {len(C.CELLS)} cells x {C.N_CYCLES} cycles")
            for msg in check_invariants(rows):
                print(f"  {msg}")
                if not msg.startswith("INFO"):
                    ok = False

        if kind == "curves":
            bad = check_curves(rows)
            if bad:
                print(f"  WARN {len(bad)} runs where NRMSE does not increase with lead time, "
                      f"e.g. {bad[:2]}")
            gap_rows = gap_vs_lead(rows, os.path.join(args.results_dir, "gap_vs_lead.csv"))
            print(f"  gap-vs-lead (headline): {len(gap_rows)} rows -> gap_vs_lead.csv")

    print(f"\ntables -> {out_dir}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
