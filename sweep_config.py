import os

import numpy as np

# The preset decides the grid AND the output directory, so two presets never collide.
PRESET = os.environ.get("L96_SWEEP", "e1e2")

SEEDS = [0, 1, 2, 3, 4]          # Hard-coded for replicability, not merely recorded.
LRS = [1e-3, 3e-4, 1e-4, 3e-5]   # Swept on cold too: an undertuned ceiling understates the gap.

# Normalized units (fraction of climatological std 3.6409): raw sigma 0.182 and 0.364,
# i.e. 19.6% and 39.2% of the one-step increment. Never write raw sigma here.
NOISE_LEVELS = [0.05, 0.10]

ARMS = [
    # (name, warm_start, data_all, frozen, learning rates)
    ("cold", 0, 1, 0, LRS),
    ("warm_all", 1, 1, 0, LRS),
    ("frozen", 1, 1, 1, [LRS[0]]),   # Trains at cycle 0 only; the do-nothing baseline.
]

N_CYCLES = 10
CHUNK = 2_000
N_VAL = 4_000

# Budget in optimizer STEPS, never epochs: arms and cycles see different dataset sizes,
# so an equal epoch cap would be an up-to-86x difference in gradient steps.
STEP_BUDGET = 6_000
BATCH_SIZE = 256
HIDDEN = 512                     # ~2.63M params vs 800k training patches.
N_LAYERS = 3
KERNEL = 5
HISTORY = 4
F = 8.0

# E0 put the validation optimum near step 91 and memorization near step 6000, so the eval
# grid is log-spaced to resolve the early optimum rather than stepping past it.
EVAL_STEPS = sorted(set(int(round(v)) for v in
                        np.logspace(0, np.log10(STEP_BUDGET), 60)) | {STEP_BUDGET})

CURVE_EVERY = 10                 # Full forecast curves every Nth eval point, plus cycle end.
ROLLOUT_STEPS = 120
ROLLOUT_INITS = 64
ROLLOUT_INITS_EVAL = 16
THRESHOLDS = [0.2, 0.3, 0.5, 0.8]

# Per-cycle checkpoints let the deferred S&P sweep branch at cycle boundaries instead of
# replaying whole chains; unrecoverable if skipped.
CHECKPOINT = True
CHECKPOINT_ARMS = ("warm_all",)  # The arm S&P would branch from; set None for all arms.

PRESETS = {
    "e1e2": {},
    "smoke": dict(SEEDS=[0], NOISE_LEVELS=[0.05], LRS=[1e-3, 1e-4], N_CYCLES=3, CHUNK=500,
                  N_VAL=800, STEP_BUDGET=40, HIDDEN=64, CURVE_EVERY=2, ROLLOUT_STEPS=40,
                  ROLLOUT_INITS=8, ROLLOUT_INITS_EVAL=4),
}

for _k, _v in PRESETS.get(PRESET, {}).items():
    globals()[_k] = _v

if PRESET != "e1e2":
    ARMS = [("cold", 0, 1, 0, LRS), ("warm_all", 1, 1, 0, LRS), ("frozen", 1, 1, 1, [LRS[0]])]
    EVAL_STEPS = sorted(set(int(round(v)) for v in
                            np.logspace(0, np.log10(STEP_BUDGET), 12)) | {STEP_BUDGET})

RESULTS_DIR = os.path.join("results", PRESET)
CKPT_DIR = os.path.join(RESULTS_DIR, "ckpt")


def cells():
    # One cell = one (arm, lr, noise, seed) = one full sequential cycle chain = one SLURM task.
    out = []
    for noise in NOISE_LEVELS:
        for arm, warm, data_all, frozen, lrs in ARMS:
            for lr in lrs:
                for seed in SEEDS:
                    out.append(dict(arm=arm, warm_start=warm, data_all=data_all,
                                    frozen=frozen, lr=lr, noise=noise, seed=seed))
    return out


CELLS = cells()
TOTAL_TASKS = len(CELLS)
