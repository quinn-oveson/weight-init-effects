import os

import numpy as np

# The preset decides the grid AND the output directory, so two presets never collide.
PRESET = os.environ.get("L96_SWEEP", "e1e2")

SEEDS = [0, 1, 2, 3, 4]          # Hard-coded for replicability, not merely recorded.
LRS = [1e-3, 3e-4, 1e-4, 3e-5]   # Swept on cold too: an undertuned ceiling understates the gap.

# Normalized units (fraction of climatological std 3.6409); never write raw sigma here.
NOISE_LEVELS = [0.05, 0.10]

SWEEP_FROZEN_LRS = False         # False pins frozen to LRS[0]; True tunes it like the other arms.
ARM_LR_RAMP = False              # Adds the warm_all_lr_ramp arm.
WARMUP_STEPS = 50                # Ramp length in steps for that arm.
WARMUP_FACTOR = 40.0             # Ramp starts at lr / WARMUP_FACTOR and ends at lr.

N_CYCLES = 10
CHUNK = 2_000
N_VAL = 4_000

# Budget in optimizer STEPS, never epochs: an equal epoch cap would differ up to 86x.
STEP_BUDGET = 6_000
BATCH_SIZE = 256
HIDDEN = 512                     # ~2.63M params vs 800k training patches.
N_LAYERS = 3
KERNEL = 5
HISTORY = 4
F = 8.0

N_EVAL_POINTS = 60               # Log-spaced to resolve the early optimum, not step past it.

CURVE_EVERY = 10                 # Full forecast curves every Nth eval point, plus cycle end.
ROLLOUT_STEPS = 120
ROLLOUT_INITS = 64
ROLLOUT_INITS_EVAL = 16
THRESHOLDS = [0.2, 0.3, 0.5, 0.8]

# Lets the deferred S&P sweep branch at cycle boundaries; unrecoverable if skipped.
CHECKPOINT = True
CHECKPOINT_ARMS = ("warm_all",)  # The arm S&P would branch from; set None for all arms.

PRESETS = {
    "e1e2": {},
    # Run 2: LR grid extended past the boundary run 1 sat on, longer chains, tuned frozen.
    "e1e2b": dict(LRS=[1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5], N_CYCLES=20,
                  SWEEP_FROZEN_LRS=True, ARM_LR_RAMP=True),
    "smoke": dict(SEEDS=[0], NOISE_LEVELS=[0.05], LRS=[1e-3, 1e-4], N_CYCLES=3, CHUNK=500,
                  N_VAL=800, STEP_BUDGET=40, HIDDEN=64, CURVE_EVERY=2, ROLLOUT_STEPS=40,
                  ROLLOUT_INITS=8, ROLLOUT_INITS_EVAL=4, N_EVAL_POINTS=12,
                  ARM_LR_RAMP=True, WARMUP_STEPS=10),
}

if PRESET not in PRESETS:
    raise SystemExit(f"unknown L96_SWEEP preset {PRESET!r}; known: {sorted(PRESETS)}")
for _k, _v in PRESETS.get(PRESET, {}).items():
    globals()[_k] = _v

# Derived AFTER the overrides, so a preset that changes LRS or STEP_BUDGET stays consistent.
ARMS = [
    # (name, warm_start, data_all, frozen, warmup, learning rates)
    ("cold", 0, 1, 0, 0, LRS),
    ("warm_all", 1, 1, 0, 0, LRS),
    ("frozen", 1, 1, 1, 0, LRS if SWEEP_FROZEN_LRS else [LRS[0]]),
]
if ARM_LR_RAMP:
    ARMS.insert(2, ("warm_all_lr_ramp", 1, 1, 0, 1, LRS))
EVAL_STEPS = sorted(set(int(round(v)) for v in
                        np.logspace(0, np.log10(STEP_BUDGET), N_EVAL_POINTS)) | {STEP_BUDGET})

RESULTS_DIR = os.path.join("results", PRESET)
CKPT_DIR = os.path.join(RESULTS_DIR, "ckpt")


def cells():
    # One cell = one (arm, lr, noise, seed) = one full sequential cycle chain = one SLURM task.
    out = []
    for noise in NOISE_LEVELS:
        for arm, warm, data_all, frozen, warmup, lrs in ARMS:
            for lr in lrs:
                for seed in SEEDS:
                    out.append(dict(arm=arm, warm_start=warm, data_all=data_all,
                                    frozen=frozen, warmup=warmup, lr=lr, noise=noise,
                                    seed=seed))
    return out


CELLS = cells()
TOTAL_TASKS = len(CELLS)
