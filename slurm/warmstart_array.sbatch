#!/bin/bash --login
# SLURM array job for the warm-start degradation sweep (sweep_warmstart.py).
#
# WHICH SWEEP THIS RUNS IS NOT IN THIS FILE. Name a sweep_config.py preset via
# the L96_SWEEP environment variable on every submission:
#   sbatch --export=ALL,L96_SWEEP=e1e2 ... slurm/warmstart_array.sbatch
# The preset decides the grid AND the output directory (results/<preset>/), so
# two presets can sit in the queue without colliding. It is an env var rather
# than an edit to sweep_config.py because each compute node re-imports that
# module when its task actually STARTS -- editing it while an array is queued
# would silently change what that queued job runs.
#
# One array task = one (arm, lr, noise, seed) cell = the whole 10-cycle chain.
# Cycles are inherently sequential (warm start depends on the previous cycle),
# so they cannot be split across tasks.
#
# DON'T hand-write --array ranges. Ask for them:
#   L96_SWEEP=e1e2 python sweep_warmstart.py --print_array_specs
# For the e1e2 preset that prints two tiers:
#   mem=32G time=24:00:00 n_tasks=80  --array=0-39,45-84   (cold + warm_all)
#   mem=16G time=06:00:00 n_tasks=10  --array=40-44,85-89  (frozen)
# The tiers differ because cold/warm_all train STEP_BUDGET steps every cycle
# while frozen trains only at cycle 0 and then just evaluates.
#
# THESE ARE DELIBERATELY GENEROUS FIRST-RUN VALUES, not measurements. Estimated
# real cost is well under them (~1-2 GB actual, roughly 35-75 min on a modern
# GPU), but the true numbers are unknown until this runs once. After the first
# submission, read actual usage off
#   sacct -j <jobid> --format=JobID,MaxRSS,Elapsed,State
# and tighten resource_tier() in sweep_warmstart.py. Over-requesting only costs
# queue time; under-requesting costs the whole run.
#
# Budget is in optimizer STEPS, not epochs: arms and cycles see different
# dataset sizes, so an equal epoch cap would be an up-to-86x difference in
# gradient steps and would corrupt every compute comparison.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G             # Generous first-run default; recalibrate from sacct.
#SBATCH --time=24:00:00       # Generous first-run default; recalibrate from sacct.
#SBATCH --array=0-89          # PLACEHOLDER -- override with the ranges above.
#SBATCH -J l96_warmstart
#SBATCH --output=slurm_logs/warmstart_%A_%a.out
#SBATCH --error=slurm_logs/warmstart_%A_%a.err

# mkdir -p slurm_logs before submitting -- SLURM will not create it for you.

# Fail loudly rather than silently running the default preset into the wrong directory.
: "${L96_SWEEP:?not set -- submit with --export=ALL,L96_SWEEP=e1e2}"

mamba activate weight-init-effects

cd "$SLURM_SUBMIT_DIR"   # the repo root you ran sbatch from (NOT $(dirname "$0") --
                          # SLURM copies this script to /var/spool/slurmd before running it)

# #SBATCH -J is fixed at submission time, before $SLURM_ARRAY_TASK_ID exists, so
# every task would otherwise appear in squeue as an identical row. Rename at
# runtime by decoding through sweep_warmstart.decode_task_id() -- the same source
# of truth the run itself uses -- rather than duplicating the grid here.
CELL_NAME=$(python -c "
from sweep_warmstart import decode_task_id
from sweep_config import PRESET
c = decode_task_id(${SLURM_ARRAY_TASK_ID})
print(f\"{PRESET}_{c['arm']}_lr{c['lr']:g}_noise{c['noise']:g}_seed{c['seed']}\")
")
# Address the array ELEMENT as <arrayjobid>_<taskid>; $SLURM_JOB_ID can land the
# rename on a sibling task (observed in the catdd sweep).
scontrol update JobId="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" JobName="$CELL_NAME" || true

python sweep_warmstart.py --task_id "$SLURM_ARRAY_TASK_ID"

# ============================================================================
# ONE-TIME SETUP (before the first submission)
#   mamba create -n weight-init-effects python=3.12 pytorch numpy matplotlib -c conda-forge
#   mkdir -p slurm_logs results
# No data transfer is needed: L96 trajectories are generated from the equations
# at runtime, so compute nodes need no outbound internet and no dataset copy.
#
# ROLLOUT
# 1. Calibrate ONE task of EACH tier first -- their cost profiles are unrelated,
#    so timing one tells you nothing about the other:
#      mkdir -p slurm_logs
#      sbatch --export=ALL,L96_SWEEP=e1e2 --array=0  --time=24:00:00 --mem=32G slurm/warmstart_array.sbatch
#      sbatch --export=ALL,L96_SWEEP=e1e2 --array=40 --time=06:00:00 --mem=16G slurm/warmstart_array.sbatch
#    task 0  = cold,   noise 0.05, seed 0  (trains every cycle, the expensive tier)
#    task 40 = frozen, noise 0.05, seed 0  (trains cycle 0 only)
#    Watch with:  squeue -u $USER
#    Then check ACTUAL usage:  sacct -j <jobid> --format=JobID,MaxRSS,Elapsed,State
#    The --mem/--time above are generous guesses, not measurements. The POINT of
#    this step is to replace them: record Elapsed and MaxRSS for both tiers, then
#    set resource_tier() from measured values plus margin for later sweeps.
#
# 2. Once both succeed and the CSVs look right, submit the full tiers. Send the
#    expensive tier first at normal priority and the cheap tier after with a
#    higher --nice, so the long jobs are not stuck behind short ones when the
#    concurrent-node budget is tight:
#      sbatch --export=ALL,L96_SWEEP=e1e2 --array=0-39,45-84 --time=24:00:00 --mem=32G           slurm/warmstart_array.sbatch
#      sbatch --export=ALL,L96_SWEEP=e1e2 --array=40-44,85-89 --time=06:00:00 --mem=16G --nice=100 slurm/warmstart_array.sbatch
#
# 3. Aggregate into per-model-type tables once everything finishes:
#      python aggregate.py
#
# RE-RUNNING FAILURES
# Tasks are independent and each writes only its own results/<preset>/task<NNN>_*.csv,
# so a failed task can be resubmitted alone with --array=<id>. Delete that task's
# three CSVs first -- Writer appends, so a re-run without deleting would duplicate rows.
# Per-cycle checkpoints land in results/<preset>/ckpt/ (warm_all only, ~4.2 GB total);
# the deferred shrink-and-perturb sweep branches from these instead of replaying chains.
# ============================================================================
