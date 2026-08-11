# Forecaster build: measured outcomes

All runs: L96 at F=8, circular CNN (3 layers, hidden 64, kernel 5, SiLU), 20k training pairs,
30 epochs, model timestep 0.05 MTU (6 h). Valid prediction time = steps until normalized error
crosses 0.3, expressed in Lyapunov times using λ₁(F=8) = 1.671.

> **Caveat: every number below is a single seed (seed 0).** The plan calls for ≥5 seeds with
> spread reported. Differences of 50%+ are probably real; anything smaller should not be trusted
> until the seed sweep is run.

## Step 1 — CNN, no noise

| model | valid time (Lyap) | rollout std vs truth |
|---|---|---|
| CircularCNN | 2.76 | 1.003 / 1.018 |
| persistence | 0.08 | — |

Val MSE 1e-5. The CNN essentially solves the noise-free problem, confirming that its inductive
bias matches L96 almost exactly and that the noise-free task leaves no headroom for
initialization effects to show.

## Step 2 — observation noise

Rollouts initialized from a noisy analysis and scored against clean truth.

| noise | valid time (Lyap) | MSE vs noisy target | MSE vs clean target |
|---|---|---|---|
| 0.00 | 2.76 | 0.00001 | 0.00001 |
| 0.01 | 2.76 | 0.00021 | 0.00002 |
| 0.02 | 2.59 | 0.00080 | 0.00007 |
| 0.05 | 2.01 | 0.00496 | 0.00042 |
| 0.10 | 1.25 | 0.01956 | 0.00192 |

Noise lowers the ceiling by 55% at 10% — the knob works. MSE-vs-noisy stays ~10× MSE-vs-clean
at every level, confirming the model learns signal rather than fitting noise.

Verified separately: injected noise std is exact (0.0500 requested → 0.04999 measured), and
tendency std scales linearly with stride to 0.991 of linear at stride 5, validating the dt=0.05
choice from the plan.

## Step 3 — temporal window (noise = 0.10)

| history | valid time (Lyap) | one-step corr. w/ clean tendency | pred/target variance ratio |
|---|---|---|---|
| 1 | 1.00 | 0.985 | 1.003 |
| 2 | 1.34 | 0.956 | 1.037 |
| 4 | **1.59** | 0.943 | 1.051 |
| 8 | 1.50 | 0.940 | 1.054 |

**One-step metrics rank these in the opposite order from rollout skill.** h=1 has the best
correlation and the best variance calibration and is 60% worse at the actual task. Selecting on
one-step MSE would have shipped the wrong model. This is the empirical justification for using
valid prediction time as the primary metric.

*Mechanism is unresolved.* An initial explanation — that low-history models shrink their
predictions — was tested and **refuted** (h=1 has variance ratio 1.003, not < 1). Current
conjecture: a history window lets the model separate iid observation noise from
dynamically-correlated rollout error. **Not demonstrated. Treat as a hypothesis.**

## Step 4 — stochastic output head (noise = 0.10)

| model | valid time (Lyap) | rollout std |
|---|---|---|
| h=1 deterministic | 1.00 | 0.954 |
| h=1 heteroscedastic, mean | 1.00 | 0.959 |
| h=1 heteroscedastic, sampled | 0.42 | 0.992 |
| h=4 deterministic | 1.59 | 0.977 |
| h=4 heteroscedastic, mean | 1.59 | 0.989 |
| h=4 heteroscedastic, sampled | 1.00 | 0.997 |

The head's **mean matches the deterministic model exactly** (to the metric's one-step
resolution), so the stochastic head does not produce a worse model. Sampling is what costs valid
time — single-trajectory RMSE structurally favors a deterministic forecast over any sampled one.
Sampling also nearly closes the variance deficit (0.977 → 0.997).

**This comparison cannot decide the question.** Judging a probabilistic forecast on
single-trajectory RMSE is the wrong test; it needs ensemble metrics (CRPS, spread-skill ratio),
which are not implemented.

## Configuration carried forward

**Deterministic circular CNN, history = 4, noise = 0.05–0.10.**

Rationale: h=4 is the rollout optimum; deterministic keeps the loss as plain MSE, which matters
because the re-initialization study is about optimization dynamics and an NLL objective would be
a confound. The heteroscedastic head remains available via `--hetero` and should be revisited
only with ensemble metrics in place.

## Plan verification items

| # | Item | Result |
|---|---|---|
| 1 | Tendency scales linearly with dt | 0.991 of linear at stride 5 — dt=0.05 confirmed |
| 2 | Beats persistence | 2.76 vs 0.08 Lyapunov times, noise-free |
| 3 | Rollout stable over 10⁴ steps | finite, max\|x\|=3.54, std 0.995 vs truth 0.999 (≈7 simulated years) |
| 4 | Attractor statistics match | marginal total-variation distance 0.0086; mean 0.012 vs −0.002 |
| 5 | Metrics normalized per regime | implemented in `metrics.lambda1` from the measured λ₁(F) table |
| 6 | Noise never leaks into training | asserted: training inputs and targets are the noisy tensors |

## Open items before the re-initialization experiments

1. **Seed sweep** — everything here is n=1.
2. **CRPS / spread-skill** if the stochastic head is to be judged fairly.
3. **Test the history conjecture** in Step 3, or drop the explanation and report the effect alone.
4. **Width sweep** — `TRANSFER.md` flags dimensionality as the main threat to transferring
   conclusions to operational scale; whether the re-init ranking is width-stable is the check.
