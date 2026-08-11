# Does Lorenz 96 transfer to weather and quant?

Written against measured values from `explore.py`, not from intuition.

## Measured properties

| F | std of X | λ₁ (1/MTU) | Lyapunov time (MTU) |
|---|---|---|---|
| 3 | 1.46 | −0.003 | — (non-chaotic) |
| 4 | 1.90 | 0.002 | — (edge of chaos) |
| 5 | 2.37 | 0.523 | 1.91 |
| 6 | 2.83 | 0.964 | 1.04 |
| 8 | 3.64 | 1.671 | 0.60 |
| 10 | 4.38 | 2.278 | 0.44 |
| 12 | 5.06 | 2.873 | 0.35 |
| 16 | 6.31 | 3.849 | 0.26 |

Chaos onsets between F=4 and F=5. λ₁ at F=8 matches the literature (≈1.7), and 40 independent trajectory pairs diverge as e^{λ₁t} across eight orders of magnitude, so the integrator and tangent linear model are both correct.

## Weather: a good proxy

**The timescale is right, not just qualitatively.** Lorenz set the time unit so the dissipative −X term has a 5-day e-folding, i.e. 1 MTU ≈ 5 days. At F=8 the error doubling time is ln2/1.671 = 0.415 MTU ≈ **2.1 days**, against ~1.5–2 days for the real atmosphere. The deterministic predictability limit (~2 weeks) lands at ≈4.7 Lyapunov times. This is why L96 has survived as a weather surrogate for thirty years.

**Structurally shared:** forced-dissipative, spatially extended on a periodic ring (a latitude circle), translation-equivariant, positive λ₁, and — the property this project depends on — autoregressive rollout where errors compound at a rate set by the physics rather than the model.

**F-shift is a credible analogue of changing forcing.** It moves mean, variance, *and* λ₁ together, which is what changing climate forcing does. It is not a synthetic relabeling.

## Weather: what's missing, ranked by how much it threatens this study

1. **Dimensionality — the serious one.** 40 states vs ~10⁷. Plasticity is known to interact with scale: foundation models retain plasticity under incremental fine-tuning where small models degrade (Liu et al., IJCNN 2025). A re-initialization strategy that wins at K=40 may be irrelevant at AIFS scale. **Mitigation: sweep model width and report whether the ranking of re-init schemes is width-stable.** If it isn't, that is itself the headline result and must be stated rather than buried.
2. **No seasonality or recurrence.** Real forcing is cyclic; `step` and `ramp` are monotone. Cyclic regimes probe forgetting-and-relearning, a failure mode monotone schedules cannot reach. **Add a `cyclic` schedule** (F oscillating 8↔16). This is a genuine gap in the current `f_schedule`.
3. **No observational noise or data assimilation.** AIFS trains on reanalysis, itself a model–data hybrid; operational "new data" is analysis, not truth. **Add an optional observation-noise knob** — cheap, and it doubles as the bridge toward the low-SNR case below.
4. **No fat tails / no-analog extremes.** AI weather models underperform NWP precisely on record-breaking events (arXiv 2603.23043), which is among the strongest motivations for the project — and L96 cannot reproduce it. Acknowledge as a scope limit; do not claim to address extremes.
5. **Single-scale.** Real weather spans convection → synoptic → planetary. The two-scale X/Y variant supplies this. **Verdict: not needed.** The question here is optimization dynamics under distribution shift, not subgrid parameterization. Two-scale L96 would double the complexity and introduce a confound (is degradation from re-init, or from the unresolved scale?) without addressing the re-init question.

## Quant: a weak proxy — keep it in the motivation, not the testbed

Four differences, of which the first and third are categorical rather than differences of degree:

1. **L96 is deterministic.** Given X(t) and F, X(t+Δt) is exact — one-step prediction is essentially noiseless. Financial returns are irreducibly stochastic with SNR on the order of 0.01–0.05. Nearly every practical difficulty in quant modeling comes from that gap, and L96 does not contain it.
2. **L96 is stationary given F.** A market's generating process genuinely changes — new participants, new instruments, regulation — and is not parameterized by one scalar. A single-knob shift is the easy case.
3. **No feedback.** Predicting L96 does not change L96. In markets, prediction changes the predicted (alpha decay), and this is arguably the *main* reason quant models require retraining. L96 cannot represent it at all. A finding about update strategy here has no bearing on that mechanism.
4. **No volatility clustering or fat tails.**

Adding observation noise would narrow (1) but not (2)–(4). **Recommendation: cite quant as evidence that non-stationary retraining is a broad industrial problem, and make no empirical claim about it.** Better to state this yourself than have a reviewer state it.

## Consequences for the experiment design

- **Valid prediction time must be normalized per-regime.** λ₁ more than doubles from F=8 to F=16, so a fixed-step forecast horizon is not comparable across regimes: a model can look worse purely because the target got more chaotic. Use the λ₁(F) table above to express horizons in Lyapunov times, and include a per-regime from-scratch oracle as the achievable ceiling.
- **Frozen normalization is confirmed necessary, with a number.** std ranges 1.46 → 6.31 across F. Normalizing per-regime would erase a 4.3× variance change and manufacture the exact degradation signal under study. Already enforced in `lorenz96/data.py`.
- **Keep F ∈ [5, 16].** Below F≈5 the system is non-chaotic, so a shift crossing that boundary changes the problem class, not just the regime.
- **Add `cyclic` to `f_schedule`** and an optional noise knob before the training harness is built.

## Verdict

L96 is a well-calibrated weather surrogate — the error-doubling timescale matches the atmosphere to within a factor of ~1.4, and the properties this project needs (compounding autoregressive error, controllable forcing shift) are all present. It is a poor quant surrogate for reasons that are structural rather than fixable. The one limitation that could invalidate transfer to operational systems is dimensionality, and it is testable via a width sweep rather than assumed away.
