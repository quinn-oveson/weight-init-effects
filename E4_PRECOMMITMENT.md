# E4 pre-commitment

**Written 2026-08-12, before any E4 data exists.** Its only value comes from that ordering. If
this document is ever revised, the revision and its reason must be appended below with a date —
never edited in place.

E4 tests whether Lorenz 96 reproduces ECMWF's documented 50r1 adaptation failure: fine-tuning on
new-regime data alone *"either overfitted or failed to adequately learn the dynamics of the new
analysis, depending on the learning rate."* The hazard is obvious — if that pattern does not
appear, it is tempting to adjust the shift magnitude or the data volume until it does, then
report a reproduction. This document exists to make that impossible.

---

## 1. The separation rule

> **Any quantity used to calibrate the E4 pilot must be measurable without comparing warm-start
> to cold-start.**

Calibrating on single-arm quantities makes the experiment *well-posed*. Calibrating on the
warm-vs-cold contrast *manufactures the result*. That is the entire distinction.

The pilot may inspect:

- whether a distribution shift exists at all — `frozen` on the new regime vs. the oracle
- whether the shift is regional — the in-patch / halo / far split
- whether the oracle can fit the shifted system at all

None of these require knowing how warm-start compares to cold-start.

The pilot may **not** inspect any warm-vs-cold difference, at any metric, before the
configuration is frozen.

## 2. Two ceilings

| | Definition | Question it answers |
|---|---|---|
| `cold` | retrain from scratch on all data seen so far | how much warm-starting costs vs. the expensive option a lab could actually run — an **operational** ceiling |
| `oracle` | trained from scratch on ample data from the **shifted** system | how much of the physically achievable in-patch skill anyone captured — a **physical** ceiling, never deployed |

The oracle is required because shifting F to 12 on the patch raises λ₁ from 1.69 to 2.89 (+71%),
so in-patch error is worse than out-of-patch *regardless of what any model learns*. Without it,
"failed to adapt" cannot be distinguished from "this region is intrinsically harder." All
in-patch claims are therefore expressed **relative to the oracle**, never relative to
out-of-patch error.

## 3. Permitted adjustments

Bounded in advance. Each may be set only from the single-arm criterion listed.

| Parameter | Allowed values | Criterion |
|---|---|---|
| `F_patch` | {10, 12, 14}; must stay ≥ 5 so the patch remains chaotic | `frozen`-on-new-regime measurably worse than oracle, **and** oracle reaches skill comparable to the F=8 baseline |
| positional-embedding width | {4, 8, 16} channels | oracle can fit the site-dependent dynamics; if none can, stop — that is an architecture problem, not a finding |
| adaptation step budget | value carried from E1/E2, ±2× | oracle converges within it |

Anything not in this table is not adjustable after the pilot.

## 4. Frozen before the pilot runs

- patch width **8** of 40 sites (calibrated on the locality table — a pure data property,
  measured before any model was trained)
- arm definitions
- 5 seeds, hard-coded {0,1,2,3,4}
- all metric definitions and thresholds
- the three-way **in-patch / halo / far** split (binary would put the halo, which carries ~38%
  of the in-patch effect, into the control group)
- lead times expressed in **Lyapunov times**, not MTU, since λ₁ differs across regions
- observation noise {0.05, 0.10} in normalized units

## 5. Outcome commitments

Recorded now so that no outcome can be reframed after the fact.

| Outcome | Reported as |
|---|---|
| `warm_new` fails in **both** directions across the LR sweep (overfits at high LR, under-adapts at low), and `warm_all` beats it | ECMWF pattern **reproduced** |
| Fails in **one** direction only | **Partial reproduction** — state which direction, and state that it is partial |
| No separation between arms | The analogue **does not reproduce**. Reported as a finding about the limits of L96 as an operational proxy — not as a failed experiment |
| Warm arms **beat** cold | Reported as-is, contradicting the hypothesis |

The third row is the one this document exists to protect. Non-reproduction is a publishable
result about how far the analogy extends, and it must not be converted into a tuning exercise.

## 6. Deliberately unresolved

These are open, and must be settled **before the pilot runs**, not after:

- **Number of permitted pilot iterations** before the configuration freezes (one, or two with a
  stated well-posedness reason).
- **The pre-registered primary endpoint** — the single number the result stands on. Leading
  candidate: in-patch valid prediction time in Lyapunov units as a fraction of the oracle's.
- **Whether the pilot uses held-out seeds** (e.g. pilot on {90, 91}, report on {0–4}) so that
  tuning to pilot noise cannot leak into reported numbers.

Listed here as open rather than silently decided, so that filling them in later is a visible act.

## 7. Prerequisites not yet built

E4 cannot run until these exist, and none of them depend on E4 results:

- `lorenz96/system.py` — `integrate` currently reads a non-scalar `F` as a per-*step* schedule
  and rejects a per-*site* vector.
- `lorenz96/models.py` — per-site learned positional embeddings. A weight-shared circular CNN is
  translation-equivariant and **structurally cannot** represent site-dependent dynamics. Pangu
  uses a learned Earth-Specific Positional Bias and GraphCast uses static per-node features, so
  the current model is more translation-invariant than any deployed weather model.
- **The model must not be given `F_i`.** Position only; let it infer what that position implies.
  Real systems get coordinates and static fields, not a list of which physics changed where.
  Supplying F would trivialize the adaptation problem.

---

## Revision log

*(append-only; date and reason required)*

- 2026-08-12 — created, before any E4 data existed.
