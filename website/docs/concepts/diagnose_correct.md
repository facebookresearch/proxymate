---
id: diagnose_correct
title: Diagnose-then-Correct
sidebar_position: 3
---

# Diagnose-then-Correct

Each level in `proxymate` follows a three-step cycle:

1. **Diagnose** — run checks, compute diagnostic statistics, compare to thresholds
2. **Adjust** — if checks fail, apply the targeted correction whose assumptions are satisfied
3. **Revalidate** — re-run checks on the corrected data to confirm the adjustment worked

## Why Not Just Adjust?

**Theorem 1** proves that applying adjustments without first diagnosing can make things worse:

$$\text{MSE}(\hat\theta^{\text{PPI}}) < \text{MSE}(\hat\theta^*) \quad \Longleftrightarrow \quad |\phi| > \frac{\sigma_\Delta}{\sqrt{n}}$$

When the proxy is already adequate ($\phi \approx 0$), PPI strictly increases MSE by adding variance from the correction term without reducing any bias. The diagnostic layer detects this and skips the adjustment.

## The Bridge Table

Each failure mode maps to a specific adjustment:

| Failure | Level | Diagnosed By | Corrected By |
|---------|-------|-------------|-------------|
| Labeling gap | Repr. | SMD, ESS | IW weights supplied as input (no adjuster class) |
| Systematic unit bias | Unit | T-test, MAPE | `AdditiveBiasCorrector` |
| Miscalibration | Unit | ECE, Brier | `PlattRecalibrator` / `IsotonicRecalibrator` / `MCGradRecalibrator` |
| Poor discrimination | Unit | AUC, Kendall τ | Proxy replacement (no post-hoc fix) |
| Aggregate bias | Estimate | Rel. abs. error | `PPIEstimator` |
| CI miscalibration | Estimate | Bootstrap coverage | `PPIEstimator` (interval) |
| Proxy already adequate | Estimate | Coverage ≈ nominal | Skip adjustment (Theorem 1) |
| Cross-domain bias | Domain | R², slope | `RegressionCalibrator` |
| High heterogeneity | Domain | I², τ² | `EstimateLevelAdjustment` |
| Model misspecification | Domain | LOO error | Warn; report bounds |

## Feed-Forward

Corrections at one level propagate to downstream levels:

```
Representativity Level
  │ IW weights
  ▼
Unit Level (weighted checks)
  │ Corrected Y* (e.g., recalibrated probabilities)
  ▼
Estimate Level (aggregation uses corrected values)
  │ Per-domain (θ̂_Y, θ̂_Y*, covariance)
  ▼
Domain Level (meta-regression on corrected estimates)
```

This means a representativity failure caught early prevents all downstream checks from being computed on the wrong distribution.
