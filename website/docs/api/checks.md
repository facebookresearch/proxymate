---
id: checks
title: Checks Reference
sidebar_position: 2
---

# Checks Reference

All checks inherit from `CheckBase` and follow the same interface:

```python
class CheckBase(ABC, Generic[TResult, TInput]):
    validation_level: ClassVar[ValidationLevel]
    sub_property: ClassVar[...]
    supported_outcome_types: ClassVar[list[OutcomeType]]

    def run(self, input: TInput, threshold: float | None = None) -> CheckResult[TResult]
    def compute(self, input: TInput) -> TResult        # pure computation
    def project(self, result: TResult) -> float         # normalize to [0, 1]
    def get_default_threshold(self) -> float             # default: 0.8
```

## Representativity Level (5 checks)

| Check | Property | Statistic | Default Threshold |
|-------|----------|-----------|-------------------|
| `CoverageCheck` | Coverage | $\|V_k\| / \|P_k\|$ | 0.8 |
| `SampleRepresentativityCheck` | Sample Balance | Max SMD, KS statistic | 0.9 |
| `EffectiveSampleSizeCheck` | Weight Stability | Kish's ESS | 0.5 |
| `WeightExtremeCheck` | Weight Extremity | Max/min ratio, CV | 0.5 |
| `WeightingEffectivenessCheck` | Gap Reduction | SMD reduction after IW | 0.5 |

**Input type:** `CheckInput` (paired unit-level data)

**Import:** `from proxymate.checks.representativity_level.representativity import ...`

## Unit Level (27 checks)

### Accuracy (2)

| Check | Statistic | Threshold |
|-------|-----------|-----------|
| `TTestCheck` | Welch's t-test on $E[Y^* - Y]$ | 0.8 |
| `MAPECheck` | Mean absolute percentage error | 0.8 |

### Precision (3)

| Check | Statistic | Threshold |
|-------|-----------|-----------|
| `RSquaredCheck` | $R^2$ | 0.8 |
| `RMSECheck` | Root mean squared error | 0.8 |
| `PearsonCorrelationCheck` | Pearson $r$ | 0.8 |

### Agreement (3)

| Check | Statistic | Threshold |
|-------|-----------|-----------|
| `LinCCCCheck` | Lin's concordance correlation | 0.8 |
| `BlandAltmanCheck` | Limits of agreement | 0.8 |
| `DirectionalConcordanceCheck` | Sign agreement rate | 0.8 |

### Calibration (5)

| Check | Statistic | Threshold |
|-------|-----------|-----------|
| `ECECheck` | Expected calibration error | 0.8 |
| `BrierScoreCheck` | Brier score | 0.8 |
| `BrierSkillScoreCheck` | Brier skill vs. base rate | 0.8 |
| `CalibrationRatioCheck` | Predicted/observed mean | 0.8 |
| `KuiperCalibrationCheck` | Kuiper/ECCE statistic | 0.8 |

### Discrimination (11)

| Check | Statistic | Threshold |
|-------|-----------|-----------|
| `ROCAUCCheck` | ROC AUC | 0.8 |
| `PrecisionRecallAUCCheck` | PR AUC | 0.8 |
| `KendallTauCheck` | Kendall $\tau$ | 0.8 |
| `SpearmanCorrelationCheck` | Spearman $\rho$ | 0.8 |
| `ClassificationAccuracyCheck` | $(TP + TN) / N$ | 0.8 |
| `ClassificationPrecisionCheck` | Precision | 0.8 |
| `SensitivityCheck` | Sensitivity / Recall | 0.8 |
| `SpecificityCheck` | Specificity | 0.8 |
| `NPVCheck` | Negative predictive value | 0.8 |
| `MCCCheck` | Matthews correlation | 0.8 |
| `ConfusionMatrixCheck` | TP/FP/FN/TN counts (informational) | — |

### Distributional Fidelity (3)

| Check | Statistic | Threshold |
|-------|-----------|-----------|
| `KSDistCheck` | Kolmogorov-Smirnov distance | 0.8 |
| `TVDistCheck` | Total variation distance | 0.8 |
| `WassersteinDistCheck` | Wasserstein-1 / Earth mover's | 0.8 |

**Input type:** `CheckInput`

**Import:** `from proxymate.checks.unit_level.<module> import ...`

## Estimate Level (8 checks)

| Check | Property | Statistic | Threshold |
|-------|----------|-----------|-----------|
| `AggregateBiasCheck` | Aggregate Bias | Relative absolute error | 0.95 |
| `InferentialValidityCheck` | Inferential Validity | Bootstrap CI coverage | 0.05 |
| `CICoverageCheck` | CI Coverage | Empirical coverage rate | 0.5 |
| `CIOverlapCheck` | CI Overlap | Overlap coefficient | 0.5 |
| `RelativePrecisionCheck` | Relative Precision | SE ratio | 0.5 |
| `SignAgreementCheck` | Sign Agreement | Binary | 0.8 |
| `EffectSizeAgreementCheck` | Effect Size Agreement | Bidirectional CI coverage | 0.5 |
| `UnbiasednessCheck` | Unbiasedness | Normalized bias | 0.5 |

**Input type:** `AggregateInput` or `BootstrappedEstimatesInput`

**Import:** `from proxymate.checks.estimate_level.single_domain import ...`

## Domain Level (6 checks)

| Check | Property | Statistic | Threshold |
|-------|----------|-----------|-----------|
| `AggregateAssociationCheck` | Association | Precision-weighted $R^2$ | 0.8 |
| `EffectSensitivityCheck` | Sensitivity | Slope $\gamma_1$ | 0.8 |
| `DirectionalAlignmentCheck` | Alignment | Sign concordance rate | 0.8 |
| `PredictionAccuracyCheck` | Prediction | LOO cross-validated MAE | 0.8 |
| `HeterogeneityCheck` | Heterogeneity | $I^2$, $\tau^2$, Cochran's Q | 0.8 |
| `ResidualAutocorrelationCheck` | Independence | Durbin-Watson | 0.5 |

When `domain_covariances` are provided in `MetaAnalysisInput`, these checks use iterative GLS regression that accounts for measurement error in both variables.

**Input type:** `MetaAnalysisInput`

**Import:** `from proxymate.checks.domain_level.meta_analysis import ...`
