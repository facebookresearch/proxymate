---
id: validator
title: Validator
sidebar_position: 1
---

# Validator

The `Validator` is the single entry point for all validation. It auto-detects the [case](../concepts/cases.md) from the inputs you provide.

## `validate()`

```python
def validate(
    self,
    unit_level_checks=None,         # list[CheckBase] — Unit Level checks
    aggregate_checks=None,           # list[CheckBase] — Estimate Level checks
    bootstrap_checks=None,           # list[CheckBase] — Bootstrap-based Estimate Level checks
    meta_analysis_checks=None,       # list[CheckBase] — Domain Level checks
    representativity_checks=None,    # list[CheckBase] — Representativity checks
    unit_level_input=None,           # UnitLevelInput — manual unit-level data
    aggregate_input=None,            # AggregateInput — manual pre-computed estimates
    bootstrap_input=None,            # BootstrappedEstimatesInput — manual bootstrap
    bootstrap_config=None,           # BootstrapConfig — override bootstrap params
    meta_analysis_input=None,        # MetaAnalysisInput — manual Case 3 input
    data_handler=None,               # DataHandler — standard two-table interface
    segment_groups=None,             # list[SegmentGroup] — for segmented analysis
    estimator=None,                  # EstimatorBase | dict[str, EstimatorBase]
    outcome_type=None,               # OutcomeType
    transformations=None,            # list[TransformationBase]
    transformation_suffix=None,      # str
    domain_col=None,                 # str — column to group by (triggers Case 4)
    treatment_col=None,              # str — binary arm indicator for ATE covariances
) -> ValidationReport
```

### Key Parameters

| Parameter | Type | Purpose |
|-----------|------|---------|
| `data_handler` | `DataHandler` | Standard data interface. If provided, `unit_level_input` is built automatically. |
| `estimator` | `EstimatorBase` or `dict[str, EstimatorBase]` | Computes aggregate estimates. A dict maps domain names to per-domain estimators. |
| `domain_col` | `str` | Column that identifies domains. Triggers Case 4 (per-domain pipeline). |
| `treatment_col` | `str` | Binary column for treatment arms. Enables per-arm ATE covariance computation. |

### Case Detection Logic

| Inputs Provided | Case |
|----------------|------|
| No `domain_col`, no `estimator` | **Case 1** — Representativity + Unit |
| No `domain_col`, `estimator` provided | **Case 2** — Representativity + Unit + Estimate |
| `meta_analysis_input` provided | **Case 3** — Domain Level only |
| `domain_col` + `estimator` | **Case 4** — Full per-domain pipeline + Domain Level |

## `adjust()`

```python
def adjust(self, report: ValidationReport) -> AdjustmentResult
```

Implements the diagnostic-to-adjustment bridge. Iterates over failed checks and applies the corresponding adjustment when data is available. Returns an `AdjustmentResult` listing what was applied and what was skipped.

## `ValidationReport`

```python
@dataclass
class ValidationReport:
    per_proxy_results: dict[str, ProxyValidationResult]
    per_domain_proxy_results: dict[str, dict[str, ProxyValidationResult]]
    comparison: ProxyComparison | None
    segment_results: list[SegmentValidationResult]
    meta_analysis_inputs: dict[str, MetaAnalysisInput]
    # ... plus stored inputs for plotting
```

- `per_proxy_results` — top-level results keyed by proxy column name
- `per_domain_proxy_results` — Case 4: `{domain_name: {proxy_col: ProxyValidationResult}}`
- `comparison` — side-by-side ranking when multiple proxies are validated

## `ProxyValidationResult`

```python
@dataclass
class ProxyValidationResult:
    unit_level_results: dict[UnitLevelProperty, list[CheckResult]]
    estimate_level_results: dict[EstimateLevelProperty, list[CheckResult]]
    domain_level_results: dict[DomainLevelProperty, list[CheckResult]]
    representativity_results: list[CheckResult]

    def summary(self) -> dict[str, float]     # pass rates per property
    def to_dataframe(self) -> pd.DataFrame    # flat table for display
```
