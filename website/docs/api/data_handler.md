---
id: data_handler
title: Data Handler
sidebar_position: 4
---

# Data Handler

The `DataHandler` provides a standard interface for loading and preparing data for validation.

**Import:** `from proxymate.data.data_handler import DataHandler`

## Construction

### `from_dataframes()`

```python
handler = DataHandler.from_dataframes(
    metrics_df=metrics_df,          # Long-format: row_id, model_name, metric, [weight]
    entities_df=entities_df,        # Entity metadata: row_id, [covariates...]
    primary_outcome_name="primary",
    proxy_outcome_names=["proxy_a", "proxy_b"],
    weight_col="weight",            # Optional: column with sampling/IW weights
)
```

The metrics table is in long format with one row per (entity, model). The handler pivots to wide format internally:

| row_id | model_name | metric |
|--------|-----------|--------|
| r_0 | primary | 10.5 |
| r_0 | proxy_a | 9.8 |
| r_0 | proxy_b | 11.2 |

## Key Properties

| Property | Type | Description |
|----------|------|-------------|
| `handler.df` | `pd.DataFrame` | Merged wide-format DataFrame |
| `handler.primary_col` | `str` | Primary outcome column name |
| `handler.proxy_cols` | `list[str]` | Proxy outcome column names |
| `handler.weight_col` | `str \| None` | Weight column name |

## Building MetaAnalysisInput

```python
meta_inputs = handler.build_meta_analysis_input(
    domain_col="experiment_id",
    estimator=estimator,                  # or dict[str, EstimatorBase]
    min_domain_rows=2,
    treatment_col="is_treatment",         # optional: for ATE covariances
)
# Returns dict[str, MetaAnalysisInput] — one per proxy column
```

When `treatment_col` is provided, per-arm covariances are computed:

$$\text{cov}(\text{ATE}_Y, \text{ATE}_{Y^*}) = \frac{\text{cov}(Y, Y^*)}{n_{\text{treat}}} + \frac{\text{cov}(Y, Y^*)}{n_{\text{control}}}$$

Without `treatment_col`, assumes a mean estimand: $\text{cov}(\bar{Y}, \bar{Y^*}) = \text{cov}(Y, Y^*) / n$.

## Bootstrap

```python
handler.run_bootstrap(
    estimator=estimator,
    n_iterations=100,
    seed=42,
)

# Retrieve bootstrap results
boot_input = handler.get_bootstrapped_estimates_input("proxy_a")
```

## Segments

```python
from proxymate.lib.segment import Segment

segment = Segment("US", "country = 'US'", ["country"])
filtered_handler = handler.filter_segment(segment)
```

## Transformations

Apply unit-level corrections (e.g., recalibration) and validate the corrected proxy alongside the original:

```python
from proxymate.adjustment.unit_level import PlattRecalibrator

handler_with_transformed = handler.apply_transformations(
    [PlattRecalibrator().fit(check_input)],
    suffix="_calibrated",
)
# Now has proxy columns: ["proxy_a", "proxy_a_calibrated"]
```
