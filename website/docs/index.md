---
id: index
title: proxymate
slug: /
sidebar_position: 1
---

# proxymate

**Diagnosis and Adjustment of Proxy Estimates for Reliable Inference**

Proxy outcomes — short-term signals, model predictions, or surrogate endpoints — are frequently used in place of primary outcomes that are too slow, rare, or costly to measure directly. But valid inference on a proxy does not guarantee valid inference on the primary: proxy-based estimates can be systematically biased, and their confidence intervals miscalibrated.

`proxymate` is a Python framework for proxy validation and adjustment. It organizes validation into four levels, each with diagnostic checks, default thresholds, and targeted adjustment strategies.

## The Four Levels

| Level | Question | Checks |
|-------|----------|--------|
| **Representativity** | Should the validation data be weighted? | 5 |
| **Unit** | Does the proxy track the primary at the individual level? | 28 |
| **Estimate** | Does the proxy give the same aggregate answer? | 8 |
| **Domain** | Is the proxy-primary relationship stable across domains? | 6 |

## Key Principles

1. **Diagnose before correcting.** Applying adjustments (e.g., PPI) without first checking whether the proxy is biased can strictly increase estimation error (Theorem 1).
2. **Corrections feed forward.** Representativity weights flow into Unit Level checks. Corrected proxy values flow into Estimate Level aggregation. Per-domain estimates flow into Domain Level meta-analysis.
3. **Auto-detection.** The framework detects which [case](./concepts/cases.md) applies from the data you provide and runs the appropriate pipeline.

## Quick Example

```python
from proxymate.data.data_handler import DataHandler
from proxymate.orchestrator.validator import Validator
from proxymate.checks.unit_level.precision import RSquaredCheck, PearsonCorrelationCheck
from proxymate.checks.unit_level.calibration import ECECheck
from proxymate.estimators.averages_estimator import WeightedAverageEstimator

# Load data
handler = DataHandler.from_dataframes(
    metrics_df=metrics_df,
    entities_df=entities_df,
    primary_outcome_name="primary",
    proxy_outcome_names=["proxy_model"],
)

# Validate
report = Validator().validate(
    unit_level_checks=[RSquaredCheck(), PearsonCorrelationCheck(), ECECheck()],
    data_handler=handler,
)

# Inspect results
for proxy, result in report.per_proxy_results.items():
    print(result.to_dataframe())
```

## Getting Started

- [Framework Overview](./concepts/framework.md) — understand the four levels
- [Quick Start](./tutorial/quickstart.md) — end-to-end example with synthetic data
- [API Reference](./api/validator.md) — full API documentation
