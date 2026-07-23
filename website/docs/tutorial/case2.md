---
id: case2
title: "Case 2: Adding Aggregation"
sidebar_position: 3
---

# Case 2: Adding Aggregation

**Scenario:** Same as Case 1, but now you have an estimator that computes aggregate estimates (e.g., weighted mean, ATE). This enables Estimate Level checks: does the proxy produce the same aggregate answer as the primary?

## Setup

```python
import numpy as np
import pandas as pd
from proxymate.data.data_handler import DataHandler

np.random.seed(42)
n = 2000

primary = np.random.normal(10, 3, n)
proxy = primary + np.random.normal(0.5, 1, n)  # biased proxy (mean shift of +0.5)

metrics_rows = []
for i in range(n):
    metrics_rows.append({"row_id": f"r_{i}", "model_name": "primary", "metric": primary[i]})
    metrics_rows.append({"row_id": f"r_{i}", "model_name": "proxy", "metric": proxy[i]})

handler = DataHandler.from_dataframes(
    metrics_df=pd.DataFrame(metrics_rows),
    entities_df=pd.DataFrame({"row_id": [f"r_{i}" for i in range(n)]}),
    primary_outcome_name="primary",
    proxy_outcome_names=["proxy"],
)
```

## Run All Three Levels

```python
from proxymate.orchestrator.validator import Validator
from proxymate.estimators.averages_estimator import WeightedAverageEstimator
from proxymate.checks.unit_level.accuracy import TTestCheck
from proxymate.checks.unit_level.precision import RSquaredCheck
from proxymate.checks.estimate_level.single_domain import (
    AggregateBiasCheck, CICoverageCheck, SignAgreementCheck,
)

report = Validator().validate(
    unit_level_checks=[TTestCheck(), RSquaredCheck()],
    aggregate_checks=[AggregateBiasCheck(), SignAgreementCheck()],
    bootstrap_checks=[CICoverageCheck()],
    data_handler=handler,
    estimator=WeightedAverageEstimator(),
)
```

## The Diagnose-then-Correct Cycle

Check the results:

```python
result = report.per_proxy_results["proxy"]
print(result.to_dataframe().to_string(index=False))
```

The `TTestCheck` will likely fail (the proxy has a +0.5 bias). The `AggregateBiasCheck` will also flag this at the estimate level.

### Apply the Adjustment

```python
adj_result = Validator().adjust(report)

for check_name, adj_name, adj_obj in adj_result.applied:
    print(f"  {check_name} → {adj_name}")

for check_name, reason in adj_result.skipped:
    print(f"  {check_name} → skipped: {reason}")
```

The adjustment bridge maps:

- T-test failure → `AdditiveBiasCorrector` (subtracts mean difference)
- Aggregate bias failure → `PPIEstimator` (but gated by Theorem 1 — skipped if Inferential Validity passes)

### Theorem 1 in Action

If the proxy's CI already covers the primary (Inferential Validity passes), PPI is skipped because it would add variance without reducing bias. This is the gating rule from Theorem 1:

$$\text{MSE}(\hat\theta^{\text{PPI}}) < \text{MSE}(\hat\theta^*) \quad \Longleftrightarrow \quad |\phi| > \frac{\sigma_\Delta}{\sqrt{n}}$$
