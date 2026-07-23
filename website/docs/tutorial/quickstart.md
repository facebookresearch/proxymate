---
id: quickstart
title: Quick Start
sidebar_position: 1
---

# Quick Start

This tutorial walks through a complete `proxymate` validation using synthetic data. We'll generate data with known properties, run the full pipeline, and interpret the results.

## Generate Synthetic Data

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 5000

# Primary outcome and two proxies
primary = np.random.normal(100, 15, n)
good_proxy = primary * 0.9 + np.random.normal(0, 5, n)
bad_proxy = primary * 0.3 + np.random.normal(50, 20, n)

# Build the two-table format proxymate expects
metrics_rows = []
for i in range(n):
    rid = f"row_{i}"
    metrics_rows.append({"row_id": rid, "model_name": "primary", "metric": primary[i]})
    metrics_rows.append({"row_id": rid, "model_name": "good_proxy", "metric": good_proxy[i]})
    metrics_rows.append({"row_id": rid, "model_name": "bad_proxy", "metric": bad_proxy[i]})

metrics_df = pd.DataFrame(metrics_rows)
entities_df = pd.DataFrame({"row_id": [f"row_{i}" for i in range(n)]})
```

## Create a DataHandler

The `DataHandler` merges the two tables and pivots to wide format:

```python
from proxymate.data.data_handler import DataHandler

handler = DataHandler.from_dataframes(
    metrics_df=metrics_df,
    entities_df=entities_df,
    primary_outcome_name="primary",
    proxy_outcome_names=["good_proxy", "bad_proxy"],
)

print(f"Shape: {handler.df.shape}")
print(f"Primary: {handler.primary_col}")
print(f"Proxies: {handler.proxy_cols}")
```

## Run Validation

```python
from proxymate.orchestrator.validator import Validator
from proxymate.checks.unit_level.precision import RSquaredCheck, PearsonCorrelationCheck
from proxymate.checks.unit_level.accuracy import TTestCheck
from proxymate.checks.unit_level.agreement import LinCCCCheck

validator = Validator()
report = validator.validate(
    unit_level_checks=[
        RSquaredCheck(),
        PearsonCorrelationCheck(),
        TTestCheck(),
        LinCCCCheck(),
    ],
    data_handler=handler,
)
```

## Inspect Results

```python
for proxy_col, result in report.per_proxy_results.items():
    print(f"\n--- {proxy_col} ---")
    df = result.to_dataframe()
    print(df[["check_name", "projection", "threshold", "passed"]].to_string(index=False))
```

Expected output: `good_proxy` passes most checks, `bad_proxy` fails.

## Multi-Proxy Comparison

When multiple proxies are validated, `proxymate` produces a comparison:

```python
if report.comparison:
    for check_name, scores in report.comparison.ranking.items():
        best = report.comparison.best_proxy_per_check[check_name]
        print(f"{check_name}: best={best}, scores={scores}")
```

## Next Steps

- [Case 1](./case1.md) — detailed unit-level validation
- [Case 2](./case2.md) — adding aggregation and bootstrap
- [Case 4](./case4.md) — multi-domain pipeline
