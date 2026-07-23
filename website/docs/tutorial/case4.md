---
id: case4
title: "Case 4: Multi-Domain Pipeline"
sidebar_position: 5
---

# Case 4: Multi-Domain Pipeline

**Scenario:** You have unit-level paired data across $K$ domains (e.g., experiments), identified by a domain column. This is the most complete scenario: `proxymate` runs the full within-domain pipeline (Representativity → Unit → Estimate) per domain, then the cross-domain meta-analysis on top.

## Setup

```python
import numpy as np
import pandas as pd
from proxymate.data.data_handler import DataHandler

np.random.seed(42)
K = 10
n_per_domain = 500

rows = []
for k in range(K):
    domain_bias = np.random.normal(0, 0.3)
    for i in range(n_per_domain):
        primary = np.random.normal(10, 3)
        proxy = primary * (0.9 + domain_bias * 0.1) + np.random.normal(0, 1)
        rows.append({
            "row_id": f"d{k}_r{i}",
            "experiment_id": f"exp_{k:03d}",
            "primary": primary,
            "proxy": proxy,
        })

df = pd.DataFrame(rows)

# Convert to two-table format
metrics_rows = []
for _, row in df.iterrows():
    rid = row["row_id"]
    metrics_rows.append({"row_id": rid, "model_name": "primary", "metric": row["primary"]})
    metrics_rows.append({"row_id": rid, "model_name": "proxy", "metric": row["proxy"]})

entities_df = df[["row_id", "experiment_id"]].drop_duplicates()

handler = DataHandler.from_dataframes(
    metrics_df=pd.DataFrame(metrics_rows),
    entities_df=entities_df,
    primary_outcome_name="primary",
    proxy_outcome_names=["proxy"],
)
```

## Run the Full Pipeline

```python
from proxymate.orchestrator.validator import Validator
from proxymate.estimators.averages_estimator import WeightedAverageEstimator
from proxymate.checks.unit_level.precision import RSquaredCheck
from proxymate.checks.unit_level.accuracy import TTestCheck
from proxymate.checks.estimate_level.single_domain import AggregateBiasCheck
from proxymate.checks.domain_level.meta_analysis import (
    AggregateAssociationCheck, HeterogeneityCheck,
)
from proxymate.checks.representativity_level.representativity import (
    CoverageCheck,
)

report = Validator().validate(
    unit_level_checks=[RSquaredCheck(), TTestCheck()],
    aggregate_checks=[AggregateBiasCheck()],
    meta_analysis_checks=[AggregateAssociationCheck(), HeterogeneityCheck()],
    representativity_checks=[CoverageCheck()],
    data_handler=handler,
    estimator=WeightedAverageEstimator(),
    domain_col="experiment_id",
)
```

## What Happens Under the Hood

For each domain (`exp_000`, `exp_001`, ...):

1. Data is filtered to that domain
2. Representativity checks run
3. Unit-level checks run (with any weights from step 2)
4. Aggregate checks run using the domain's estimator
5. Bootstrap checks run (if provided)

Then across all domains:

6. `MetaAnalysisInput` is built from per-domain estimates (with covariances computed from paired data)
7. Domain-level meta-analysis checks run

## Inspect Per-Domain Results

Each domain gets a full `ProxyValidationResult`:

```python
for domain_name, proxy_results in report.per_domain_proxy_results.items():
    for proxy_col, result in proxy_results.items():
        summary = result.summary()
        n_pass = sum(1 for v in summary.values() if v == 1.0)
        n_total = len(summary)
        print(f"  {domain_name}/{proxy_col}: {n_pass}/{n_total} properties pass")
```

You can drill into any domain:

```python
domain_result = report.per_domain_proxy_results["exp_003"]["proxy"]
print(domain_result.to_dataframe().to_string(index=False))
```

## Inspect Cross-Domain Results

The domain-level checks are in the top-level `per_proxy_results`:

```python
for proxy_col, result in report.per_proxy_results.items():
    df = result.to_dataframe()
    domain_checks = df[df["level"] == "domain_level"]
    print(domain_checks[["check_name", "projection", "passed"]].to_string(index=False))
```

## Heterogeneous Estimators

When different experiments use different metrics:

```python
report = Validator().validate(
    unit_level_checks=[RSquaredCheck()],
    aggregate_checks=[AggregateBiasCheck()],
    meta_analysis_checks=[HeterogeneityCheck()],
    data_handler=handler,
    estimator={
        "exp_000": WeightedAverageEstimator(),
        "exp_001": AverageTreatmentEffectEstimator(test_name="treatment"),
        # ... one per domain
    },
    domain_col="experiment_id",
)
```

Domains not in the dict are skipped with a warning.

## With Treatment Arms

For ATE estimands, pass `treatment_col` to compute per-arm covariances:

```python
report = Validator().validate(
    ...,
    domain_col="experiment_id",
    treatment_col="is_treatment",
)
```

This computes `cov(ATE_primary, ATE_proxy) = cov(Y,Y*)/n_treat + cov(Y,Y*)/n_control` per domain, enabling the covariance-aware GLS meta-regression.
