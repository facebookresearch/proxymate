---
id: case1
title: "Case 1: Unit-Level Validation"
sidebar_position: 2
---

# Case 1: Unit-Level Validation

**Scenario:** You have paired $(Y, Y^*)$ data in a single domain. No aggregation function. You want to assess whether the proxy tracks the primary at the individual level, and whether your validation data is representative.

## Setup

```python
import numpy as np
import pandas as pd
from proxymate.data.data_handler import DataHandler

np.random.seed(42)
n = 3000

# Simulate a biased labeled sample: high-value units are over-represented
primary = np.random.normal(50, 10, n)
proxy = primary * 0.95 + np.random.normal(0, 3, n)

# Sampling weight: inverse of selection probability
# (simulates a review queue that over-samples high values)
selection_prob = 1 / (1 + np.exp(-(primary - 50) / 5))
weights = 1.0 / selection_prob

metrics_rows = []
for i in range(n):
    metrics_rows.append({"row_id": f"r_{i}", "model_name": "primary", "metric": primary[i], "weight": weights[i]})
    metrics_rows.append({"row_id": f"r_{i}", "model_name": "proxy", "metric": proxy[i], "weight": weights[i]})

handler = DataHandler.from_dataframes(
    metrics_df=pd.DataFrame(metrics_rows),
    entities_df=pd.DataFrame({"row_id": [f"r_{i}" for i in range(n)]}),
    primary_outcome_name="primary",
    proxy_outcome_names=["proxy"],
    weight_col="weight",
)
```

## Run Representativity + Unit Level

```python
from proxymate.orchestrator.validator import Validator
from proxymate.checks.representativity_level.representativity import (
    CoverageCheck, SampleRepresentativityCheck, EffectiveSampleSizeCheck,
    WeightExtremeCheck, WeightingEffectivenessCheck,
)
from proxymate.checks.unit_level.precision import RSquaredCheck, RMSECheck
from proxymate.checks.unit_level.accuracy import TTestCheck
from proxymate.checks.unit_level.agreement import LinCCCCheck

report = Validator().validate(
    representativity_checks=[
        CoverageCheck(),
        SampleRepresentativityCheck(),
        EffectiveSampleSizeCheck(),
        WeightExtremeCheck(),
        WeightingEffectivenessCheck(),
    ],
    unit_level_checks=[
        RSquaredCheck(),
        RMSECheck(),
        TTestCheck(),
        LinCCCCheck(),
    ],
    data_handler=handler,
)
```

## Interpret Results

```python
for proxy_col, result in report.per_proxy_results.items():
    print(result.to_dataframe()[["level", "check_name", "projection", "passed"]].to_string(index=False))
```

Check the representativity results first:

- If **ESS** is low → weights are too variable, downstream results may be unreliable
- If **Weighting Effectiveness** fails → IW didn't reduce the covariate gap sufficiently
- If representativity passes → unit-level checks are computed on a sound distribution

## What Flows Forward

In Case 1, the representativity diagnostics inform your confidence in the unit-level results. If representativity checks flag severe issues, the unit-level results should be interpreted with caution — `proxymate` does not automatically discard them, but the representativity flags tell you the data foundation is shaky.
