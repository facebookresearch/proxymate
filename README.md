# proxymate

**A four-level framework for proxy metric validation: diagnose, then correct.**

[![CI](https://github.com/facebookresearch/proxymate/actions/workflows/ci.yml/badge.svg)](https://github.com/facebookresearch/proxymate/actions)
[![Documentation](https://img.shields.io/badge/docs-facebookresearch.github.io%2Fproxymate-blue.svg)](https://facebookresearch.github.io/proxymate)
[![PyPI version](https://img.shields.io/pypi/v/proxymate.svg)](https://pypi.org/project/proxymate/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://pypi.org/project/proxymate/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

proxymate validates proxy metrics (surrogate endpoints, model predictions, short-term signals) against a primary outcome across four levels:

| Level | Question | Data Required |
|-------|----------|---------------|
| **Representativity** | Does the proxy population represent the target? | Coverage + weights |
| **Unit Level** | Does the proxy track the primary at the individual level? | Paired $(Y_i, Y^*_i)$ |
| **Estimate Level** | Does the proxy give the same aggregate answer? | Estimates $(\hat\theta, \hat\theta^*)$ |
| **Domain Level** | Is the relationship stable across experiments? | $K$ domain pairs |

## Installation

```bash
pip install proxymate
```

Optional extras:

```bash
pip install "proxymate[calibration]"   # calibration checks via the mcgrad package
pip install "proxymate[dev]"           # test + lint tooling
```

The documentation site is built with [Docusaurus](https://docusaurus.io/); build it locally with `cd website && npm install && npm start`.

## Quick Start

```python
import numpy as np
import pandas as pd

from proxymate import DataHandler, Validator
from proxymate.checks.unit_level.accuracy import TTestCheck
from proxymate.checks.unit_level.calibration import ECECheck
from proxymate.checks.unit_level.precision import RSquaredCheck

# proxymate ingests two tables:
#   metrics_df  -- long format: one row per (row_id, model_name) with its metric
#   entities_df -- row_id plus any segment/covariate columns (e.g. experiment_id)
# Here we synthesize a primary outcome and a correlated proxy ("surrogate").
rng = np.random.default_rng(0)
n = 2_000
primary = rng.normal(0.0, 1.0, n)
surrogate = primary + rng.normal(0.0, 0.5, n)

metrics_df = pd.DataFrame(
    {
        "row_id": np.tile(np.arange(n), 2),
        "model_name": ["primary"] * n + ["surrogate"] * n,
        "metric": np.concatenate([primary, surrogate]),
    }
)
entities_df = pd.DataFrame(
    {"row_id": np.arange(n), "experiment_id": [f"exp_{i % 10}" for i in range(n)]}
)

handler = DataHandler.from_dataframes(
    metrics_df=metrics_df,
    entities_df=entities_df,
    primary_outcome_name="primary",
    proxy_outcome_names=["surrogate"],
)

# Diagnose: run checks across the four levels
validator = Validator()
report = validator.validate(
    unit_level_checks=[TTestCheck(), RSquaredCheck(), ECECheck()],
    data_handler=handler,
)

# View per-proxy results
for proxy, result in report.per_proxy_results.items():
    print(result.to_dataframe())

# Correct: apply only the adjustments whose diagnostics failed
adjustment = validator.adjust(report)
```

## The Diagnose-Then-Correct Principle

proxymate's key insight: **applying corrections blindly can make things worse.** PPI applied to an already-unbiased proxy increases MSE by up to 8x. proxymate diagnoses first, then applies only the corrections whose assumptions are satisfied.

| Failure Mode | Diagnosed By | Corrected By |
|---|---|---|
| Systematic bias | T-test | `AdditiveBiasCorrector` |
| Miscalibration | Expected Calibration Error | `PlattRecalibrator` / `IsotonicRecalibrator` |
| Subgroup miscalibration | Per-segment calibration | `MCGradRecalibrator` |
| Aggregate bias | Bootstrap coverage | `PPIEstimator` |
| Cross-domain instability | Association R² | `RegressionCalibrator` |
| Residual heterogeneity | I², τ² | `EstimateLevelAdjustment` |

## Features

- **46 diagnostic checks** across 4 levels
- **7 adjustment strategies** mapped to specific failures
- **Precision-weighted WLS** for domain-level meta-analysis
- **Auto-adjustment bridge** — `validator.adjust(report)` applies corrections automatically
- **Challenge-based planning** — auto-selects checks based on your proxy challenge type
- **Per-domain analysis** — aggregate checks run per experiment when `domain_col` is set
- **Publication-ready plots** — scatter, forest, love plot, prevalence comparison

## Example: Domain Level Validation

```python
from proxymate import DataHandler, Validator
from proxymate.checks.domain_level.meta_analysis import (
    AggregateAssociationCheck,
    HeterogeneityCheck,
)
from proxymate.estimators.averages_estimator import WeightedAverageEstimator

# Reuses the metrics_df / entities_df built in the Quick Start above.
handler = DataHandler.from_dataframes(
    metrics_df=metrics_df,
    entities_df=entities_df,
    primary_outcome_name="primary",
    proxy_outcome_names=["surrogate"],
)

report = Validator().validate(
    meta_analysis_checks=[
        AggregateAssociationCheck(),
        HeterogeneityCheck(),
    ],
    data_handler=handler,
    estimator=WeightedAverageEstimator(),
    domain_col="experiment_id",
)

# Aggregate association and heterogeneity across the 10 experiments
for proxy, result in report.per_proxy_results.items():
    print(result.to_dataframe())
```

## Documentation

Full documentation, tutorials, and API reference are available at
[facebookresearch.github.io/proxymate](https://facebookresearch.github.io/proxymate).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to proxymate.

## Citation

```bibtex
@misc{darmon2026proxymate,
  title={proxymate: Diagnosis and Adjustment of Proxy Estimates for Reliable Inference},
  author={Darmon, Alexandra and Sinha, Deeksha and Wilkins-Reeves, Steve and Gocmen, Caner},
  note={Manuscript under review, 2026}
}
```

## License

BSD-3-Clause. See [LICENSE](LICENSE) for details.
