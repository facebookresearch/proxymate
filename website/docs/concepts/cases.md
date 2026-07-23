---
id: cases
title: The Four Cases
sidebar_position: 2
---

# The Four Cases

`proxymate` auto-detects which validation scenario applies based on the data you provide. There are four cases:

## Case 1: Single Domain, Unit-Level Data, No Aggregation

**What you have:** Paired $(Y_i, Y^*_i)$ observations in one domain. Some rows may have missing proxy or primary values. You may have weights.

**What runs:** Representativity Level → Unit Level

**When to use:** You want to assess proxy measurement quality but don't have an aggregation function (e.g., you're evaluating a classifier's calibration on a labeled sample).

```python
report = validator.validate(
    unit_level_checks=[ECECheck(), ROCAUCCheck()],
    representativity_checks=[SampleRepresentativityCheck(), EffectiveSampleSizeCheck()],
    data_handler=handler,
)
```

## Case 2: Single Domain, Unit-Level Data + Estimator

**What you have:** Same as Case 1, plus an estimator that computes aggregate estimates (e.g., mean, ATE).

**What runs:** Representativity Level → Unit Level → Estimate Level

**When to use:** You want to validate that the proxy produces the same aggregate answer as the primary within a single experiment or time period.

```python
report = validator.validate(
    unit_level_checks=[RSquaredCheck(), PearsonCorrelationCheck()],
    aggregate_checks=[AggregateBiasCheck(), SignAgreementCheck()],
    bootstrap_checks=[CICoverageCheck(), UnbiasednessCheck()],
    representativity_checks=[CoverageCheck()],
    data_handler=handler,
    estimator=WeightedAverageEstimator(),
)
```

## Case 3: Multiple Domains, Aggregate-Only

**What you have:** Pre-computed paired estimates $(\hat\theta_k, \hat\theta^*_k)$ from $K$ domains. No unit-level data.

**What runs:** Domain Level only

**When to use:** You have summary statistics from multiple experiments (e.g., ATEs with standard errors) and want to assess cross-domain transportability.

```python
from proxymate.types.inputs import MetaAnalysisInput

meta_input = MetaAnalysisInput(
    domain_estimates=[(primary_est_k, proxy_est_k) for k in domains],
    domain_names=[str(k) for k in domains],
    domain_covariances=covariances,  # optional
)

report = validator.validate(
    meta_analysis_checks=[
        AggregateAssociationCheck(),
        HeterogeneityCheck(),
        PredictionAccuracyCheck(),
    ],
    meta_analysis_input=meta_input,
)
```

## Case 4: Multiple Domains, Unit-Level Data

**What you have:** Unit-level paired data across $K$ domains, identified by a domain column (e.g., `experiment_id`).

**What runs:** For each domain: Representativity → Unit → Estimate. Then across all domains: Domain Level.

**When to use:** You have raw data from multiple experiments and want the full validation pipeline — including per-domain diagnostics and cross-domain meta-analysis.

```python
report = validator.validate(
    unit_level_checks=[RSquaredCheck(), ECECheck()],
    aggregate_checks=[AggregateBiasCheck()],
    bootstrap_checks=[CICoverageCheck()],
    meta_analysis_checks=[HeterogeneityCheck(), AggregateAssociationCheck()],
    representativity_checks=[SampleRepresentativityCheck()],
    data_handler=handler,
    estimator=AverageTreatmentEffectEstimator(test_name="treatment"),
    domain_col="experiment_id",
    treatment_col="is_treatment",
)

# Per-domain results (each domain gets a full ProxyValidationResult)
for domain, proxy_results in report.per_domain_proxy_results.items():
    for proxy, result in proxy_results.items():
        print(f"{domain}/{proxy}: {result.summary()}")

# Cross-domain results
for proxy, result in report.per_proxy_results.items():
    print(result.to_dataframe())
```

### Heterogeneous Estimators (Case 4)

When different domains use different metric types, pass a dict mapping domain names to estimators:

```python
report = validator.validate(
    ...,
    estimator={
        "exp_001": WeightedAverageEstimator(),
        "exp_002": AverageTreatmentEffectEstimator(test_name="treatment"),
        "exp_003": WeightedAverageEstimator(weight_col="sample_weight"),
    },
    domain_col="experiment_id",
)
```

## How Auto-Detection Works

The validator inspects what you passed:

| Condition | Case |
|-----------|------|
| `domain_col` is None, no `estimator` | Case 1 |
| `domain_col` is None, `estimator` provided | Case 2 |
| `meta_analysis_input` provided directly | Case 3 |
| `domain_col` provided + `estimator` | Case 4 |
