---
id: planner
title: Validation Planner
sidebar_position: 5
---

# Validation Planner

The planner maps proxy challenges to recommended checks, helping practitioners decide what to run without deep framework knowledge.

**Import:** `from proxymate.planner.validation_planner import ValidationPlanner, build_default_planner`

## Proxy Challenges

| Challenge | When to Use | Key Checks |
|-----------|------------|------------|
| `LONG_TERM_PRIMARY` | Primary takes weeks/months to mature | Association, heterogeneity, prediction accuracy |
| `RESTRICTED_LABELS` | Ground truth requires costly human review | Representativity, calibration, discrimination |
| `LOW_DETECTABILITY` | Outcome is rare or high-variance | Precision, agreement, aggregate bias |
| `LOW_SENSITIVITY` | Metric is too blunt for small effects | Discrimination, sensitivity, directional alignment |

Challenges can be combined:

```python
from proxymate.types.enums import ProxyChallenge

challenge = ProxyChallenge.LONG_TERM_PRIMARY | ProxyChallenge.LOW_SENSITIVITY
```

## Usage

### Quick Start with Default Planner

```python
planner = build_default_planner()

# See what's recommended
names = planner.recommend_names(ProxyChallenge.RESTRICTED_LABELS)
print(names)
# ['coverage', 'sample_representativity', 'effective_sample_size', ...]

# Get instantiated checks grouped for validate()
check_groups = planner.auto_checks(ProxyChallenge.RESTRICTED_LABELS)
# {'representativity_checks': [...], 'unit_level_checks': [...], ...}

# Pass directly to validator
report = Validator().validate(**check_groups, data_handler=handler)
```

### Human-Readable Explanation

```python
print(planner.explain(ProxyChallenge.LONG_TERM_PRIMARY))
```

Output:
```
Recommended checks for LONG_TERM_PRIMARY:
  1. aggregate_association
  2. effect_sensitivity
  3. directional_alignment
  4. heterogeneity
  5. prediction_accuracy
```

### Custom Planner

Register only the checks you want:

```python
from proxymate.planner.validation_planner import ValidationPlanner

planner = ValidationPlanner()
planner.register(RSquaredCheck())
planner.register(ECECheck())
planner.register(AggregateBiasCheck())

checks = planner.recommend(ProxyChallenge.RESTRICTED_LABELS)
```
