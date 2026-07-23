---
id: framework
title: Framework Overview
sidebar_position: 1
---

# Framework Overview

`proxymate` decomposes proxy validation into four levels. The first addresses **data quality** — whether the validation data should be weighted. The remaining three address **proxy quality** at increasing levels of aggregation.

## Representativity Level

*Should we weight the validation data?*

The validation data $\mathcal{V}_k$ may not be representative of the domain population $P_k$. For example, a review queue over-represents flagged content. The Representativity Level detects such mismatches via covariate balance checks (SMD, KS test) and weight diagnostics (ESS, weight extremity), then determines whether importance weighting (IW) can correct for them.

**Outcomes:** No gap (skip weighting), IW adjustment (weights applied downstream), or gap too severe (flag and recommend fallbacks).

## Unit Level

*Does the proxy track the primary at the individual level?*

Evaluates six properties of measurement quality:

- **Accuracy** — systematic bias ($E[Y^* - Y] = 0$?)
- **Precision** — noise level ($R^2$, RMSE)
- **Agreement** — identity-line concordance (Lin's CCC)
- **Calibration** — probability calibration (ECE, Brier)
- **Discrimination** — ranking quality (AUC, Kendall $\tau$)
- **Distributional fidelity** — marginal distribution match (KS, Wasserstein)

## Estimate Level

*Does the proxy give the same aggregate answer as the primary?*

Even if a proxy is accurate at the unit level, individual errors can compound in unpredictable ways when aggregated. The Estimate Level checks aggregate bias, CI coverage, sign agreement, and effect size agreement within a single domain.

## Domain Level

*Is the proxy-primary relationship stable across domains?*

The most demanding level. Uses meta-regression across $K$ historical domains to assess association ($R^2$), sensitivity (slope $\gamma_1 \approx 1$?), directional alignment, prediction accuracy (LOO), and heterogeneity ($I^2$, $\tau^2$).

When domain covariances are available, `proxymate` uses an iterative GLS regression that accounts for measurement error in both the proxy and primary estimates.

## Level Independence

The levels are complementary, not sequential in the strict sense:

- Strong Unit Level performance does **not** guarantee Estimate Level validity (errors can compound rather than cancel)
- Estimate Level validity within one domain does **not** guarantee Domain Level transportability
- Representativity corrections are prerequisite for valid downstream checks
