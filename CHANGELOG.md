# Changelog

All notable changes to proxymate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Representativity validation level (coverage and weighting checks), completing the
  four-level framework: representativity, unit, estimate, and domain.

### Changed
- Calibration checks and the `MCGradRecalibrator` now use the `mcgrad` package,
  installed via the `calibration` extra (`pip install proxymate[calibration]`).

## [0.1.0]

### Added
- Initial public release: a four-level framework for proxy-metric validation with
  the diagnose-then-correct workflow, adjustment strategies, meta-analysis, and
  publication-ready plots.
