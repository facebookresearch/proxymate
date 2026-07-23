# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from unittest import TestCase

import matplotlib
import matplotlib.pyplot as plt
from proxymate.lib.base_classes.data_types import Estimate
from proxymate.orchestrator.results import ValidationReport
from proxymate.types.inputs import MetaAnalysisInput
from proxymate.viz.meta_analysis_plots import MetaAnalysisPlots, plot_all_meta_analysis

matplotlib.use("Agg")


def _make_estimate(
    val: float,
    se: float = 0.1,
    n: int = 100,
) -> Estimate:
    return Estimate(
        confidence_interval_level=0.95,
        estimate_val=val,
        lower_bound=val - 1.96 * se,
        upper_bound=val + 1.96 * se,
        sample_size=n,
        standard_error=se,
    )


def _make_report(
    with_names: bool = True,
) -> ValidationReport:
    pairs = [
        (_make_estimate(1.0), _make_estimate(1.1)),
        (_make_estimate(2.0), _make_estimate(2.2)),
        (_make_estimate(0.5, se=0.2), _make_estimate(0.4, se=0.15)),
        (_make_estimate(3.0, se=0.05), _make_estimate(2.8, se=0.08)),
        (_make_estimate(-1.0), _make_estimate(-0.9)),
    ]
    meta_input = MetaAnalysisInput(
        domain_estimates=pairs,
        domain_names=["A", "B", "C", "D", "E"] if with_names else None,
    )
    return ValidationReport(
        meta_analysis_inputs={"proxy": meta_input},
    )


def _make_empty_report() -> ValidationReport:
    meta_input = MetaAnalysisInput(domain_estimates=[])
    return ValidationReport(
        meta_analysis_inputs={"proxy": meta_input},
    )


class ScatterTest(TestCase):
    def test_returns_figure(self) -> None:
        report = _make_report()
        fig = MetaAnalysisPlots.scatter(report, proxy_col="proxy")
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_ax(self) -> None:
        report = _make_report()
        parent_fig, ax = plt.subplots()
        fig = MetaAnalysisPlots.scatter(report, proxy_col="proxy", ax=ax)
        self.assertIs(fig, parent_fig)
        plt.close(parent_fig)

    def test_empty_domains(self) -> None:
        report = _make_empty_report()
        fig = MetaAnalysisPlots.scatter(report, proxy_col="proxy")
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_domain_names(self) -> None:
        report = _make_report(with_names=False)
        fig = MetaAnalysisPlots.scatter(report, proxy_col="proxy")
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_auto_proxy_col(self) -> None:
        report = _make_report()
        fig = MetaAnalysisPlots.scatter(report)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_invalid_proxy_col_raises(self) -> None:
        report = _make_report()
        with self.assertRaises(ValueError):
            MetaAnalysisPlots.scatter(report, proxy_col="nonexistent")

    def test_no_meta_inputs_raises(self) -> None:
        report = ValidationReport()
        with self.assertRaises(ValueError):
            MetaAnalysisPlots.scatter(report)


class ForestPlotTest(TestCase):
    def test_returns_figure(self) -> None:
        report = _make_report()
        fig = MetaAnalysisPlots.forest_plot(report, proxy_col="proxy")
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_ax(self) -> None:
        report = _make_report()
        parent_fig, ax = plt.subplots()
        fig = MetaAnalysisPlots.forest_plot(report, proxy_col="proxy", ax=ax)
        self.assertIs(fig, parent_fig)
        plt.close(parent_fig)

    def test_empty_domains(self) -> None:
        report = _make_empty_report()
        fig = MetaAnalysisPlots.forest_plot(report, proxy_col="proxy")
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


class BlandAltmanTest(TestCase):
    def test_returns_figure(self) -> None:
        report = _make_report()
        fig = MetaAnalysisPlots.bland_altman(report, proxy_col="proxy")
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_ax(self) -> None:
        report = _make_report()
        parent_fig, ax = plt.subplots()
        fig = MetaAnalysisPlots.bland_altman(report, proxy_col="proxy", ax=ax)
        self.assertIs(fig, parent_fig)
        plt.close(parent_fig)

    def test_empty_domains(self) -> None:
        report = _make_empty_report()
        fig = MetaAnalysisPlots.bland_altman(report, proxy_col="proxy")
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_domain(self) -> None:
        """Single domain: std_diff should be 0, limits of agreement collapse."""
        pairs = [(_make_estimate(1.0), _make_estimate(1.1))]
        meta_input = MetaAnalysisInput(domain_estimates=pairs)
        report = ValidationReport(meta_analysis_inputs={"proxy": meta_input})
        fig = MetaAnalysisPlots.bland_altman(report, proxy_col="proxy")
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


class FunnelPlotTest(TestCase):
    def test_returns_figure(self) -> None:
        report = _make_report()
        fig = MetaAnalysisPlots.funnel_plot(report, proxy_col="proxy")
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_ax(self) -> None:
        report = _make_report()
        parent_fig, ax = plt.subplots()
        fig = MetaAnalysisPlots.funnel_plot(report, proxy_col="proxy", ax=ax)
        self.assertIs(fig, parent_fig)
        plt.close(parent_fig)

    def test_zero_se(self) -> None:
        """Domains with SE=0 should not crash (precision becomes NaN)."""
        pairs = [
            (_make_estimate(1.0, se=0.0), _make_estimate(1.1, se=0.0)),
            (_make_estimate(2.0, se=0.1), _make_estimate(2.2, se=0.1)),
        ]
        meta_input = MetaAnalysisInput(domain_estimates=pairs)
        report = ValidationReport(meta_analysis_inputs={"proxy": meta_input})
        fig = MetaAnalysisPlots.funnel_plot(report, proxy_col="proxy")
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


class CIWidthComparisonTest(TestCase):
    def test_returns_figure(self) -> None:
        report = _make_report()
        fig = MetaAnalysisPlots.ci_width_comparison(report, proxy_col="proxy")
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_ax(self) -> None:
        report = _make_report()
        parent_fig, ax = plt.subplots()
        fig = MetaAnalysisPlots.ci_width_comparison(report, proxy_col="proxy", ax=ax)
        self.assertIs(fig, parent_fig)
        plt.close(parent_fig)


class ResidualByDomainTest(TestCase):
    def test_returns_figure(self) -> None:
        report = _make_report()
        fig = MetaAnalysisPlots.residual_by_domain(report, proxy_col="proxy")
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_ax(self) -> None:
        report = _make_report()
        parent_fig, ax = plt.subplots()
        fig = MetaAnalysisPlots.residual_by_domain(report, proxy_col="proxy", ax=ax)
        self.assertIs(fig, parent_fig)
        plt.close(parent_fig)

    def test_empty_domains(self) -> None:
        report = _make_empty_report()
        fig = MetaAnalysisPlots.residual_by_domain(report, proxy_col="proxy")
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


class CumulativeEffectTest(TestCase):
    def test_returns_figure(self) -> None:
        report = _make_report()
        fig = MetaAnalysisPlots.cumulative_effect(report, proxy_col="proxy")
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_ax(self) -> None:
        report = _make_report()
        parent_fig, ax = plt.subplots()
        fig = MetaAnalysisPlots.cumulative_effect(report, proxy_col="proxy", ax=ax)
        self.assertIs(fig, parent_fig)
        plt.close(parent_fig)

    def test_zero_se_falls_back(self) -> None:
        """When all SEs are zero, falls back to equal weights."""
        pairs = [
            (_make_estimate(1.0, se=0.0), _make_estimate(1.1, se=0.0)),
            (_make_estimate(2.0, se=0.0), _make_estimate(2.2, se=0.0)),
            (_make_estimate(3.0, se=0.0), _make_estimate(3.3, se=0.0)),
        ]
        meta_input = MetaAnalysisInput(domain_estimates=pairs)
        report = ValidationReport(meta_analysis_inputs={"proxy": meta_input})
        fig = MetaAnalysisPlots.cumulative_effect(report, proxy_col="proxy")
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


class SampleSizeVsEffectTest(TestCase):
    def test_returns_figure(self) -> None:
        report = _make_report()
        fig = MetaAnalysisPlots.sample_size_vs_effect(report, proxy_col="proxy")
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_ax(self) -> None:
        report = _make_report()
        parent_fig, ax = plt.subplots()
        fig = MetaAnalysisPlots.sample_size_vs_effect(report, proxy_col="proxy", ax=ax)
        self.assertIs(fig, parent_fig)
        plt.close(parent_fig)


class PlotAllMetaAnalysisTest(TestCase):
    def test_returns_figure(self) -> None:
        report = _make_report()
        fig = plot_all_meta_analysis(report, proxy_col="proxy")
        self.assertIsInstance(fig, plt.Figure)
        axes = fig.get_axes()
        # 2x3 grid = 6 axes total, all 6 used
        self.assertEqual(len(axes), 6)
        plt.close(fig)

    def test_auto_proxy_col(self) -> None:
        report = _make_report()
        fig = plot_all_meta_analysis(report)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_figsize(self) -> None:
        report = _make_report()
        fig = plot_all_meta_analysis(report, figsize=(20, 10))
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_meta_inputs_raises(self) -> None:
        report = ValidationReport()
        with self.assertRaises(ValueError):
            plot_all_meta_analysis(report)

    def test_subplot_methods_do_not_call_tight_layout(self) -> None:
        """Verify that subplot-embedded methods don't disrupt parent figure."""
        report = _make_report()
        fig, axes = plt.subplots(2, 3, figsize=(21, 10))
        axes_flat = axes.flatten()
        # Call each method with ax — should not raise
        MetaAnalysisPlots.scatter(report, proxy_col="proxy", ax=axes_flat[0])
        MetaAnalysisPlots.forest_plot(report, proxy_col="proxy", ax=axes_flat[1])
        MetaAnalysisPlots.bland_altman(report, proxy_col="proxy", ax=axes_flat[2])
        MetaAnalysisPlots.funnel_plot(report, proxy_col="proxy", ax=axes_flat[3])
        MetaAnalysisPlots.residual_by_domain(report, proxy_col="proxy", ax=axes_flat[4])
        MetaAnalysisPlots.cumulative_effect(report, proxy_col="proxy", ax=axes_flat[5])
        fig.tight_layout()
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
