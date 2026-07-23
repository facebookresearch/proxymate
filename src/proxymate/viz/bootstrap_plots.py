# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from proxymate.orchestrator.results import ValidationReport
from proxymate.types.inputs import BootstrappedEstimatesInput
from proxymate.viz._plot_helpers import (
    annotate_check_results,
    clean_axes,
    GRAY,
    GREEN,
    PRIMARY_COLOR,
    PROXY_COLOR,
    RED,
)


def _resolve_bootstrap_input(
    report: ValidationReport,
    proxy_col: str | None = None,
) -> BootstrappedEstimatesInput:
    """Resolve bootstrap input from the report."""
    if report.bootstrap_input is None:
        raise ValueError("ValidationReport has no bootstrap_input stored.")
    return report.bootstrap_input


class BootstrapPlots:
    """Diagnostic plots for bootstrap-based estimate-level checks.

    Methods accept a ``ValidationReport`` with a stored ``bootstrap_input``,
    or a ``BootstrappedEstimatesInput`` directly.

    Pass an ``ax`` argument to draw on an existing axes (e.g. for subplots)
    instead of creating a new figure.
    """

    @staticmethod
    def ci_coverage(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] = (12, 6),
        max_iterations: int = 100,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """CI coverage plot: bootstrap confidence intervals vs primary estimate."""
        bootstrap_input = _resolve_bootstrap_input(report, proxy_col)
        primary_pe = bootstrap_input.primary_estimate.estimate_val
        estimates = bootstrap_input.proxy_bootstrapped_estimates

        # Subsample if too many
        if len(estimates) > max_iterations:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(estimates), max_iterations, replace=False)
            estimates = [estimates[i] for i in sorted(idx)]

        n = len(estimates)
        n_covered = 0

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        for i, est in enumerate(estimates):
            covers = est.lower_bound <= primary_pe <= est.upper_bound
            color = GREEN if covers else RED
            ax.plot(
                [est.lower_bound, est.upper_bound],
                [i, i],
                color=color,
                linewidth=1.5,
                alpha=0.7,
            )
            if covers:
                n_covered += 1

        # Primary estimate line
        ax.axvline(
            primary_pe,
            color="black",
            linewidth=2,
            linestyle="-",
            label=f"Primary = {primary_pe:.4f}",
        )

        coverage_rate = n_covered / n if n > 0 else 0.0
        ax.text(
            0.02,
            0.98,
            f"Coverage: {n_covered}/{n} = {coverage_rate:.0%}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.8},
        )

        ax.set_xlabel("Estimate Value", fontsize=12)
        ax.set_ylabel("Bootstrap Iteration", fontsize=12)
        ax.set_title(
            "CI Coverage: Bootstrap Proxy CIs vs Primary Estimate", fontsize=14
        )
        ax.legend(fontsize=10, loc="lower right")
        clean_axes(ax)
        if report.per_proxy_results:
            proxy_key = proxy_col or next(iter(report.per_proxy_results))
            if proxy_key in report.per_proxy_results:
                annotate_check_results(
                    ax,
                    report.per_proxy_results[proxy_key],
                )
        if standalone:
            fig.tight_layout()
        return fig

    @staticmethod
    def bootstrap_distribution(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] = (10, 6),
        bins: int = 30,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Histogram of bootstrapped proxy point estimates.

        Shows the distribution of proxy estimates across bootstrap iterations,
        with vertical lines for the primary estimate and bootstrap mean.
        """
        bootstrap_input = _resolve_bootstrap_input(report, proxy_col)
        primary_pe = bootstrap_input.primary_estimate.estimate_val
        boot_vals = np.array(
            [est.estimate_val for est in bootstrap_input.proxy_bootstrapped_estimates]
        )
        boot_mean = float(np.mean(boot_vals))
        boot_se = float(np.std(boot_vals, ddof=1))

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        ax.hist(
            boot_vals,  # pyre-ignore[bad-argument-type]
            bins=bins,
            density=True,
            alpha=0.6,
            color=PROXY_COLOR,
            edgecolor="white",
            linewidth=0.5,
            label="Bootstrap estimates",
        )

        # Primary estimate
        ax.axvline(
            primary_pe,
            color=PRIMARY_COLOR,
            linewidth=2,
            linestyle="-",
            label=f"Primary = {primary_pe:.4f}",
        )

        # Bootstrap mean
        ax.axvline(
            boot_mean,
            color=PROXY_COLOR,
            linewidth=2,
            linestyle="--",
            label=f"Bootstrap mean = {boot_mean:.4f}",
        )

        # Bias annotation
        bias = abs(boot_mean - primary_pe)
        normalized_bias = bias / boot_se if boot_se > 0 else float("inf")
        ax.text(
            0.02,
            0.95,
            f"Bias = {bias:.4f}\nBias/SE = {normalized_bias:.4f}\nn = {len(boot_vals)}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.8},
        )

        ax.set_xlabel("Estimate Value", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("Bootstrap Estimate Distribution", fontsize=14)
        ax.legend(fontsize=10)
        clean_axes(ax)
        if standalone:
            fig.tight_layout()
        return fig

    # --- Paired bootstrap vector helpers and plots ---

    @staticmethod
    def _get_paired_vectors(
        report: ValidationReport,
        proxy_col: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract paired primary and proxy bootstrap estimate vectors."""
        bootstrap_input = _resolve_bootstrap_input(report, proxy_col)
        primary_boots = (
            getattr(bootstrap_input, "primary_bootstrapped_estimates", None) or []
        )
        if not primary_boots:
            raise ValueError(
                "No primary_bootstrapped_estimates on bootstrap_input. "
                "Re-run bootstrap to compute both primary and proxy vectors."
            )
        proxy_vals = np.array(
            [e.estimate_val for e in bootstrap_input.proxy_bootstrapped_estimates]
        )
        primary_vals = np.array([e.estimate_val for e in primary_boots])
        return primary_vals, proxy_vals

    @staticmethod
    def difference_distribution(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] = (10, 6),
        bins: int = 30,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Histogram of paired bootstrap differences (primary - proxy).

        Shows the distribution of ``primary_boot[i] - proxy_boot[i]`` across
        iterations. A distribution centered at zero indicates no bias.
        """
        primary_vals, proxy_vals = BootstrapPlots._get_paired_vectors(report, proxy_col)
        diffs = primary_vals - proxy_vals

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        ax.hist(
            diffs,
            bins=bins,
            density=True,
            alpha=0.6,
            color=GRAY,
            edgecolor="white",
            linewidth=0.5,
        )
        mean_diff = float(np.mean(diffs))
        ax.axvline(
            mean_diff,
            color=PRIMARY_COLOR,
            linewidth=2,
            linestyle="--",
            label=f"Mean diff = {mean_diff:.4f}",
        )
        ax.axvline(
            0, color="black", linewidth=1, linestyle="-", alpha=0.5, label="Zero"
        )

        se_diff = float(np.std(diffs, ddof=1))
        ax.text(
            0.02,
            0.95,
            f"Mean = {mean_diff:.4f}\nSE = {se_diff:.4f}\nn = {len(diffs)}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.8},
        )

        ax.set_xlabel("Primary - Proxy", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("Bootstrap Difference Distribution", fontsize=14)
        ax.legend(fontsize=10)
        clean_axes(ax)
        if standalone:
            fig.tight_layout()
        return fig

    @staticmethod
    def qq_plot(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] = (8, 8),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """QQ plot of primary vs proxy bootstrap quantiles.

        If the two distributions are identical, points fall on the identity
        line. Deviations reveal differences in location, scale, or shape.
        """
        primary_vals, proxy_vals = BootstrapPlots._get_paired_vectors(report, proxy_col)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        primary_sorted = np.sort(primary_vals)
        proxy_sorted = np.sort(proxy_vals)

        ax.scatter(
            primary_sorted,
            proxy_sorted,
            alpha=0.5,
            s=20,
            color=PROXY_COLOR,
            edgecolors="white",
            linewidth=0.3,
        )

        # Identity line
        lo = min(primary_sorted[0], proxy_sorted[0])
        hi = max(primary_sorted[-1], proxy_sorted[-1])
        ax.plot(
            [lo, hi], [lo, hi], color=GRAY, linewidth=1, linestyle="--", label="y = x"
        )

        ax.set_xlabel("Primary Bootstrap Quantiles", fontsize=12)
        ax.set_ylabel("Proxy Bootstrap Quantiles", fontsize=12)
        ax.set_title("QQ Plot: Primary vs Proxy", fontsize=14)
        ax.legend(fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        clean_axes(ax)
        if standalone:
            fig.tight_layout()
        return fig

    @staticmethod
    def bias_funnel(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] = (10, 6),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Bias funnel: per-iteration bias vs bootstrap SE.

        Each point is one bootstrap iteration. x-axis is the proxy SE from
        that iteration, y-axis is the difference (primary - proxy). If bias
        is stable across uncertainty levels, points are scattered horizontally.
        """
        bootstrap_input = _resolve_bootstrap_input(report, proxy_col)
        primary_boots = (
            getattr(bootstrap_input, "primary_bootstrapped_estimates", None) or []
        )
        if not primary_boots:
            raise ValueError("No primary_bootstrapped_estimates on bootstrap_input.")

        diffs = np.array(
            [
                p.estimate_val - q.estimate_val
                for p, q in zip(
                    primary_boots, bootstrap_input.proxy_bootstrapped_estimates
                )
            ]
        )
        proxy_ses = np.array(
            [e.standard_error for e in bootstrap_input.proxy_bootstrapped_estimates]
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        ax.scatter(
            proxy_ses,  # pyre-ignore[bad-argument-type]
            diffs,  # pyre-ignore[bad-argument-type]
            alpha=0.5,
            s=20,
            color=PROXY_COLOR,
            edgecolors="white",
            linewidth=0.3,
        )
        ax.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.5)
        mean_diff = float(np.mean(diffs))
        ax.axhline(
            mean_diff,
            color=PRIMARY_COLOR,
            linewidth=1.5,
            linestyle="--",
            label=f"Mean bias = {mean_diff:.4f}",
        )

        ax.set_xlabel("Proxy Bootstrap SE", fontsize=12)
        ax.set_ylabel("Primary - Proxy", fontsize=12)
        ax.set_title("Bias Funnel", fontsize=14)
        ax.legend(fontsize=10)
        clean_axes(ax)
        if standalone:
            fig.tight_layout()
        return fig

    @staticmethod
    def cumulative_bias(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] = (10, 6),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Cumulative running mean of bootstrap differences.

        Shows how the bias estimate stabilizes as more bootstrap iterations
        are accumulated. A flat line indicates convergence.
        """
        primary_vals, proxy_vals = BootstrapPlots._get_paired_vectors(report, proxy_col)
        diffs = primary_vals - proxy_vals
        cumulative_mean = np.cumsum(diffs) / np.arange(1, len(diffs) + 1)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        ax.plot(
            np.arange(1, len(diffs) + 1),
            cumulative_mean,
            color=PRIMARY_COLOR,
            linewidth=1.5,
        )
        ax.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.5)
        final_bias = float(cumulative_mean[-1])
        ax.axhline(
            final_bias,
            color=GRAY,
            linewidth=1,
            linestyle="--",
            label=f"Final = {final_bias:.4f}",
        )

        ax.set_xlabel("Bootstrap Iteration", fontsize=12)
        ax.set_ylabel("Cumulative Mean (Primary - Proxy)", fontsize=12)
        ax.set_title("Cumulative Bias", fontsize=14)
        ax.legend(fontsize=10)
        clean_axes(ax)
        if standalone:
            fig.tight_layout()
        return fig

    @staticmethod
    def paired_scatter(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] = (8, 8),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Scatter of paired bootstrap estimates: primary vs proxy.

        Each point is one bootstrap iteration. Points near the identity line
        indicate strong agreement between primary and proxy estimates.
        """
        primary_vals, proxy_vals = BootstrapPlots._get_paired_vectors(report, proxy_col)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        ax.scatter(
            primary_vals,  # pyre-ignore[bad-argument-type]
            proxy_vals,  # pyre-ignore[bad-argument-type]
            alpha=0.5,
            s=20,
            color=PROXY_COLOR,
            edgecolors="white",
            linewidth=0.3,
        )

        # Identity line
        lo = min(primary_vals.min(), proxy_vals.min())
        hi = max(primary_vals.max(), proxy_vals.max())
        ax.plot(
            [lo, hi], [lo, hi], color=GRAY, linewidth=1, linestyle="--", label="y = x"
        )

        # Correlation
        corr = float(np.corrcoef(primary_vals, proxy_vals)[0, 1])
        ax.text(
            0.02,
            0.95,
            f"r = {corr:.4f}\nn = {len(primary_vals)}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.8},
        )

        ax.set_xlabel("Primary Bootstrap Estimate", fontsize=12)
        ax.set_ylabel("Proxy Bootstrap Estimate", fontsize=12)
        ax.set_title("Paired Bootstrap Scatter", fontsize=14)
        ax.legend(fontsize=10, loc="lower right")
        ax.set_aspect("equal", adjustable="datalim")
        clean_axes(ax)
        if standalone:
            fig.tight_layout()
        return fig

    @staticmethod
    def coverage_by_iteration(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] = (10, 6),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Cumulative coverage rate over bootstrap iterations.

        Shows the proportion of proxy bootstrap CIs that cover the true
        primary estimate, plotted cumulatively over iterations. A line
        converging to the nominal confidence level indicates proper coverage.
        """
        bootstrap_input = _resolve_bootstrap_input(report, proxy_col)
        primary_pe = bootstrap_input.primary_estimate.estimate_val
        estimates = bootstrap_input.proxy_bootstrapped_estimates
        ci_level = bootstrap_input.primary_estimate.confidence_interval_level

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        covers = np.array(
            [
                1.0 if e.lower_bound <= primary_pe <= e.upper_bound else 0.0
                for e in estimates
            ]
        )
        if len(estimates) == 0:
            ax.text(0.5, 0.5, "No bootstrap estimates", ha="center", va="center")
            if standalone:
                fig.tight_layout()
            return fig
        cumulative_coverage = np.cumsum(covers) / np.arange(1, len(covers) + 1)

        ax.plot(
            np.arange(1, len(covers) + 1),
            cumulative_coverage,
            color=PRIMARY_COLOR,
            linewidth=1.5,
        )
        ax.axhline(
            ci_level,
            color=GRAY,
            linewidth=1,
            linestyle="--",
            label=f"Nominal = {ci_level:.0%}",
        )

        final_rate = float(cumulative_coverage[-1])
        ax.text(
            0.02,
            0.05,
            f"Final coverage: {final_rate:.1%}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="bottom",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.8},
        )

        ax.set_xlabel("Bootstrap Iteration", fontsize=12)
        ax.set_ylabel("Cumulative Coverage Rate", fontsize=12)
        ax.set_title("Coverage by Iteration", fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)
        clean_axes(ax)
        if standalone:
            fig.tight_layout()
        return fig
