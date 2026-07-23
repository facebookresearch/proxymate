# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from proxymate.orchestrator.results import ValidationReport
from proxymate.types.data import Estimate
from proxymate.viz._plot_helpers import (
    clean_axes,
    GRAY,
    GREEN,
    PRIMARY_COLOR,
    PROXY_COLOR,
)
from proxymate.viz.bootstrap_plots import BootstrapPlots


def _draw_ci_overlap(
    ax: plt.Axes,
    primary: Estimate,
    proxy: Estimate,
    title: str = "Confidence Interval Overlap",
) -> None:
    """Draw CI overlap bars on an existing axes.

    Reusable drawing logic shared by ``ci_overlap`` (full-data) and
    ``segment_ci_overlap`` (per-segment grid).
    """
    p_lo, p_hi = primary.lower_bound, primary.upper_bound
    q_lo, q_hi = proxy.lower_bound, proxy.upper_bound

    bar_height = 0.3

    # Primary CI
    ax.barh(
        1,
        p_hi - p_lo,
        left=p_lo,
        height=bar_height,
        color=PRIMARY_COLOR,
        alpha=0.5,
        label="Primary CI",
    )
    ax.plot(primary.estimate_val, 1, "o", color=PRIMARY_COLOR, markersize=8, zorder=5)

    # Proxy CI
    ax.barh(
        0,
        q_hi - q_lo,
        left=q_lo,
        height=bar_height,
        color=PROXY_COLOR,
        alpha=0.5,
        label="Proxy CI",
    )
    ax.plot(proxy.estimate_val, 0, "s", color=PROXY_COLOR, markersize=8, zorder=5)

    # Annotate point estimates and CI bounds
    for y, est, lo, hi in [
        (1, primary.estimate_val, p_lo, p_hi),
        (0, proxy.estimate_val, q_lo, q_hi),
    ]:
        ax.text(
            hi + (p_hi - p_lo) * 0.02,
            y,
            f"{est:.4f}\n[{lo:.4f}, {hi:.4f}]",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#333333",
        )

    # Overlap region
    overlap_lo = max(p_lo, q_lo)
    overlap_hi = min(p_hi, q_hi)
    if overlap_hi > overlap_lo:
        ax.axvspan(
            overlap_lo,
            overlap_hi,
            alpha=0.15,
            color=GREEN,
            label="Overlap",
        )
        shorter_ci = min(p_hi - p_lo, q_hi - q_lo)
        overlap_coeff = (
            (overlap_hi - overlap_lo) / shorter_ci if shorter_ci > 0 else 0.0
        )
    else:
        overlap_coeff = 0.0

    ax.text(
        0.02,
        0.95,
        f"Overlap coefficient: {overlap_coeff:.4f}",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.8},
    )

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Proxy", "Primary"], fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_ylim(-0.5, 1.5)
    # Expand x-axis to fit annotation text
    all_bounds = [p_lo, p_hi, q_lo, q_hi]
    span = max(all_bounds) - min(all_bounds)
    ax.set_xlim(right=max(all_bounds) + span * 0.45)
    ax.legend(
        fontsize=10,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
    )
    clean_axes(ax)


class EstimateLevelPlots:
    """Diagnostic plots for estimate-level checks.

    Includes both non-bootstrap (AggregateInput) and bootstrap
    (BootstrappedEstimatesInput) visualizations. All methods accept an
    ``ax`` argument for subplot embedding.

    When ``report.aggregate_inputs`` is populated (true estimates from the
    estimator on full data), plots use the true confidence intervals.
    """

    @staticmethod
    def _resolve_estimates(
        report: ValidationReport,
        proxy_col: str | None = None,
    ) -> tuple["Estimate", "Estimate"]:
        """Resolve the true primary and proxy Estimates for plotting.

        Prefers ``report.aggregate_inputs`` (true CIs from estimator on
        full data). Falls back to bootstrap_input primary + bootstrap
        distribution summary if aggregate_inputs is not available.
        """
        from proxymate.types.data import Estimate as _Estimate

        # Try aggregate_inputs first (true CIs)
        agg_inputs = getattr(report, "aggregate_inputs", None)
        if agg_inputs:
            if proxy_col is not None and proxy_col in agg_inputs:
                agg = agg_inputs[proxy_col]
                return agg.primary_estimate, agg.proxy_estimate
            # Fall back to first available proxy
            first_key = next(iter(agg_inputs))
            agg = agg_inputs[first_key]
            return agg.primary_estimate, agg.proxy_estimate

        # Fall back to bootstrap_input
        if report.bootstrap_input is None:
            raise ValueError(
                "ValidationReport has no aggregate_inputs or bootstrap_input."
            )
        primary = report.bootstrap_input.primary_estimate

        # Use true proxy estimate if available
        proxy_estimate = getattr(report.bootstrap_input, "proxy_estimate", None)
        if proxy_estimate is not None:
            return primary, proxy_estimate

        # Last resort: derive proxy estimate from bootstrap distribution
        boot_vals = np.array(
            [
                est.estimate_val
                for est in report.bootstrap_input.proxy_bootstrapped_estimates
            ]
        )
        proxy_mean = float(np.mean(boot_vals))
        proxy_se = float(np.std(boot_vals, ddof=1))
        ci_level = primary.confidence_interval_level
        alpha = 1.0 - ci_level
        lo = float(np.percentile(boot_vals, 100 * alpha / 2))
        hi = float(np.percentile(boot_vals, 100 * (1 - alpha / 2)))
        proxy_estimate = _Estimate(
            confidence_interval_level=ci_level,
            estimate_val=proxy_mean,
            lower_bound=lo,
            upper_bound=hi,
            sample_size=len(boot_vals),
            standard_error=proxy_se,
        )
        return primary, proxy_estimate

    @staticmethod
    def ci_overlap(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] = (12, 5),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Confidence interval overlap visualization.

        Shows primary and proxy CIs as horizontal bars with the overlap
        region highlighted. Uses true confidence intervals from the
        estimator. Displays the overlap coefficient.
        """
        primary, proxy = EstimateLevelPlots._resolve_estimates(report, proxy_col)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        _draw_ci_overlap(ax, primary, proxy)
        if standalone:
            fig.tight_layout(rect=(0, 0, 0.82, 1))
        return fig

    # Delegate to BootstrapPlots for bootstrap-specific plots
    ci_coverage = BootstrapPlots.ci_coverage
    bootstrap_distribution = BootstrapPlots.bootstrap_distribution
    difference_distribution = BootstrapPlots.difference_distribution
    qq_plot = BootstrapPlots.qq_plot
    bias_funnel = BootstrapPlots.bias_funnel
    cumulative_bias = BootstrapPlots.cumulative_bias
    paired_scatter = BootstrapPlots.paired_scatter
    coverage_by_iteration = BootstrapPlots.coverage_by_iteration

    @staticmethod
    def segment_ci_overlap(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> plt.Figure:  # noqa: C901
        """CI overlap comparing primary vs proxy across segments.

        Uses two subplots:
        - **Top**: Overall (full-data) CI overlap using the same style as
          ``ci_overlap`` (horizontal bars, overlap shading, overlap
          coefficient, annotated point estimates).
        - **Bottom**: Per-segment CI overlap bars stacked vertically, one
          pair (primary + proxy) per segment, with overlap shading and
          overlap coefficients. Segments are grouped by segment group
          with separator lines and group labels.

        Args:
            report: ValidationReport with ``segment_results`` populated.
            proxy_col: Which proxy to plot. If None, uses the first available.
            figsize: Optional figure size override.
        """
        # --- Resolve overall estimates ---
        overall_primary: Estimate | None = None
        overall_proxy: Estimate | None = None
        try:
            overall_primary, overall_proxy = EstimateLevelPlots._resolve_estimates(
                report, proxy_col
            )
        except ValueError:
            pass
        has_overall = overall_primary is not None and overall_proxy is not None

        # --- Collect per-segment estimates, grouped ---
        grouped: dict[str, list[tuple[str, Estimate, Estimate]]] = {}
        for seg in report.segment_results:
            agg_inputs = getattr(seg, "aggregate_inputs", None)
            if not agg_inputs:
                continue
            if proxy_col is not None and proxy_col in agg_inputs:
                agg = agg_inputs[proxy_col]
            elif agg_inputs:
                agg = next(iter(agg_inputs.values()))
            else:
                continue
            grouped.setdefault(seg.group_name, []).append(
                (seg.segment_name, agg.primary_estimate, agg.proxy_estimate)
            )

        if not grouped and not has_overall:
            fig, ax = plt.subplots()
            ax.text(
                0.5,
                0.5,
                "No segment aggregate inputs to display.\n"
                "Run validation with segments + estimator + aggregate checks.",
                ha="center",
                va="center",
                fontsize=12,
            )
            return fig

        # --- Build flat segment list with group metadata ---
        seg_entries: list[tuple[str, Estimate, Estimate]] = []
        group_boundary_indices: list[int] = []
        group_label_info: list[tuple[float, str]] = []

        ROW_SPACING = 3  # y-units per segment (2 bars + gap)
        BAR_HEIGHT = 0.35

        for group_name, segments in grouped.items():
            group_start_idx = len(seg_entries)
            for seg_name, primary, proxy in segments:
                seg_entries.append((seg_name, primary, proxy))
            group_end_idx = len(seg_entries) - 1
            group_boundary_indices.append(group_start_idx)
            start_y = group_start_idx * ROW_SPACING + 0.5
            end_y = group_end_idx * ROW_SPACING + 0.5
            group_label_info.append(((start_y + end_y) / 2, group_name))

        n_segments = len(seg_entries)
        has_segments = n_segments > 0

        # --- Figure layout ---
        if has_overall and has_segments:
            from matplotlib.gridspec import GridSpec

            seg_height = max(4, n_segments * 1.2 + 1)
            overall_height = 3
            if figsize is None:
                figsize = (14, overall_height + seg_height)
            fig = plt.figure(figsize=figsize)
            gs = GridSpec(
                2,
                1,
                height_ratios=[overall_height, seg_height],
                hspace=0.35,
                figure=fig,
            )
            ax_overall = fig.add_subplot(gs[0])
            ax_segments = fig.add_subplot(gs[1])
        elif has_overall:
            if figsize is None:
                figsize = (14, 5)
            fig, ax_overall = plt.subplots(figsize=figsize)
            ax_segments = None
        else:
            seg_height = max(4, n_segments * 1.2 + 1)
            if figsize is None:
                figsize = (14, seg_height)
            fig, ax_segments = plt.subplots(figsize=figsize)
            ax_overall = None

        # --- Top subplot: Overall CI overlap ---
        if (
            ax_overall is not None
            and overall_primary is not None
            and overall_proxy is not None
        ):
            _draw_ci_overlap(
                ax_overall,
                overall_primary,
                overall_proxy,
                title="Overall",
            )

        # --- Bottom subplot: Per-segment CI overlap ---
        if ax_segments is not None and has_segments:
            legend_added_primary = False
            legend_added_proxy = False
            legend_added_overlap = False

            for idx, (_label, primary, proxy) in enumerate(seg_entries):
                y_center = idx * ROW_SPACING + 0.5
                y_primary = y_center + 0.5
                y_proxy = y_center - 0.5

                p_lo, p_hi = primary.lower_bound, primary.upper_bound
                q_lo, q_hi = proxy.lower_bound, proxy.upper_bound

                # Primary CI bar
                ax_segments.barh(
                    y_primary,
                    p_hi - p_lo,
                    left=p_lo,
                    height=BAR_HEIGHT,
                    color=PRIMARY_COLOR,
                    alpha=0.45,
                    label="Primary CI" if not legend_added_primary else None,
                )
                ax_segments.plot(
                    primary.estimate_val,
                    y_primary,
                    "o",
                    color=PRIMARY_COLOR,
                    markersize=7,
                    zorder=5,
                )
                legend_added_primary = True

                # Proxy CI bar
                ax_segments.barh(
                    y_proxy,
                    q_hi - q_lo,
                    left=q_lo,
                    height=BAR_HEIGHT,
                    color=PROXY_COLOR,
                    alpha=0.45,
                    label="Proxy CI" if not legend_added_proxy else None,
                )
                ax_segments.plot(
                    proxy.estimate_val,
                    y_proxy,
                    "s",
                    color=PROXY_COLOR,
                    markersize=7,
                    zorder=5,
                )
                legend_added_proxy = True

                # Overlap region
                overlap_lo = max(p_lo, q_lo)
                overlap_hi = min(p_hi, q_hi)
                if overlap_hi > overlap_lo:
                    ax_segments.fill_betweenx(
                        [y_proxy - BAR_HEIGHT / 2, y_primary + BAR_HEIGHT / 2],
                        overlap_lo,
                        overlap_hi,
                        alpha=0.15,
                        color=GREEN,
                        label="Overlap" if not legend_added_overlap else None,
                    )
                    legend_added_overlap = True
                    shorter_ci = min(p_hi - p_lo, q_hi - q_lo)
                    oc = (
                        (overlap_hi - overlap_lo) / shorter_ci
                        if shorter_ci > 0
                        else 0.0
                    )
                else:
                    oc = 0.0

                # OC annotation
                x_max = max(p_hi, q_hi)
                x_range = max(p_hi - p_lo, q_hi - q_lo)
                ax_segments.text(
                    x_max + x_range * 0.03,
                    y_center,
                    f"OC={oc:.2f}",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="#333333",
                )

                # Point estimate annotations
                ax_segments.text(
                    p_hi + x_range * 0.03,
                    y_primary,
                    f"{primary.estimate_val:.4f}",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color=PRIMARY_COLOR,
                )
                ax_segments.text(
                    q_hi + x_range * 0.03,
                    y_proxy,
                    f"{proxy.estimate_val:.4f}",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color=PROXY_COLOR,
                )

            # Auto-adjust x-axis to fit all segment bars and annotations
            all_lo = [min(p.lower_bound, q.lower_bound) for _, p, q in seg_entries]
            all_hi = [max(p.upper_bound, q.upper_bound) for _, p, q in seg_entries]
            global_min = min(all_lo)
            global_max = max(all_hi)
            global_span = global_max - global_min
            ax_segments.set_xlim(
                left=global_min - global_span * 0.05,
                right=global_max + global_span * 0.45,
            )

            # Group separator lines
            for boundary_idx in group_boundary_indices[1:]:
                y_sep = boundary_idx * ROW_SPACING - ROW_SPACING / 2 + 0.1
                ax_segments.axhline(
                    y_sep,
                    color=GRAY,
                    linewidth=0.8,
                    linestyle="--",
                    alpha=0.3,
                )

            # Y-axis labels
            y_ticks = [idx * ROW_SPACING + 0.5 for idx in range(n_segments)]
            y_labels_list = [label for label, _, _ in seg_entries]
            ax_segments.set_yticks(y_ticks)
            ax_segments.set_yticklabels(y_labels_list, fontsize=10)

            # Group name annotations
            for y_center, group_name in group_label_info:
                ax_segments.text(
                    1.01,
                    y_center,
                    group_name,
                    transform=ax_segments.get_yaxis_transform(),
                    fontsize=10,
                    fontweight="bold",
                    color=GRAY,
                    va="center",
                    ha="left",
                )

            ax_segments.set_ylim(
                -ROW_SPACING / 2 + 0.1,
                n_segments * ROW_SPACING - ROW_SPACING / 2 + 0.9,
            )
            ax_segments.set_title("By Segment", fontsize=16, fontweight="bold")
            ax_segments.legend(
                fontsize=10,
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                frameon=False,
            )
            clean_axes(ax_segments)

        fig.suptitle(
            "CI Overlap: Overall & Segments", fontsize=16, fontweight="bold", y=1.01
        )
        fig.tight_layout(rect=[0, 0, 0.88, 0.96])
        return fig

    @staticmethod
    def estimate_comparison(
        estimates: list[tuple[str, Estimate]],
        ax: "plt.Axes | None" = None,
        figsize: tuple[float, float] = (8, 4),
        title: str = "Estimate Comparison",
        colors: list[str] | None = None,
    ) -> "plt.Figure":
        """Compare multiple estimates with confidence intervals.

        Horizontal bar chart showing point estimates with 95% CIs.
        Useful for comparing naive, IW-corrected, and PPI-corrected
        prevalence estimates side by side.

        Args:
            estimates: List of (label, Estimate) tuples.
            ax: Optional axes for subplot embedding.
            figsize: Figure size (ignored when ax is provided).
            title: Plot title.
            colors: Optional list of colors per estimate.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = cast(plt.Figure, ax.figure)

        default_colors = [
            "#1f77b4",
            "#d62728",
            "#2ca02c",
            "#ff7f0e",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
        ]
        if colors is None:
            colors = default_colors

        valid = [
            (label, est, colors[i % len(colors)])
            for i, (label, est) in enumerate(estimates)
            if not np.isnan(est.estimate_val)
        ]

        y_pos = np.arange(len(valid))
        for i, (_label, est, color) in enumerate(valid):
            ax.barh(i, est.estimate_val, color=color, alpha=0.7, height=0.5)
            ax.errorbar(
                est.estimate_val,
                i,
                xerr=[
                    [est.estimate_val - est.lower_bound],
                    [est.upper_bound - est.estimate_val],
                ],
                fmt="none",
                color="black",
                capsize=4,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels([v[0] for v in valid])
        ax.set_xlabel("Estimate")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        clean_axes(ax)
        return fig


def plot_all_estimate_level(
    report: ValidationReport,
    proxy_name: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot all estimate-level diagnostics on a single figure.

    Combines non-bootstrap aggregate plots and bootstrap plots into a
    3x3 grid:

    1. **CI overlap** — Confidence interval overlap visualization
    2. **Bootstrap distribution** — Histogram of bootstrap estimates
    3. **Bootstrap CI coverage** — Forest plot of bootstrap CIs
    4. **Difference distribution** — Histogram of paired differences
    5. **QQ plot** — Primary vs proxy bootstrap quantiles
    6. **Bias funnel** — Per-iteration bias vs SE
    7. **Cumulative bias** — Running mean of differences
    8. **Paired scatter** — Primary vs proxy bootstrap scatter
    9. **Coverage by iteration** — Cumulative coverage rate

    Args:
        report: ValidationReport with bootstrap_input populated.
        proxy_name: Which proxy to plot for. If None, uses the first available.
        figsize: Optional figure size override.

    Returns a single figure with subplots.
    """
    if report.bootstrap_input is None:
        raise ValueError(
            "ValidationReport has no bootstrap_input. "
            "Run bootstrap checks first (via BootstrapConfig or manual input)."
        )

    nrows = 3
    ncols = 3

    if figsize is None:
        figsize = (7 * ncols, 5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat: list[plt.Axes] = list(np.array(axes).flatten())

    EstimateLevelPlots.ci_overlap(report, proxy_col=proxy_name, ax=axes_flat[0])
    BootstrapPlots.bootstrap_distribution(report, proxy_col=proxy_name, ax=axes_flat[1])
    BootstrapPlots.difference_distribution(
        report, proxy_col=proxy_name, ax=axes_flat[2]
    )
    BootstrapPlots.ci_coverage(report, proxy_col=proxy_name, ax=axes_flat[3])
    BootstrapPlots.paired_scatter(report, proxy_col=proxy_name, ax=axes_flat[4])
    BootstrapPlots.qq_plot(report, proxy_col=proxy_name, ax=axes_flat[5])
    BootstrapPlots.bias_funnel(report, proxy_col=proxy_name, ax=axes_flat[6])
    BootstrapPlots.cumulative_bias(report, proxy_col=proxy_name, ax=axes_flat[7])
    BootstrapPlots.coverage_by_iteration(report, proxy_col=proxy_name, ax=axes_flat[8])

    fig.suptitle("Estimate-Level Diagnostics", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig
