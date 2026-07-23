# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from proxymate.orchestrator.results import ProxyValidationResult, ValidationReport
from proxymate.viz._plot_helpers import (
    clean_axes,
    GRAY,
    GREEN,
    iter_all_check_results,
    RED,
)

# Re-export plot classes and functions from their dedicated modules so that
# ``from proxymate.viz.summary_plots import UnitLevelPlots`` etc.
# keeps working.
from proxymate.viz.aggregate_level_plots import (
    EstimateLevelPlots as EstimateLevelPlots,
    plot_all_estimate_level as plot_all_estimate_level,
)
from proxymate.viz.bootstrap_plots import BootstrapPlots as BootstrapPlots
from proxymate.viz.meta_analysis_plots import (
    MetaAnalysisPlots as MetaAnalysisPlots,
    plot_all_meta_analysis as plot_all_meta_analysis,
)
from proxymate.viz.representativity_plots import (
    plot_all_representativity as plot_all_representativity,
    RepresentativityPlots as RepresentativityPlots,
)
from proxymate.viz.unit_level_plots import (
    plot_all_unit_level as plot_all_unit_level,
    UnitLevelPlots as UnitLevelPlots,
)

# Backward-compatible free-function aliases
plot_scatter: Callable[..., plt.Figure] = UnitLevelPlots.scatter
plot_residuals: Callable[..., plt.Figure] = UnitLevelPlots.residuals
plot_distributions: Callable[..., plt.Figure] = UnitLevelPlots.distributions
plot_calibration: Callable[..., plt.Figure] = UnitLevelPlots.calibration
plot_roc_auc: Callable[..., plt.Figure] = UnitLevelPlots.roc_auc
plot_precision_recall: Callable[..., plt.Figure] = UnitLevelPlots.precision_recall
plot_score_distribution: Callable[..., plt.Figure] = UnitLevelPlots.score_distribution
plot_lift: Callable[..., plt.Figure] = UnitLevelPlots.lift
plot_ks: Callable[..., plt.Figure] = UnitLevelPlots.ks
plot_confusion_matrix: Callable[..., plt.Figure] = UnitLevelPlots.confusion_matrix
plot_classification_metrics: Callable[..., plt.Figure] = (
    UnitLevelPlots.classification_metrics
)
plot_cohens_kappa: Callable[..., plt.Figure] = UnitLevelPlots.cohens_kappa
plot_prevalence: Callable[..., plt.Figure] = UnitLevelPlots.prevalence
plot_ci_coverage: Callable[..., plt.Figure] = BootstrapPlots.ci_coverage
plot_bootstrap_distribution: Callable[..., plt.Figure] = (
    BootstrapPlots.bootstrap_distribution
)
plot_ci_overlap: Callable[..., plt.Figure] = EstimateLevelPlots.ci_overlap
plot_segment_ci_overlap: Callable[..., plt.Figure] = (
    EstimateLevelPlots.segment_ci_overlap
)
plot_meta_analysis: Callable[..., plt.Figure] = MetaAnalysisPlots.scatter
plot_love_plot: Callable[..., plt.Figure] = RepresentativityPlots.love_plot
plot_distribution_comparison: Callable[..., plt.Figure] = (
    RepresentativityPlots.distribution_comparison
)
plot_weight_distribution: Callable[..., plt.Figure] = (
    RepresentativityPlots.weight_distribution
)
plot_gap_reduction: Callable[..., plt.Figure] = RepresentativityPlots.gap_reduction_plot


# ========================================================================
# Results-level plots
# ========================================================================


def plot_summary(
    report: ValidationReport,
    proxy_col: str | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Bar chart of projection scores grouped by sub-property.

    If proxy_col is None and only one proxy exists, uses that one.
    """
    if proxy_col is None:
        if len(report.per_proxy_results) == 1:
            proxy_col = next(iter(report.per_proxy_results))
        else:
            raise ValueError(
                "Multiple proxies found. Specify proxy_col or use plot_comparison."
            )

    proxy_result = report.per_proxy_results[proxy_col]
    return _plot_proxy_result(
        proxy_result, title=f"Validation Summary: {proxy_col}", figsize=figsize
    )


def _plot_proxy_result(
    proxy_result: ProxyValidationResult,
    title: str = "Validation Summary",
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Bar chart of projection scores for a single proxy."""
    names: list[str] = []
    projections: list[float] = []
    thresholds: list[float] = []
    colors: list[str] = []
    sub_props: list[str] = []

    for prop_name, r in iter_all_check_results(proxy_result):
        names.append(r.check_name)
        projections.append(r.projection_result)
        thresholds.append(r.threshold)
        colors.append(GREEN if r.thresholding_result else RED)
        sub_props.append(prop_name)

    if not names:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No check results to display", ha="center", va="center")
        return fig

    n_passed = sum(1 for c in colors if c == GREEN)
    n_total = len(colors)

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(names))

    ax.barh(y_pos, projections, color=colors, alpha=0.85, height=0.7)

    # Threshold markers as vertical dashed lines per bar
    for i, thresh in enumerate(thresholds):
        ax.plot(
            [thresh, thresh],
            [i - 0.35, i + 0.35],
            color="black",
            linewidth=1.5,
            linestyle="--",
            alpha=0.6,
        )

    # Sub-property separators
    prev_prop = sub_props[0] if sub_props else ""
    for i, prop in enumerate(sub_props):
        if prop != prev_prop:
            ax.axhline(i - 0.5, color=GRAY, linewidth=0.5, linestyle="-", alpha=0.4)
        prev_prop = prop

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Projection Score", fontsize=12)
    ax.set_xlim(0, 1.05)
    ax.set_title(f"{title} ({n_passed}/{n_total} passed)", fontsize=14)
    ax.invert_yaxis()
    clean_axes(ax)
    fig.tight_layout()
    return fig


def plot_comparison(
    report: ValidationReport,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:  # noqa: C901
    """Grouped bar chart comparing projection scores across proxies, organized
    by sub-property."""
    if len(report.per_proxy_results) < 2:
        raise ValueError("Need at least 2 proxies for comparison.")

    proxy_cols = sorted(report.per_proxy_results.keys())

    # Collect ordered checks grouped by sub-property across all proxies
    all_checks_by_prop: dict[str, list[str]] = {}
    for proxy_col in proxy_cols:
        proxy_result = report.per_proxy_results[proxy_col]
        for prop_name, r in iter_all_check_results(proxy_result):
            all_checks_by_prop.setdefault(prop_name, [])
            if r.check_name not in all_checks_by_prop[prop_name]:
                all_checks_by_prop[prop_name].append(r.check_name)

    # Build per-proxy score lookup
    proxy_scores: dict[str, dict[str, float]] = {}
    for proxy_col in proxy_cols:
        proxy_scores[proxy_col] = _extract_check_scores(
            report.per_proxy_results[proxy_col]
        )

    # Flatten into ordered check list and track group boundaries
    check_names: list[str] = []
    prop_labels: list[str] = []
    group_boundaries: list[int] = []  # x positions where new groups start
    group_centers: list[float] = []  # center x for group label
    group_display: list[str] = []  # display name

    for prop_name, checks in all_checks_by_prop.items():
        group_boundaries.append(len(check_names))
        for check in checks:
            check_names.append(check)
            prop_labels.append(prop_name)
        center = group_boundaries[-1] + (len(checks) - 1) / 2
        group_centers.append(center)
        group_display.append(prop_name.replace("_", " ").title())

    n_checks = len(check_names)
    n_proxies = len(proxy_cols)
    bar_width = 0.8 / n_proxies

    from proxymate.viz._plot_helpers import BLUE, ORANGE, PROXY_COLOR

    proxy_colors = [PROXY_COLOR, BLUE, ORANGE, GREEN, GRAY]

    if figsize is None:
        figsize = (max(10, n_checks * 1.2), 6)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_checks)

    for i, proxy_col in enumerate(proxy_cols):
        values = [proxy_scores[proxy_col].get(name, 0.0) for name in check_names]
        offset = (i - n_proxies / 2 + 0.5) * bar_width
        color = proxy_colors[i % len(proxy_colors)]
        ax.bar(x + offset, values, bar_width, label=proxy_col, alpha=0.85, color=color)

        # Star on best proxy per check
        for j, name in enumerate(check_names):
            best_val = max(proxy_scores[col].get(name, 0.0) for col in proxy_cols)
            if values[j] >= best_val and values[j] > 0:
                ax.plot(
                    x[j] + offset,
                    values[j] + 0.02,
                    "*",
                    color=color,
                    markersize=10,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                )

    # Sub-property separator lines
    for boundary in group_boundaries[1:]:
        ax.axvline(boundary - 0.5, color=GRAY, linewidth=0.5, linestyle="-", alpha=0.4)

    # X-axis: check names
    ax.set_xticks(x)
    ax.set_xticklabels(check_names, rotation=45, ha="right", fontsize=10)

    # Sub-property group labels below check names
    for center, label in zip(group_centers, group_display):
        ax.text(
            center,
            -0.18,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=GRAY,
        )

    ax.set_ylabel("Projection Score", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title("Multi-Proxy Comparison", fontsize=14)
    ax.legend(fontsize=10)
    clean_axes(ax)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    return fig


def plot_radar(
    report: ValidationReport,
    figsize: tuple[float, float] = (8, 8),
    threshold: float = 0.8,
) -> plt.Figure:
    """Radar chart comparing sub-property pass rates across proxies."""
    if len(report.per_proxy_results) < 1:
        raise ValueError("No proxy results to plot.")

    from proxymate.viz._plot_helpers import BLUE, ORANGE, PROXY_COLOR

    all_props: list[str] = []
    for proxy_result in report.per_proxy_results.values():
        summary = proxy_result.summary()
        for prop_name in summary:
            if prop_name not in all_props:
                all_props.append(prop_name)

    if not all_props:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5, "No sub-property results to display", ha="center", va="center"
        )
        return fig

    n_props = len(all_props)
    angles = np.linspace(0, 2 * np.pi, n_props, endpoint=False).tolist()
    angles.append(angles[0])

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})

    # Threshold circle
    threshold_vals = [threshold] * (n_props + 1)
    ax.plot(
        angles,
        threshold_vals,
        "--",
        color=GRAY,
        linewidth=1,
        alpha=0.6,
        label=f"Threshold ({threshold:.0%})",
    )
    ax.fill(angles, threshold_vals, alpha=0.03, color=GRAY)

    proxy_colors = [PROXY_COLOR, BLUE, ORANGE, GREEN]
    for idx, (proxy_col, proxy_result) in enumerate(report.per_proxy_results.items()):
        summary = proxy_result.summary()
        values = [summary.get(prop, 0.0) for prop in all_props]
        values.append(values[0])
        color = proxy_colors[idx % len(proxy_colors)]
        ax.plot(angles, values, "o-", label=proxy_col, linewidth=2, color=color)
        ax.fill(angles, values, alpha=0.08, color=color)

    # Format labels
    display_labels = [p.replace("_", " ").title() for p in all_props]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(display_labels, size=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Sub-Property Pass Rates", y=1.08, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    fig.tight_layout()
    return fig


def _extract_check_scores(proxy_res: ProxyValidationResult) -> dict[str, float]:
    """Extract check_name → projection_result mapping from a ProxyValidationResult."""
    return {
        r.check_name: r.projection_result for _, r in iter_all_check_results(proxy_res)
    }


def _build_segment_rows(
    report: ValidationReport, proxy_col: str | None
) -> tuple[list[dict[str, float]], list[str]]:
    """Build row data and labels from segment results for heatmap."""
    rows_data: list[dict[str, float]] = []
    row_labels: list[str] = []
    for seg in report.segment_results:
        for col, proxy_res in seg.proxy_results.items():
            if proxy_col is not None and col != proxy_col:
                continue
            label = f"{seg.group_name}/{seg.segment_name}"
            if proxy_col is None and len(seg.proxy_results) > 1:
                label = f"{label} ({col})"
            row_labels.append(label)
            rows_data.append(_extract_check_scores(proxy_res))
    return rows_data, row_labels


def plot_segment_heatmap(
    report: ValidationReport,
    proxy_col: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Heatmap of check projections across segments."""
    if not report.segment_results:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No segment results to display", ha="center", va="center")
        return fig

    rows_data, row_labels = _build_segment_rows(report, proxy_col)

    if not rows_data:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No matching segment results", ha="center", va="center")
        return fig

    # Build matrix
    all_checks = sorted({k for row in rows_data for k in row})
    matrix = np.full((len(rows_data), len(all_checks)), np.nan)
    for i, row in enumerate(rows_data):
        for j, check in enumerate(all_checks):
            if check in row:
                matrix[i, j] = row[check]

    n_rows, n_cols = matrix.shape
    if figsize is None:
        figsize = (max(8, n_cols * 1.2), max(4, n_rows * 0.5 + 1))

    fig, ax = plt.subplots(figsize=figsize)

    # Pastel red-yellow-green colormap
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "pastel_ryg", ["#f4a6a0", "#fde8a0", "#a8dba8"]
    )
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    # Cell text
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                )

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(all_checks, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_title("Segment Check Projections", fontsize=14)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    fig.colorbar(im, ax=ax, label="Projection Score", shrink=0.8)
    fig.tight_layout()
    return fig


# ========================================================================
# Comprehensive plot_all
# ========================================================================


def plot_all(
    report: ValidationReport,
    proxy_col: str | None = None,
) -> list[plt.Figure]:
    """Auto-select and render all relevant plots based on what's in the report.

    Inspects the ``ValidationReport`` to determine which data tiers are
    populated and calls the corresponding ``plot_all_*`` helpers.  Each call
    is wrapped in a try/except so that a failure in one tier does not prevent
    the remaining plots from rendering.

    Args:
        report: A populated ``ValidationReport``.
        proxy_col: Which proxy column to plot.  When ``None`` and only one
            proxy exists, that one is used automatically.

    Returns:
        A list of ``matplotlib.Figure`` objects that were successfully created.
    """
    figures: list[plt.Figure] = []
    plotted: list[str] = []
    skipped: list[str] = []

    # Resolve proxy_col if not specified and only one proxy exists
    if proxy_col is None and len(report.per_proxy_results) == 1:
        proxy_col = next(iter(report.per_proxy_results))

    proxy_result = (
        report.per_proxy_results.get(proxy_col) if proxy_col is not None else None
    )

    # --- Unit-level plots ---
    has_unit_level = proxy_result is not None and bool(proxy_result.unit_level_results)
    if has_unit_level:
        try:
            fig = plot_all_unit_level(report, proxy_col=proxy_col)
            figures.append(fig)
            plotted.append("unit-level")
        except Exception as e:
            skipped.append(f"unit-level (error: {e})")
    else:
        skipped.append("unit-level (no data)")

    # --- Aggregate + bootstrap plots ---
    has_aggregate = proxy_result is not None and bool(
        proxy_result.estimate_level_results or proxy_result.domain_level_results
    )
    has_bootstrap = report.bootstrap_input is not None
    if has_aggregate or has_bootstrap:
        try:
            fig = plot_all_estimate_level(report, proxy_name=proxy_col)
            figures.append(fig)
            plotted.append("estimate-level")
        except Exception as e:
            skipped.append(f"estimate-level (error: {e})")
    else:
        skipped.append("estimate-level (no data)")

    # --- Meta-analysis plots ---
    has_meta = bool(report.meta_analysis_inputs)
    if has_meta:
        try:
            fig = plot_all_meta_analysis(report, proxy_col=proxy_col)
            figures.append(fig)
            plotted.append("meta-analysis")
        except Exception as e:
            skipped.append(f"meta-analysis (error: {e})")
    else:
        skipped.append("meta-analysis (no data)")

    # --- Representativity plots ---
    has_representativity = (
        report.unit_level_input is not None
        and report.unit_level_input.weight_col is not None
    )
    if has_representativity:
        try:
            fig = plot_all_representativity(report, proxy_col=proxy_col)
            figures.append(fig)
            plotted.append("representativity")
        except Exception as e:
            skipped.append(f"representativity (error: {e})")
    else:
        skipped.append("representativity (no data)")

    # --- Summary ---
    print(f"plot_all: rendered {len(figures)} figure(s).")
    if plotted:
        print(f"  Plotted: {', '.join(plotted)}")
    if skipped:
        print(f"  Skipped: {', '.join(skipped)}")

    return figures
