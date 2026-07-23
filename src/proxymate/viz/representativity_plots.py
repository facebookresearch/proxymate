# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from proxymate.checks.representativity_level.representativity import (
    _get_target_proxy_split,
    SampleRepresentativityResult,
    WeightExtremeResult,
    WeightingEffectivenessResult,
)
from proxymate.orchestrator.results import ValidationReport
from proxymate.types.inputs import CheckInput
from proxymate.viz._plot_helpers import (
    clean_axes,
    GREEN,
    ORANGE,
    PRIMARY_COLOR,
    PROXY_COLOR,
    RED,
)


def _build_check_input_from_report(
    report: ValidationReport,
    proxy_col: str | None = None,
) -> CheckInput | None:
    """Build a CheckInput from the report's UnitLevelInput for population splitting."""
    unit_input = report.unit_level_input
    if unit_input is None:
        return None
    if proxy_col is None:
        proxy_col = unit_input.proxy_cols[0] if unit_input.proxy_cols else None
    if proxy_col is None:
        return None
    return CheckInput(
        df=unit_input.df,
        primary_col=unit_input.primary_col,
        proxy_col=proxy_col,
        weight_col=unit_input.weight_col,
        covariate_cols=unit_input.covariate_cols,
        population_col=unit_input.population_col,
    )


def _find_sample_representativity_result(
    report: ValidationReport,
    proxy_col: str | None = None,
) -> SampleRepresentativityResult | None:
    """Extract the SampleRepresentativityResult from the report."""
    if not report.per_proxy_results:
        return None
    if proxy_col is not None:
        proxy_result = report.per_proxy_results.get(proxy_col)
    else:
        proxy_result = next(iter(report.per_proxy_results.values()))
    if proxy_result is None:
        return None
    for check_result in proxy_result.representativity_results:
        if isinstance(check_result.check_result, SampleRepresentativityResult):
            return check_result.check_result
    return None


def _find_weight_extreme_result(
    report: ValidationReport,
    proxy_col: str | None = None,
) -> WeightExtremeResult | None:
    """Extract the WeightExtremeResult from the report."""
    if not report.per_proxy_results:
        return None
    if proxy_col is not None:
        proxy_result = report.per_proxy_results.get(proxy_col)
    else:
        proxy_result = next(iter(report.per_proxy_results.values()))
    if proxy_result is None:
        return None
    for check_result in proxy_result.representativity_results:
        if isinstance(check_result.check_result, WeightExtremeResult):
            return check_result.check_result
    return None


def _find_weighting_effectiveness_result(
    report: ValidationReport,
    proxy_col: str | None = None,
) -> WeightingEffectivenessResult | None:
    """Extract the WeightingEffectivenessResult from the report."""
    if not report.per_proxy_results:
        return None
    if proxy_col is not None:
        proxy_result = report.per_proxy_results.get(proxy_col)
    else:
        proxy_result = next(iter(report.per_proxy_results.values()))
    if proxy_result is None:
        return None
    for check_result in proxy_result.representativity_results:
        if isinstance(check_result.check_result, WeightingEffectivenessResult):
            return check_result.check_result
    return None


class RepresentativityPlots:
    """Diagnostic plots for representativity checks.

    Visualizes how well the proxy/labeled population P represents the
    target population T, based on covariate distributions.
    """

    @staticmethod
    def love_plot(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Covariate balance (Love) plot.

        Shows the Standardized Mean Difference (SMD) for each covariate
        as a horizontal lollipop chart. Reference lines at SMD = 0.1
        (well-balanced) and SMD = 0.25 (concerning) divide the plot
        into green, yellow, and red zones.

        Args:
            report: ValidationReport with representativity results.
            proxy_col: Which proxy to use. If None, uses the first.
            figsize: Optional figure size override.
            ax: Optional axes for subplot embedding.
        """
        result = _find_sample_representativity_result(report, proxy_col)
        if result is None:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = cast(plt.Figure, ax.figure)
            ax.text(
                0.5,
                0.5,
                "No SampleRepresentativityCheck results found.",
                ha="center",
                va="center",
                fontsize=12,
            )
            return fig

        covariate_stats = result.covariate_stats
        if not covariate_stats:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = cast(plt.Figure, ax.figure)
            ax.text(
                0.5,
                0.5,
                "No covariate statistics available.",
                ha="center",
                va="center",
                fontsize=12,
            )
            return fig

        # Sort covariates by SMD descending
        sorted_covs = sorted(
            covariate_stats.items(),
            key=lambda x: x[1]["smd"],
            reverse=True,
        )
        names = [name for name, _ in sorted_covs]
        smds = [s["smd"] for _, s in sorted_covs]
        n = len(names)

        if ax is None:
            if figsize is None:
                figsize = (10, max(4, n * 0.4 + 1))
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        y_positions = np.arange(n)

        # Color each dot by SMD threshold
        colors = []
        for smd in smds:
            if smd < 0.1:
                colors.append(GREEN)
            elif smd < 0.25:
                colors.append(ORANGE)
            else:
                colors.append(RED)

        # Lollipop: horizontal lines from 0 to SMD
        for i, (smd, color) in enumerate(zip(smds, colors)):
            ax.hlines(
                y=i,
                xmin=0,
                xmax=smd,
                color=color,
                linewidth=1.5,
                alpha=0.7,
            )

        # Dots
        ax.scatter(
            smds,
            y_positions,
            c=colors,
            s=80,
            zorder=5,
            edgecolors="white",
            linewidths=0.5,
        )

        # Reference lines
        ax.axvline(
            0.1,
            color=GREEN,
            linewidth=1,
            linestyle="--",
            alpha=0.7,
            label="SMD = 0.1 (balanced)",
        )
        ax.axvline(
            0.25,
            color=RED,
            linewidth=1,
            linestyle="--",
            alpha=0.7,
            label="SMD = 0.25 (imbalanced)",
        )

        # Background shading for zones
        x_max = max(smds) * 1.3 if smds else 0.5
        x_max = max(x_max, 0.35)
        ax.axvspan(0, 0.1, alpha=0.04, color=GREEN)
        ax.axvspan(0.1, 0.25, alpha=0.04, color=ORANGE)
        ax.axvspan(0.25, x_max, alpha=0.04, color=RED)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel("Standardized Mean Difference (SMD)", fontsize=12)
        ax.set_title("Covariate Balance (Love Plot)", fontsize=14)
        ax.set_xlim(0, x_max)
        ax.set_ylim(-0.5, n - 0.5)
        ax.legend(fontsize=10, loc="lower right")
        clean_axes(ax)
        if standalone:
            fig.tight_layout()
        return fig

    @staticmethod
    def distribution_comparison(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> plt.Figure:  # noqa: C901
        """Side-by-side KDE overlays for target vs proxy covariate distributions.

        Shows one subplot per covariate with overlaid kernel density
        estimates for the target and proxy populations. Each subplot is
        annotated with the KS statistic and p-value from the
        SampleRepresentativityCheck.

        Args:
            report: ValidationReport with representativity results.
            proxy_col: Which proxy to use. If None, uses the first.
            figsize: Optional figure size override.
        """
        # Build target/proxy split from UnitLevelInput
        check_input = _build_check_input_from_report(report, proxy_col)
        if check_input is None:
            fig, ax = plt.subplots()
            ax.text(
                0.5,
                0.5,
                "UnitLevelInput required for distribution plots.\n"
                "Provide data_handler or unit_level_input to validate().",
                ha="center",
                va="center",
                fontsize=12,
            )
            return fig

        target_df, proxy_df = _get_target_proxy_split(check_input)

        # Get check results for annotations (optional)
        sr_result = _find_sample_representativity_result(report, proxy_col)
        covariate_stats = sr_result.covariate_stats if sr_result else {}

        # Determine covariates to plot
        if check_input.covariate_cols is not None:
            columns = check_input.covariate_cols
        else:
            target_cols = set(target_df.columns)
            proxy_cols_set = set(proxy_df.columns)
            exclude = {check_input.primary_col, check_input.proxy_col}
            if check_input.weight_col is not None:
                exclude.add(check_input.weight_col)
            if check_input.population_col is not None:
                exclude.add(check_input.population_col)
            columns = sorted((target_cols & proxy_cols_set) - exclude)

        # Filter to numeric columns
        numeric_cols: list[str] = []
        for col in columns:
            if pd.api.types.is_numeric_dtype(
                target_df[col]
            ) and pd.api.types.is_numeric_dtype(proxy_df[col]):
                numeric_cols.append(col)

        if not numeric_cols:
            fig, ax = plt.subplots()
            ax.text(
                0.5,
                0.5,
                "No numeric covariate columns to plot.",
                ha="center",
                va="center",
                fontsize=12,
            )
            return fig

        n = len(numeric_cols)
        ncols = min(3, n)
        nrows = math.ceil(n / ncols)

        if figsize is None:
            figsize = (5 * ncols, 4 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes_flat: list[plt.Axes] = list(np.array(axes).flatten())

        for i, col in enumerate(numeric_cols):
            ax = axes_flat[i]

            target_vals = target_df[col].dropna().astype(float).values
            proxy_vals = proxy_df[col].dropna().astype(float).values

            # Winsorize at P1/P99 for visualization stability
            if len(target_vals) > 0 and len(proxy_vals) > 0:
                all_vals = np.concatenate([target_vals, proxy_vals])
                p1, p99 = (
                    float(np.percentile(all_vals, 1)),
                    float(np.percentile(all_vals, 99)),
                )
                if p99 > p1:
                    target_vals = np.clip(target_vals, p1, p99)
                    proxy_vals = np.clip(proxy_vals, p1, p99)

            # KDE plots
            if len(target_vals) > 1:
                ax.hist(
                    target_vals,
                    bins=30,
                    density=True,
                    alpha=0.3,
                    color=PRIMARY_COLOR,
                    label="Target",
                )
                _plot_kde(ax, target_vals, PRIMARY_COLOR)
            if len(proxy_vals) > 1:
                ax.hist(
                    proxy_vals,
                    bins=30,
                    density=True,
                    alpha=0.3,
                    color=PROXY_COLOR,
                    label="Proxy",
                )
                _plot_kde(ax, proxy_vals, PROXY_COLOR)

            # Annotate with KS stat if available
            stats = covariate_stats.get(col)
            if stats:
                ks = stats["ks_statistic"]
                pval = stats["ks_p_value"]
                smd = stats["smd"]
                annotation = f"KS={ks:.3f} (p={pval:.3f})\nSMD={smd:.3f}"
                ax.text(
                    0.97,
                    0.95,
                    annotation,
                    transform=ax.transAxes,
                    fontsize=9,
                    va="top",
                    ha="right",
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": "white",
                        "alpha": 0.8,
                    },
                )

            ax.set_title(col, fontsize=14)
            ax.set_ylabel("Density", fontsize=12)
            if i == 0:
                ax.legend(fontsize=10)
            clean_axes(ax)

        # Hide unused axes
        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(
            "Covariate Distribution: Target vs Proxy",
            fontsize=14,
            y=1.01,
        )
        fig.tight_layout()
        return fig

    @staticmethod
    def weight_distribution(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Histogram and KDE of IW weights with percentile markers.

        Shows the distribution of weights from the UnitLevelInput,
        with vertical lines at the 1st, 5th, 95th, and 99th percentiles.
        Annotates with summary statistics from WeightExtremeCheck (CV,
        weight ratio, min/max) if available.

        Args:
            report: ValidationReport with representativity results.
            proxy_col: Which proxy to use. If None, uses the first.
            figsize: Optional figure size override.
            ax: Optional axes for subplot embedding.
        """
        # Get weights from UnitLevelInput
        unit_input = report.unit_level_input
        weight_col = unit_input.weight_col if unit_input is not None else None

        if unit_input is None or weight_col is None:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = cast(plt.Figure, ax.figure)
            ax.text(
                0.5,
                0.5,
                "No weights available.\nProvide UnitLevelInput with weight_col.",
                ha="center",
                va="center",
                fontsize=12,
            )
            return fig

        weights = unit_input.df[weight_col].dropna().astype(float).values

        if ax is None:
            if figsize is None:
                figsize = (10, 6)
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        # Winsorize extreme weights for visualization (clip at P99)
        p99 = float(np.percentile(weights, 99))
        weights_clipped = np.clip(weights, a_min=None, a_max=p99)

        # Histogram + KDE on winsorized weights
        ax.hist(
            weights_clipped,
            bins=50,
            density=True,
            alpha=0.3,
            color=PROXY_COLOR,
            edgecolor="white",
            linewidth=0.5,
        )
        _plot_kde(ax, weights_clipped, PROXY_COLOR)

        # Percentile markers
        percentiles = [1, 5, 50, 95, 99]
        pct_values = np.percentile(weights, percentiles)
        pct_colors = [RED, ORANGE, GREEN, ORANGE, RED]
        pct_styles = ["--", "--", "-", "--", "--"]

        for pct, val, color, style in zip(
            percentiles, pct_values, pct_colors, pct_styles
        ):
            ax.axvline(
                val,
                color=color,
                linewidth=1.2,
                linestyle=style,
                alpha=0.8,
                label=f"P{pct} = {val:.3f}",
            )

        # Annotate with WeightExtremeCheck stats if available
        we_result = _find_weight_extreme_result(report, proxy_col)
        if we_result is not None:
            annotation = (
                f"CV = {we_result.cv:.3f}\n"
                f"Ratio = {we_result.weight_ratio:.1f}\n"
                f"Min = {we_result.min_weight:.4f}\n"
                f"Max = {we_result.max_weight:.4f}\n"
                f"N = {we_result.n_weights}"
            )
            ax.text(
                0.97,
                0.95,
                annotation,
                transform=ax.transAxes,
                fontsize=9,
                va="top",
                ha="right",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "alpha": 0.8,
                },
            )

        ax.set_xlabel("Weight", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("Weight Distribution", fontsize=14)
        ax.legend(fontsize=10, loc="upper left")
        clean_axes(ax)
        if standalone:
            fig.tight_layout()
        return fig

    @staticmethod
    def gap_reduction_plot(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Gap reduction plot for weighting effectiveness (R.5).

        Shows the unweighted vs weighted gap for the primary outcome and
        each covariate as a grouped horizontal bar chart. Each row shows
        two bars: the absolute gap before weighting (unweighted) and after
        weighting (weighted). Annotates with the gap reduction percentage.

        Args:
            report: ValidationReport with representativity results.
            proxy_col: Which proxy to use. If None, uses the first.
            figsize: Optional figure size override.
            ax: Optional axes for subplot embedding.
        """
        result = _find_weighting_effectiveness_result(report, proxy_col)
        if result is None:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = cast(plt.Figure, ax.figure)
            ax.text(
                0.5,
                0.5,
                "No WeightingEffectivenessCheck results found.",
                ha="center",
                va="center",
                fontsize=12,
            )
            return fig

        # Build data: primary outcome + per-covariate
        labels = ["Primary Outcome"]
        gaps_unweighted = [abs(result.gap_unweighted)]
        gaps_weighted = [abs(result.gap_weighted)]
        reductions = [result.gap_reduction]

        for cov_name, stats in result.per_covariate_gap_reduction.items():
            labels.append(cov_name)
            target_mean = stats["target_mean"]
            unwt_mean = stats["proxy_mean_unweighted"]
            wt_mean = stats["proxy_mean_weighted"]
            gaps_unweighted.append(abs(unwt_mean - target_mean))
            gaps_weighted.append(abs(wt_mean - target_mean))
            reductions.append(stats["gap_reduction"])

        n = len(labels)

        if ax is None:
            if figsize is None:
                figsize = (10, max(4, n * 0.6 + 1))
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        y_positions = np.arange(n)
        bar_height = 0.35

        # Normalize gaps to [0, 1] range for visual comparability
        max_gap = max(max(gaps_unweighted), max(gaps_weighted))
        if max_gap > 0:
            norm_unweighted = [g / max_gap for g in gaps_unweighted]
            norm_weighted = [g / max_gap for g in gaps_weighted]
        else:
            norm_unweighted = gaps_unweighted
            norm_weighted = gaps_weighted

        # Bars
        ax.barh(
            y_positions + bar_height / 2,
            norm_unweighted,
            bar_height,
            label="Unweighted gap",
            color=RED,
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.barh(
            y_positions - bar_height / 2,
            norm_weighted,
            bar_height,
            label="Weighted gap",
            color=GREEN,
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
        )

        # Annotate with gap reduction percentage
        for i, (nu, nw, red) in enumerate(
            zip(norm_unweighted, norm_weighted, reductions)
        ):
            x_pos = max(nu, nw) + 0.02
            color = GREEN if red >= 0.5 else (ORANGE if red >= 0 else RED)
            ax.text(
                x_pos,
                i,
                f"{red:.0%}",
                va="center",
                ha="left",
                fontsize=10,
                fontweight="bold",
                color=color,
            )

        # Reference line at 50% threshold
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Normalized gap (relative to max)", fontsize=12)
        ax.set_title(
            "Weighting Effectiveness: Gap Reduction (R.5)",
            fontsize=14,
        )
        ax.set_xlim(0, max(max(norm_unweighted), max(norm_weighted)) * 1.25)
        ax.set_ylim(-0.5, n - 0.5)
        ax.invert_yaxis()
        ax.legend(fontsize=10, loc="lower right")
        clean_axes(ax)
        if standalone:
            fig.tight_layout()
        return fig


def plot_all_representativity(
    report: ValidationReport,
    proxy_col: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:  # noqa: C901
    """Combined representativity diagnostics in a single figure.

    Layout:
      - Top row(s): covariate distribution subplots + weight distribution
        (side by side)
      - Gap reduction row: unweighted vs weighted gaps for primary outcome
        and each covariate (R.5), if WeightingEffectivenessCheck was run
      - Bottom row: love plot spanning the full width

    If no distribution data, weights, or gap reduction results are available,
    falls back to love plot only.

    Args:
        report: ValidationReport with representativity results.
        proxy_col: Which proxy to use. If None, uses the first.
        figsize: Optional figure size override.
    """
    from matplotlib.gridspec import GridSpec

    # Build target/proxy split from UnitLevelInput
    check_input = _build_check_input_from_report(report, proxy_col)

    # Determine numeric covariates for distribution plots
    numeric_cols: list[str] = []
    target_df: pd.DataFrame | None = None
    proxy_df: pd.DataFrame | None = None

    if check_input is not None:
        target_df, proxy_df = _get_target_proxy_split(check_input)
        if check_input.covariate_cols is not None:
            columns = check_input.covariate_cols
        else:
            target_cols = set(target_df.columns)
            proxy_cols_set = set(proxy_df.columns)
            exclude = {check_input.primary_col, check_input.proxy_col}
            if check_input.weight_col is not None:
                exclude.add(check_input.weight_col)
            if check_input.population_col is not None:
                exclude.add(check_input.population_col)
            columns = sorted((target_cols & proxy_cols_set) - exclude)
        for col in columns:
            if pd.api.types.is_numeric_dtype(
                target_df[col]
            ) and pd.api.types.is_numeric_dtype(proxy_df[col]):
                numeric_cols.append(col)

    has_distributions = len(numeric_cols) > 0 and target_df is not None
    unit_input = report.unit_level_input
    has_weights = unit_input is not None and unit_input.weight_col is not None

    has_gap_reduction = (
        _find_weighting_effectiveness_result(report, proxy_col) is not None
    )

    if not has_distributions and not has_weights and not has_gap_reduction:
        # Love plot only
        return RepresentativityPlots.love_plot(
            report, proxy_col=proxy_col, figsize=figsize
        )

    # Top row cells: distribution subplots + weight distribution (if present)
    top_cells = list(numeric_cols)
    if has_weights:
        top_cells.append("__weights__")

    n_top = len(top_cells)
    top_ncols = min(3, n_top) if n_top > 0 else 3
    top_nrows = math.ceil(n_top / top_ncols) if n_top > 0 else 0

    # Total rows: top rows + gap reduction (if present) + love plot
    bottom_rows = 0
    if has_gap_reduction:
        bottom_rows += 1
    bottom_rows += 1  # love plot always present
    nrows = top_nrows + bottom_rows

    if figsize is None:
        figsize = (6 * top_ncols, 4 * nrows)

    fig = plt.figure(figsize=figsize)
    height_ratios = [1] * top_nrows + [1.2] * bottom_rows
    gs = GridSpec(
        nrows,
        top_ncols,
        hspace=0.45,
        wspace=0.3,
        height_ratios=height_ratios,
        figure=fig,
    )

    # --- Top rows: distribution subplots + weight distribution ---
    sr_result = _find_sample_representativity_result(report, proxy_col)
    covariate_stats = sr_result.covariate_stats if sr_result else {}

    for i, cell in enumerate(top_cells):
        row = i // top_ncols
        col_idx = i % top_ncols
        ax = fig.add_subplot(gs[row, col_idx])

        if cell == "__weights__":
            RepresentativityPlots.weight_distribution(
                report, proxy_col=proxy_col, ax=ax
            )
        elif target_df is not None and proxy_df is not None:
            target_vals = target_df[cell].dropna().astype(float).values
            proxy_vals = proxy_df[cell].dropna().astype(float).values

            if len(target_vals) > 1:
                ax.hist(
                    target_vals,
                    bins=30,
                    density=True,
                    alpha=0.3,
                    color=PRIMARY_COLOR,
                    label="Target",
                )
                _plot_kde(ax, target_vals, PRIMARY_COLOR)
            if len(proxy_vals) > 1:
                ax.hist(
                    proxy_vals,
                    bins=30,
                    density=True,
                    alpha=0.3,
                    color=PROXY_COLOR,
                    label="Proxy",
                )
                _plot_kde(ax, proxy_vals, PROXY_COLOR)

            # Annotate with KS stat if available
            stats = covariate_stats.get(cell)
            if stats:
                ks = stats["ks_statistic"]
                pval = stats["ks_p_value"]
                smd = stats["smd"]
                annotation = f"KS={ks:.3f} (p={pval:.3f})\nSMD={smd:.3f}"
                ax.text(
                    0.97,
                    0.95,
                    annotation,
                    transform=ax.transAxes,
                    fontsize=9,
                    va="top",
                    ha="right",
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": "white",
                        "alpha": 0.8,
                    },
                )

            ax.set_title(cell, fontsize=14)
            ax.set_ylabel("Density", fontsize=12)
            if i == 0:
                ax.legend(fontsize=10)
            clean_axes(ax)

    # Hide unused top-row cells
    for j in range(n_top, top_nrows * top_ncols):
        row = j // top_ncols
        col_idx = j % top_ncols
        ax_empty = fig.add_subplot(gs[row, col_idx])
        ax_empty.set_visible(False)

    # --- Bottom rows: gap reduction (if present) + love plot ---
    current_row = top_nrows
    if has_gap_reduction:
        ax_gap = fig.add_subplot(gs[current_row, :])
        RepresentativityPlots.gap_reduction_plot(report, proxy_col=proxy_col, ax=ax_gap)
        current_row += 1

    ax_love = fig.add_subplot(gs[current_row, :])
    RepresentativityPlots.love_plot(report, proxy_col=proxy_col, ax=ax_love)

    fig.suptitle("Representativity Diagnostics", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


def _plot_kde(
    ax: plt.Axes,
    values: np.ndarray,
    color: str,
) -> None:
    """Plot a KDE curve on the given axes."""
    from scipy.stats import gaussian_kde

    try:
        clean = values[np.isfinite(values)]
        if len(clean) < 2 or float(np.std(clean)) == 0:
            return
        kde = gaussian_kde(clean)
        x_range = np.linspace(float(np.min(clean)), float(np.max(clean)), 200)
        kde_vals = kde(x_range)
        if np.all(np.isfinite(kde_vals)):
            ax.plot(x_range, kde_vals, color=color, linewidth=2)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        pass
