# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from proxymate.orchestrator.results import ValidationReport
from proxymate.types.enums import OutcomeType, UnitLevelProperty
from proxymate.types.inputs import UnitLevelInput
from proxymate.utils.stats_utils import is_binary, is_probability
from proxymate.viz._plot_helpers import (
    annotate_check_results,
    clean_axes,
    get_paired_arrays,
    GRAY,
    GREEN,
    ORANGE,
    PRIMARY_COLOR,
    PROXY_COLOR,
    RED,
    resolve_proxy_col,
    resolve_proxy_result,
)


def _check_and_resolve(
    report: ValidationReport,
    proxy_col: str | None,
) -> tuple[UnitLevelInput, str, np.ndarray, np.ndarray]:
    """Validate report and resolve proxy/primary arrays."""
    if report.unit_level_input is None:
        raise ValueError("ValidationReport has no unit_level_input stored.")
    unit_input = report.unit_level_input
    resolved_col = resolve_proxy_col(unit_input, proxy_col)
    primary, proxy = get_paired_arrays(unit_input, resolved_col)
    return unit_input, resolved_col, primary, proxy


def _annotate_and_finalize(
    ax: plt.Axes,
    report: ValidationReport,
    proxy_col: str,
    sub_properties: list[str],
    standalone: bool,
    fig: plt.Figure,
) -> plt.Figure:
    """Annotate plot with check results and apply tight_layout if standalone."""
    proxy_result = resolve_proxy_result(report, proxy_col)
    if proxy_result is not None:
        annotate_check_results(ax, proxy_result, sub_properties=sub_properties)
    if standalone:
        fig.tight_layout()
    return fig


class UnitLevelPlots:
    """Diagnostic plots for unit-level (D1) paired data.

    All methods accept a ``ValidationReport`` with a stored
    ``unit_level_input`` and annotate with relevant check results.

    Pass an ``ax`` argument to draw on an existing axes (e.g. for subplots)
    instead of creating a new figure.
    """

    @staticmethod
    def scatter(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] = (8, 8),
        alpha: float = 0.3,
        max_points: int = 5000,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Scatter plot of primary vs proxy values with identity line and R²."""
        unit_input, proxy_col, primary, proxy = _check_and_resolve(report, proxy_col)

        # Subsample for performance
        if len(primary) > max_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(primary), max_points, replace=False)
            primary, proxy = primary[idx], proxy[idx]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        ax.scatter(
            primary,  # pyre-ignore[bad-argument-type]
            proxy,  # pyre-ignore[bad-argument-type]
            alpha=alpha,
            s=10,
            color=PRIMARY_COLOR,
            edgecolors="none",
        )

        # Identity line
        lo = min(primary.min(), proxy.min())
        hi = max(primary.max(), proxy.max())
        margin = (hi - lo) * 0.05
        ax.plot(
            [lo - margin, hi + margin],
            [lo - margin, hi + margin],
            "--",
            color=RED,
            linewidth=1.5,
            label="y = x",
        )

        # OLS fit
        if len(primary) >= 2:
            coeffs = np.polyfit(primary, proxy, 1)
            fit_line = np.poly1d(coeffs)
            xs = np.linspace(lo, hi, 100)
            ax.plot(
                xs, fit_line(xs), "-", color=PROXY_COLOR, linewidth=2, label="OLS fit"
            )

            ss_res = float(np.sum((proxy - fit_line(primary)) ** 2))
            ss_tot = float(np.sum((proxy - np.mean(proxy)) ** 2))
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            ax.text(
                0.05,
                0.95,
                f"R² = {r_squared:.4f}\nn = {len(primary):,}",
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox={
                    "boxstyle": "round,pad=0.4",
                    "facecolor": "white",
                    "alpha": 0.8,
                },
            )

        ax.set_xlabel(unit_input.primary_col, fontsize=12)
        ax.set_ylabel(proxy_col, fontsize=12)
        ax.set_title(f"Primary vs Proxy: {proxy_col}", fontsize=14)
        ax.legend(fontsize=10)
        clean_axes(ax)
        return _annotate_and_finalize(
            ax,
            report,
            proxy_col,
            [UnitLevelProperty.ACCURACY.value, UnitLevelProperty.PRECISION.value],
            standalone,
            fig,
        )

    @staticmethod
    def residuals(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] = (10, 6),
        alpha: float = 0.3,
        max_points: int = 5000,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Bland-Altman residual plot: mean vs difference."""
        _, proxy_col, primary, proxy = _check_and_resolve(report, proxy_col)

        if len(primary) > max_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(primary), max_points, replace=False)
            primary, proxy = primary[idx], proxy[idx]

        means = (primary + proxy) / 2
        diffs = proxy - primary
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1))
        upper_loa = mean_diff + 1.96 * std_diff
        lower_loa = mean_diff - 1.96 * std_diff

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        ax.scatter(
            means, diffs, alpha=alpha, s=10, color=PRIMARY_COLOR, edgecolors="none"
        )

        # Zero line
        ax.axhline(0, color=GRAY, linewidth=1)

        # Mean bias
        ax.axhline(
            mean_diff,
            color=PROXY_COLOR,
            linewidth=1.5,
            linestyle="-",
            label=f"Mean bias: {mean_diff:.4f}",
        )

        # Limits of agreement
        ax.axhline(
            upper_loa,
            color=RED,
            linewidth=1,
            linestyle="--",
            label=f"+1.96 SD: {upper_loa:.4f}",
        )
        ax.axhline(
            lower_loa,
            color=RED,
            linewidth=1,
            linestyle="--",
            label=f"-1.96 SD: {lower_loa:.4f}",
        )
        ax.axhspan(lower_loa, upper_loa, alpha=0.06, color=RED)

        ax.set_xlabel("Mean of Primary and Proxy", fontsize=12)
        ax.set_ylabel("Proxy - Primary", fontsize=12)
        ax.set_title(f"Bland-Altman Plot: {proxy_col}", fontsize=14)
        ax.legend(fontsize=9, loc="upper right")
        clean_axes(ax)
        return _annotate_and_finalize(
            ax,
            report,
            proxy_col,
            [UnitLevelProperty.AGREEMENT.value],
            standalone,
            fig,
        )

    @staticmethod
    def distributions(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] = (10, 6),
        bins: int = 50,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Overlaid histograms of primary vs proxy distributions."""
        unit_input, proxy_col, primary, proxy = _check_and_resolve(report, proxy_col)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        ax.hist(
            primary,  # pyre-ignore[bad-argument-type]
            bins=bins,
            density=True,
            alpha=0.5,
            color=PRIMARY_COLOR,
            label=unit_input.primary_col,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.hist(
            proxy,  # pyre-ignore[bad-argument-type]
            bins=bins,
            density=True,
            alpha=0.5,
            color=PROXY_COLOR,
            label=proxy_col,
            edgecolor="white",
            linewidth=0.5,
        )

        # Mean lines
        ax.axvline(
            float(np.mean(primary)), color=PRIMARY_COLOR, linewidth=1.5, linestyle="--"
        )
        ax.axvline(
            float(np.mean(proxy)), color=PROXY_COLOR, linewidth=1.5, linestyle="--"
        )

        ax.set_xlabel("Value", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"Distribution Comparison: {proxy_col}", fontsize=14)
        ax.legend(fontsize=10)
        clean_axes(ax)
        return _annotate_and_finalize(
            ax,
            report,
            proxy_col,
            [UnitLevelProperty.DISTRIBUTIONAL_FIDELITY.value],
            standalone,
            fig,
        )

    @staticmethod
    def calibration(
        report: ValidationReport,
        proxy_col: str | None = None,
        n_bins: int = 20,
        figsize: tuple[float, float] = (8, 8),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Calibration curve for binary primary / probability proxy outcomes.

        When called standalone (no ``ax``), includes a histogram of predicted
        probabilities below the calibration curve.  When embedded in a subplot
        grid, only the calibration curve is drawn.
        """
        _, proxy_col, primary, proxy = _check_and_resolve(report, proxy_col)

        bins = np.linspace(0, 1, n_bins + 1)
        bin_means = (bins[:-1] + bins[1:]) / 2
        digitized = np.digitize(proxy, bins, right=False)

        fractions: list[float] = []
        counts: list[int] = []
        for i in range(1, n_bins + 1):
            in_bin = digitized == i
            if np.any(in_bin):
                fractions.append(float(np.mean(primary[in_bin])))
                counts.append(int(np.sum(in_bin)))
            else:
                fractions.append(float("nan"))
                counts.append(0)

        if ax is None:
            fig, (ax_cal, ax_hist) = plt.subplots(
                2,
                1,
                figsize=figsize,
                gridspec_kw={"height_ratios": [3, 1]},
                sharex=True,
            )
            standalone = True
        else:
            ax_cal = ax
            ax_hist = None
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        # Calibration curve
        ax_cal.plot(
            [0, 1],
            [0, 1],
            "--",
            color=GRAY,
            linewidth=1.5,
            label="Perfect calibration",
        )
        ax_cal.plot(
            bin_means,
            fractions,
            "o-",
            color=PROXY_COLOR,
            markersize=6,
            linewidth=2,
            label="Observed",
        )
        ax_cal.set_ylabel("Fraction of Positives", fontsize=12)
        ax_cal.set_title(f"Calibration Curve: {proxy_col}", fontsize=14)
        ax_cal.legend(fontsize=10)
        ax_cal.set_ylim(-0.05, 1.05)
        clean_axes(ax_cal)

        # Histogram of predictions (standalone only)
        if ax_hist is not None:
            ax_hist.bar(
                bin_means,
                counts,
                width=1.0 / n_bins * 0.9,
                color=PRIMARY_COLOR,
                alpha=0.7,
            )
            ax_hist.set_xlabel("Predicted Probability", fontsize=12)
            ax_hist.set_ylabel("Count", fontsize=12)
            ax_hist.set_xlim(-0.02, 1.02)
            clean_axes(ax_hist)
        else:
            ax_cal.set_xlabel("Predicted Probability", fontsize=12)
            ax_cal.set_xlim(-0.02, 1.02)

        return _annotate_and_finalize(
            ax_cal,
            report,
            proxy_col,
            [UnitLevelProperty.CALIBRATION.value],
            standalone,
            fig,
        )

    @staticmethod
    def roc_auc(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] = (8, 8),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """ROC curve for binary primary / probability proxy outcomes."""
        from sklearn.metrics import roc_auc_score, roc_curve

        _, proxy_col, primary, proxy = _check_and_resolve(report, proxy_col)

        # Compute ROC curve using sklearn (O(N log N))
        fpr_sorted, tpr_sorted, _ = roc_curve(primary, proxy)
        auc_val = float(roc_auc_score(primary, proxy))

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        ax.plot(
            fpr_sorted,
            tpr_sorted,
            color=PROXY_COLOR,
            linewidth=2,
            label=f"ROC (AUC = {auc_val:.4f})",
        )
        ax.plot([0, 1], [0, 1], "--", color=GRAY, linewidth=1.5, label="Random")
        ax.fill_between(fpr_sorted, tpr_sorted, alpha=0.15, color=PROXY_COLOR)

        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"ROC Curve: {proxy_col}", fontsize=14)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=10, loc="lower right")
        clean_axes(ax)
        return _annotate_and_finalize(
            ax,
            report,
            proxy_col,
            [UnitLevelProperty.DISCRIMINATION.value],
            standalone,
            fig,
        )

    @staticmethod
    def precision_recall(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] = (8, 8),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Precision-Recall curve for binary primary / probability proxy outcomes."""
        from sklearn.metrics import average_precision_score, precision_recall_curve

        _, proxy_col, primary, proxy = _check_and_resolve(report, proxy_col)

        prec_sorted, rec_sorted, _ = precision_recall_curve(primary, proxy)
        auc_pr = float(average_precision_score(primary, proxy))

        # Baseline: prevalence
        n_pos = float(np.sum(primary == 1))
        baseline = n_pos / len(primary) if len(primary) > 0 else 0.0

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        ax.plot(
            rec_sorted,
            prec_sorted,
            color=PROXY_COLOR,
            linewidth=2,
            label=f"PR (AUC = {auc_pr:.4f})",
        )
        ax.axhline(baseline, color=GRAY, linewidth=1.5, linestyle="--", label="Random")
        ax.fill_between(rec_sorted, prec_sorted, alpha=0.15, color=PROXY_COLOR)

        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(f"Precision-Recall Curve: {proxy_col}", fontsize=14)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=10, loc="lower left")
        clean_axes(ax)
        return _annotate_and_finalize(
            ax,
            report,
            proxy_col,
            [UnitLevelProperty.DISCRIMINATION.value],
            standalone,
            fig,
        )

    @staticmethod
    def score_distribution(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] = (10, 6),
        bins: int = 50,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Proxy score distributions split by primary class (0 vs 1).

        Shows how well the proxy probability separates positive and negative
        primary outcomes.
        """
        unit_input, proxy_col, primary, proxy = _check_and_resolve(report, proxy_col)

        proxy_neg = proxy[primary == 0]
        proxy_pos = proxy[primary == 1]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        ax.hist(
            proxy_neg,
            bins=bins,
            density=True,
            alpha=0.5,
            color=GRAY,
            label=f"{unit_input.primary_col} = 0 (n={len(proxy_neg):,})",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.hist(
            proxy_pos,
            bins=bins,
            density=True,
            alpha=0.5,
            color=PRIMARY_COLOR,
            label=f"{unit_input.primary_col} = 1 (n={len(proxy_pos):,})",
            edgecolor="white",
            linewidth=0.5,
        )

        # Mean lines
        ax.axvline(
            float(np.mean(proxy_neg)),
            color=GRAY,
            linewidth=1.5,
            linestyle="--",
        )
        ax.axvline(
            float(np.mean(proxy_pos)),
            color=PRIMARY_COLOR,
            linewidth=1.5,
            linestyle="--",
        )

        ax.set_xlabel(f"Proxy Score ({proxy_col})", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"Score Distribution by Class: {proxy_col}", fontsize=14)
        ax.legend(fontsize=10)
        clean_axes(ax)
        return _annotate_and_finalize(
            ax,
            report,
            proxy_col,
            [UnitLevelProperty.DISCRIMINATION.value],
            standalone,
            fig,
        )

    @staticmethod
    def lift(
        report: ValidationReport,
        proxy_col: str | None = None,
        n_bins: int = 10,
        figsize: tuple[float, float] = (10, 6),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Lift chart showing cumulative lift across deciles.

        Each decile is sorted by proxy score descending. Lift = (positive rate
        in decile) / (overall positive rate).
        """
        _, proxy_col, primary, proxy = _check_and_resolve(report, proxy_col)

        # Sort by proxy score descending
        order = np.argsort(-proxy)
        primary_sorted = primary[order]
        baseline_rate = float(np.mean(primary)) if len(primary) > 0 else 0.0

        # Clamp n_bins to available data to avoid empty slices
        n_bins = min(n_bins, len(primary_sorted))
        if n_bins == 0:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = cast(plt.Figure, ax.figure)
            ax.set_title("Lift Chart (no data)")
            return fig

        bin_size = len(primary_sorted) // n_bins
        decile_labels: list[str] = []
        lifts: list[float] = []
        cumulative_lifts: list[float] = []

        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size if i < n_bins - 1 else len(primary_sorted)
            decile_rate = float(np.mean(primary_sorted[start:end]))
            lift_val = decile_rate / baseline_rate if baseline_rate > 0 else 0.0
            lifts.append(lift_val)
            decile_labels.append(f"{i + 1}")

            cum_rate = float(np.mean(primary_sorted[:end]))
            cum_lift = cum_rate / baseline_rate if baseline_rate > 0 else 0.0
            cumulative_lifts.append(cum_lift)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        x_pos = np.arange(n_bins)
        ax.bar(x_pos, lifts, color=PROXY_COLOR, alpha=0.7, label="Decile lift")
        ax.plot(
            x_pos,
            cumulative_lifts,
            "o-",
            color=PRIMARY_COLOR,
            linewidth=2,
            markersize=6,
            label="Cumulative lift",
        )
        ax.axhline(1.0, color=GRAY, linewidth=1.5, linestyle="--", label="Random")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(decile_labels, fontsize=10)
        ax.set_xlabel("Decile (highest proxy score → lowest)", fontsize=12)
        ax.set_ylabel("Lift", fontsize=12)
        ax.set_title(f"Lift Chart: {proxy_col}", fontsize=14)
        ax.legend(fontsize=10)
        clean_axes(ax)
        return _annotate_and_finalize(
            ax,
            report,
            proxy_col,
            [UnitLevelProperty.DISCRIMINATION.value],
            standalone,
            fig,
        )

    @staticmethod
    def confusion_matrix(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] = (8, 7),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Confusion matrix heatmap for binary primary / binary proxy outcomes."""
        _, proxy_col, primary, proxy = _check_and_resolve(report, proxy_col)

        tp = float(np.sum((primary == 1) & (proxy == 1)))
        tn = float(np.sum((primary == 0) & (proxy == 0)))
        fp = float(np.sum((primary == 0) & (proxy == 1)))
        fn = float(np.sum((primary == 1) & (proxy == 0)))
        total = tp + tn + fp + fn

        cm = np.array([[tn, fp], [fn, tp]])

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        # pyre-ignore[bad-argument-type]
        im = ax.imshow(cm, cmap="Blues", aspect="equal")

        # Annotate cells with name, count, and percentage
        names = [
            ["True Negative", "False Positive"],
            ["False Negative", "True Positive"],
        ]
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = val / total if total > 0 else 0.0
                text_color = "white" if val > total / 4 else "black"
                ax.text(
                    j,
                    i,
                    f"{names[i][j]}\n{val:,.0f}\n({pct:.1%})",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=text_color,
                    fontweight="bold",
                )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Proxy Negative", "Proxy Positive"], fontsize=10)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Primary Negative", "Primary Positive"], fontsize=10)
        ax.set_title(f"Confusion Matrix: {proxy_col}", fontsize=14)
        fig.colorbar(im, ax=ax, shrink=0.7)

        return _annotate_and_finalize(
            ax,
            report,
            proxy_col,
            [UnitLevelProperty.AGREEMENT.value],
            standalone,
            fig,
        )

    @staticmethod
    def classification_metrics(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] = (8, 6),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Horizontal bar chart of classification metrics for binary outcomes.

        Shows accuracy, precision, sensitivity (recall), specificity, and F1
        score derived from the confusion matrix.
        """
        _, proxy_col, primary, proxy = _check_and_resolve(report, proxy_col)

        tp = float(np.sum((primary == 1) & (proxy == 1)))
        tn = float(np.sum((primary == 0) & (proxy == 0)))
        fp = float(np.sum((primary == 0) & (proxy == 1)))
        fn = float(np.sum((primary == 1) & (proxy == 0)))
        total = tp + tn + fp + fn

        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (
            2 * precision * sensitivity / (precision + sensitivity)
            if (precision + sensitivity) > 0
            else 0.0
        )

        metric_names = ["Accuracy", "Precision", "Sensitivity", "Specificity", "F1"]
        values = [accuracy, precision, sensitivity, specificity, f1]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        from proxymate.viz._plot_helpers import GREEN as _GREEN

        x_pos = np.arange(len(metric_names))
        colors = [_GREEN if v >= 0.7 else RED for v in values]
        bars = ax.bar(x_pos, values, color=colors, alpha=0.85, width=0.6)

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center",
                fontsize=11,
                fontweight="bold",
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(metric_names, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(f"Classification Metrics: {proxy_col}", fontsize=14)
        clean_axes(ax)
        return _annotate_and_finalize(
            ax,
            report,
            proxy_col,
            [UnitLevelProperty.ACCURACY.value],
            standalone,
            fig,
        )

    @staticmethod
    def cohens_kappa(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] = (8, 5),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Cohen's Kappa agreement chart for binary outcomes.

        Shows observed agreement, expected (chance) agreement, and Kappa
        as a grouped bar chart.
        """
        _, proxy_col, primary, proxy = _check_and_resolve(report, proxy_col)

        tp = float(np.sum((primary == 1) & (proxy == 1)))
        tn = float(np.sum((primary == 0) & (proxy == 0)))
        fp = float(np.sum((primary == 0) & (proxy == 1)))
        fn = float(np.sum((primary == 1) & (proxy == 0)))
        total = tp + tn + fp + fn

        if total == 0:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = cast(plt.Figure, ax.figure)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return fig

        p_observed = (tp + tn) / total
        p_yes = ((tp + fn) / total) * ((tp + fp) / total)
        p_no = ((tn + fp) / total) * ((tn + fn) / total)
        p_expected = p_yes + p_no
        kappa = (
            (p_observed - p_expected) / (1 - p_expected) if p_expected < 1.0 else 0.0
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        labels = ["Expected\n(chance)", "Observed", "Kappa (\u03ba)"]
        values = [p_expected, p_observed, kappa]
        colors = [GRAY, PRIMARY_COLOR, PROXY_COLOR if kappa >= 0.6 else RED]

        bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center",
                fontsize=12,
                fontweight="bold",
            )

        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Agreement", fontsize=12)
        ax.set_title(f"Cohen's Kappa: {proxy_col}", fontsize=14)
        clean_axes(ax)
        return _annotate_and_finalize(
            ax,
            report,
            proxy_col,
            [UnitLevelProperty.AGREEMENT.value],
            standalone,
            fig,
        )

    @staticmethod
    def prevalence(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] = (8, 5),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Bar chart comparing positive rates in primary vs proxy."""
        unit_input, proxy_col, primary, proxy = _check_and_resolve(report, proxy_col)

        primary_pos_rate = float(np.mean(primary)) if len(primary) > 0 else 0.0
        proxy_pos_rate = float(np.mean(proxy)) if len(proxy) > 0 else 0.0

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        labels = [f"Primary\n({unit_input.primary_col})", f"Proxy\n({proxy_col})"]
        values = [primary_pos_rate, proxy_pos_rate]
        colors = [PRIMARY_COLOR, PROXY_COLOR]

        bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.4)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}\n({int(val * len(primary)):,}/{len(primary):,})",
                ha="center",
                fontsize=11,
                fontweight="bold",
            )

        ax.set_ylim(0, max(max(values) * 1.3, 0.1))
        ax.set_ylabel("Positive Rate", fontsize=12)
        ax.set_title(f"Prevalence Comparison: {proxy_col}", fontsize=14)
        clean_axes(ax)
        return _annotate_and_finalize(
            ax,
            report,
            proxy_col,
            [UnitLevelProperty.DISTRIBUTIONAL_FIDELITY.value],
            standalone,
            fig,
        )

    @staticmethod
    def ks(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] = (10, 6),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Kolmogorov-Smirnov plot: empirical CDFs with maximum distance."""
        unit_input, proxy_col, primary, proxy = _check_and_resolve(report, proxy_col)

        if len(primary) == 0 or len(proxy) == 0:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = cast(plt.Figure, ax.figure)
            ax.set_title("KS Plot (no data)")
            return fig

        # Compute empirical CDFs
        all_vals = np.sort(np.unique(np.concatenate([primary, proxy])))
        cdf_primary = np.searchsorted(np.sort(primary), all_vals, side="right") / len(
            primary
        )
        cdf_proxy = np.searchsorted(np.sort(proxy), all_vals, side="right") / len(proxy)

        # KS statistic and location
        diffs = np.abs(cdf_primary - cdf_proxy)
        ks_idx = int(np.argmax(diffs))
        ks_stat = float(diffs[ks_idx])
        ks_x = float(all_vals[ks_idx])

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        ax.step(
            all_vals,
            cdf_primary,
            where="post",
            color=PRIMARY_COLOR,
            linewidth=2,
            label=unit_input.primary_col,
        )
        ax.step(
            all_vals,
            cdf_proxy,
            where="post",
            color=PROXY_COLOR,
            linewidth=2,
            label=proxy_col,
        )

        # KS distance line
        ax.plot(
            [ks_x, ks_x],
            [float(cdf_primary[ks_idx]), float(cdf_proxy[ks_idx])],
            color=RED,
            linewidth=2,
            linestyle="-",
            label=f"KS = {ks_stat:.4f}",
        )
        ax.plot(
            ks_x,
            float(cdf_primary[ks_idx]),
            "o",
            color=RED,
            markersize=6,
        )
        ax.plot(
            ks_x,
            float(cdf_proxy[ks_idx]),
            "o",
            color=RED,
            markersize=6,
        )

        ax.set_xlabel("Value", fontsize=12)
        ax.set_ylabel("Cumulative Probability", fontsize=12)
        ax.set_title(f"KS Plot: {proxy_col}", fontsize=14)
        ax.legend(fontsize=10)
        clean_axes(ax)
        return _annotate_and_finalize(
            ax,
            report,
            proxy_col,
            [UnitLevelProperty.DISTRIBUTIONAL_FIDELITY.value],
            standalone,
            fig,
        )

    @staticmethod
    def qq_plot(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] = (8, 8),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """QQ plot comparing primary vs proxy quantiles."""
        _, proxy_col, primary, proxy = _check_and_resolve(report, proxy_col)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        n = min(len(primary), len(proxy))
        quantiles = np.linspace(0, 100, min(n, 200))
        primary_q = np.percentile(primary, quantiles)
        proxy_q = np.percentile(proxy, quantiles)

        ax.scatter(primary_q, proxy_q, s=15, color=PRIMARY_COLOR, alpha=0.6)

        # Identity line
        lo = min(primary_q.min(), proxy_q.min())
        hi = max(primary_q.max(), proxy_q.max())
        margin = (hi - lo) * 0.05
        ax.plot(
            [lo - margin, hi + margin],
            [lo - margin, hi + margin],
            "--",
            color=RED,
            linewidth=1.5,
            label="y = x",
        )

        ax.set_xlabel("Primary Quantiles", fontsize=12)
        ax.set_ylabel("Proxy Quantiles", fontsize=12)
        ax.set_title(f"QQ Plot: {proxy_col}", fontsize=14)
        ax.legend(fontsize=10)
        clean_axes(ax)
        return _annotate_and_finalize(
            ax,
            report,
            proxy_col,
            [UnitLevelProperty.DISTRIBUTIONAL_FIDELITY.value],
            standalone,
            fig,
        )

    @staticmethod
    def rank_concordance(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] = (8, 8),
        alpha: float = 0.3,
        max_points: int = 5000,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Rank concordance plot: primary ranks vs proxy ranks with Spearman rho."""
        from scipy.stats import rankdata, spearmanr

        _, proxy_col, primary, proxy = _check_and_resolve(report, proxy_col)

        # Compute statistics on full data before subsampling
        rho, pval = spearmanr(primary, proxy)
        n_total = len(primary)

        # Subsample for visualization only
        if len(primary) > max_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(primary), max_points, replace=False)
            primary, proxy = primary[idx], proxy[idx]

        # Compute ranks (average method for ties)
        primary_ranks = rankdata(primary)
        proxy_ranks = rankdata(proxy)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        ax.scatter(
            primary_ranks,  # pyre-ignore[bad-argument-type]
            proxy_ranks,  # pyre-ignore[bad-argument-type]
            alpha=alpha,
            s=10,
            color=PRIMARY_COLOR,
            edgecolors="none",
        )

        # Identity line
        lo = 0
        hi = max(primary_ranks.max(), proxy_ranks.max())
        ax.plot(
            [lo, hi],
            [lo, hi],
            "--",
            color=RED,
            linewidth=1.5,
            label="y = x",
        )

        ax.text(
            0.05,
            0.95,
            f"Spearman \u03c1 = {rho:.4f}\np = {pval:.2e}\nn = {n_total:,}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "alpha": 0.8,
            },
        )

        ax.set_xlabel("Primary Rank", fontsize=12)
        ax.set_ylabel("Proxy Rank", fontsize=12)
        ax.set_title(f"Rank Concordance: {proxy_col}", fontsize=14)
        ax.legend(fontsize=10)
        clean_axes(ax)
        return _annotate_and_finalize(
            ax,
            report,
            proxy_col,
            [UnitLevelProperty.DISCRIMINATION.value],
            standalone,
            fig,
        )

    @staticmethod
    def mcc_gauge(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] = (8, 5),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Semicircular gauge (half-donut) showing MCC on [-1, 1] scale.

        Color-coded: red (<0), yellow (0-0.4), green (>=0.4).
        Annotated with MCC value and interpretation label.
        """
        from matplotlib.patches import Wedge
        from sklearn.metrics import matthews_corrcoef

        _, proxy_col, primary, proxy = _check_and_resolve(report, proxy_col)

        mcc = float(matthews_corrcoef(primary, proxy))

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        # Build semicircular gauge: three colored segments covering [-1, 1]
        # mapped to 180 degrees (pi radians)
        center = (0.5, 0.0)
        outer_r = 0.45
        inner_r = 0.25

        # Segments: [-1, 0) red, [0, 0.4) yellow/orange, [0.4, 1] green
        segments = [
            (-1.0, 0.0, RED),
            (0.0, 0.4, ORANGE),
            (0.4, 1.0, GREEN),
        ]
        for seg_lo, seg_hi, color in segments:
            # Map [-1,1] -> [180, 0] degrees (left to right)
            theta1 = 180.0 - (seg_hi + 1.0) / 2.0 * 180.0
            theta2 = 180.0 - (seg_lo + 1.0) / 2.0 * 180.0
            wedge_outer = Wedge(center, outer_r, theta1, theta2, color=color, alpha=0.3)
            wedge_inner = Wedge(center, inner_r, theta1, theta2, color="white")
            ax.add_patch(wedge_outer)
            ax.add_patch(wedge_inner)

        # Needle: map MCC to angle
        mcc_clamped = min(1.0, max(-1.0, mcc))
        needle_angle = 180.0 - (mcc_clamped + 1.0) / 2.0 * 180.0
        needle_rad = np.deg2rad(needle_angle)
        needle_len = outer_r - 0.02
        nx = center[0] + needle_len * np.cos(needle_rad)
        ny = center[1] + needle_len * np.sin(needle_rad)
        ax.plot(
            [center[0], nx],
            [center[1], ny],
            color="black",
            linewidth=2.5,
            solid_capstyle="round",
        )
        ax.plot(center[0], center[1], "o", color="black", markersize=6)

        # Interpretation label
        if mcc < 0:
            label = "Poor"
        elif mcc < 0.2:
            label = "Slight"
        elif mcc < 0.4:
            label = "Fair"
        elif mcc < 0.6:
            label = "Moderate"
        elif mcc < 0.8:
            label = "Strong"
        else:
            label = "Very Strong"

        # MCC value and label
        ax.text(
            0.5,
            0.15,
            f"{mcc:+.3f}",
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.05,
            label,
            ha="center",
            va="center",
            fontsize=13,
            color=GRAY,
            transform=ax.transAxes,
        )

        # Scale labels
        ax.text(0.05, 0.0, "-1", ha="center", fontsize=9, color=GRAY)
        ax.text(0.95, 0.0, "+1", ha="center", fontsize=9, color=GRAY)
        ax.text(0.5, outer_r + 0.02, "0", ha="center", fontsize=9, color=GRAY)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.15, outer_r + 0.08)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"MCC Gauge: {proxy_col}", fontsize=14)

        return _annotate_and_finalize(
            ax,
            report,
            proxy_col,
            [UnitLevelProperty.AGREEMENT.value],
            standalone,
            fig,
        )

    @staticmethod
    def ppv_npv(
        report: ValidationReport,
        proxy_col: str | None = None,
        figsize: tuple[float, float] = (8, 6),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Horizontal bar chart of PPV, NPV, and their baselines.

        Shows PPV (precision) and NPV alongside prevalence-based baselines
        to indicate how much better the proxy is compared to random guessing.
        """
        _, proxy_col, primary, proxy = _check_and_resolve(report, proxy_col)

        tp = float(np.sum((primary == 1) & (proxy == 1)))
        tn = float(np.sum((primary == 0) & (proxy == 0)))
        fp = float(np.sum((primary == 0) & (proxy == 1)))
        fn = float(np.sum((primary == 1) & (proxy == 0)))

        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        prevalence = (tp + fn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        ppv_baseline = prevalence
        npv_baseline = 1.0 - prevalence

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        labels = ["PPV (Precision)", "PPV Baseline", "NPV", "NPV Baseline"]
        values = [ppv, ppv_baseline, npv, npv_baseline]
        colors = [
            GREEN if ppv >= 0.7 else RED,
            GRAY,
            GREEN if npv >= 0.7 else RED,
            GRAY,
        ]

        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, values, color=colors, alpha=0.85, height=0.55)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontsize=11,
                fontweight="bold",
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Score", fontsize=12)
        ax.set_title(f"PPV / NPV: {proxy_col}", fontsize=14)
        ax.invert_yaxis()
        clean_axes(ax)
        return _annotate_and_finalize(
            ax,
            report,
            proxy_col,
            [UnitLevelProperty.DISCRIMINATION.value],
            standalone,
            fig,
        )


def plot_all_unit_level(
    report: ValidationReport,
    proxy_col: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot all relevant unit-level diagnostics on a single figure.

    Uses ``report.outcome_type`` when available, otherwise infers from the data.

    - **Continuous**: scatter, residuals, distributions, ks, qq_plot, rank_concordance
    - **Binary / Probability**: roc_auc, precision_recall, calibration,
      score_distribution, lift, ks
    - **Binary / Binary**: confusion_matrix, classification_metrics,
      cohens_kappa, prevalence, mcc_gauge, ppv_npv

    Returns a single figure with subplots.
    """
    if report.unit_level_input is None:
        raise ValueError("ValidationReport has no unit_level_input stored.")
    input = report.unit_level_input
    proxy_col = resolve_proxy_col(input, proxy_col)

    # Determine outcome type: explicit or inferred
    ot = report.outcome_type
    if ot is None:
        primary, proxy = get_paired_arrays(input, proxy_col)
        primary_binary = is_binary(primary)
        proxy_binary = is_binary(proxy)
        proxy_prob = is_probability(proxy) and not proxy_binary
        if primary_binary and proxy_prob:
            ot = OutcomeType.BINARY_PROBABILITY
        elif primary_binary and proxy_binary:
            ot = OutcomeType.BINARY_BINARY
        else:
            ot = OutcomeType.CONTINUOUS

    # Select plot methods for the outcome type
    if ot == OutcomeType.BINARY_PROBABILITY:
        methods = [
            UnitLevelPlots.roc_auc,
            UnitLevelPlots.precision_recall,
            UnitLevelPlots.calibration,
            UnitLevelPlots.lift,
            UnitLevelPlots.distributions,
            UnitLevelPlots.score_distribution,
        ]
    elif ot == OutcomeType.BINARY_BINARY:
        methods = [
            UnitLevelPlots.confusion_matrix,
            UnitLevelPlots.classification_metrics,
            UnitLevelPlots.cohens_kappa,
            UnitLevelPlots.prevalence,
            UnitLevelPlots.mcc_gauge,
            UnitLevelPlots.ppv_npv,
        ]
    else:
        methods = [
            UnitLevelPlots.scatter,
            UnitLevelPlots.residuals,
            UnitLevelPlots.distributions,
            UnitLevelPlots.ks,
            UnitLevelPlots.qq_plot,
            UnitLevelPlots.rank_concordance,
        ]

    n = len(methods)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    if figsize is None:
        figsize = (7 * ncols, 5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat: list[plt.Axes] = list(np.array(axes).flatten())

    for i, method in enumerate(methods):
        method(report, proxy_col=proxy_col, ax=axes_flat[i])

    # Hide unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Unit-Level Diagnostics: {proxy_col}", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig
