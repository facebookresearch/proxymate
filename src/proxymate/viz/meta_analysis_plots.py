# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from proxymate.checks.domain_level.meta_analysis import _compute_weights, _wls_fit
from proxymate.orchestrator.results import ValidationReport
from proxymate.types.inputs import MetaAnalysisInput
from proxymate.viz._plot_helpers import (
    annotate_check_results,
    clean_axes,
    GRAY,
    GREEN,
    ORANGE,
    PRIMARY_COLOR,
    PROXY_COLOR,
    RED,
)


def _resolve_meta_input(
    report: ValidationReport,
    proxy_col: str | None,
) -> tuple[str, MetaAnalysisInput]:
    """Resolve proxy_col and return (proxy_col, MetaAnalysisInput)."""
    if not report.meta_analysis_inputs:
        raise ValueError("ValidationReport has no meta_analysis_inputs stored.")

    if proxy_col is None:
        proxy_col = next(iter(report.meta_analysis_inputs))

    if proxy_col not in report.meta_analysis_inputs:
        raise ValueError(
            f"Proxy '{proxy_col}' not found in meta_analysis_inputs. "
            f"Available: {list(report.meta_analysis_inputs.keys())}"
        )

    return proxy_col, report.meta_analysis_inputs[proxy_col]


def _extract_arrays(
    meta_input: MetaAnalysisInput,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    """Extract primary/proxy vals, SEs, and domain labels from a MetaAnalysisInput."""
    primary_vals = np.array([p.estimate_val for p, _ in meta_input.domain_estimates])
    proxy_vals = np.array([q.estimate_val for _, q in meta_input.domain_estimates])
    primary_ses = np.array([p.standard_error for p, _ in meta_input.domain_estimates])
    proxy_ses = np.array([q.standard_error for _, q in meta_input.domain_estimates])
    if meta_input.domain_names is not None:
        names = list(meta_input.domain_names)
    else:
        names = [f"Domain {i + 1}" for i in range(len(primary_vals))]
    return primary_vals, proxy_vals, primary_ses, proxy_ses, names


class MetaAnalysisPlots:
    """Diagnostic plots for meta-analysis (D3) checks.

    Methods accept a ``ValidationReport`` with stored ``meta_analysis_inputs``.
    All methods support an optional ``ax`` parameter for subplot embedding.
    """

    @staticmethod
    def scatter(
        report: ValidationReport,
        proxy_col: str | None = None,
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] = (8, 8),
    ) -> plt.Figure:
        """Meta-analysis scatter: proxy vs primary effects across domains.

        Args:
            report: A ValidationReport with meta_analysis_inputs.
            proxy_col: Which proxy's MetaAnalysisInput to plot. If None,
                uses the first available proxy.
            ax: Optional axes for subplot embedding.
            figsize: Figure size (ignored when ax is provided).
        """
        proxy_col, meta_input = _resolve_meta_input(report, proxy_col)
        primary_vals, proxy_vals, primary_ses, proxy_ses, domain_names = (
            _extract_arrays(meta_input)
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        if len(proxy_vals) == 0:
            ax.text(0.5, 0.5, "No domain estimates", ha="center", va="center")
            if standalone:
                fig.tight_layout()
            return fig

        has_errors = bool(np.any(primary_ses > 0) or np.any(proxy_ses > 0))
        if has_errors:
            ax.errorbar(
                proxy_vals,  # pyre-ignore[bad-argument-type]
                primary_vals,  # pyre-ignore[bad-argument-type]
                # pyre-ignore[bad-argument-type]
                xerr=proxy_ses if np.any(proxy_ses > 0) else None,
                # pyre-ignore[bad-argument-type]
                yerr=primary_ses if np.any(primary_ses > 0) else None,
                fmt="o",
                color=PRIMARY_COLOR,
                markersize=8,
                capsize=3,
                ecolor=GRAY,
                elinewidth=1,
                alpha=0.8,
            )
        else:
            ax.scatter(
                proxy_vals,  # pyre-ignore[bad-argument-type]
                primary_vals,  # pyre-ignore[bad-argument-type]
                color=PRIMARY_COLOR,
                s=60,
                alpha=0.8,
                zorder=5,
            )

        # Domain labels
        if meta_input.domain_names is not None:
            for i, name in enumerate(domain_names):
                ax.annotate(
                    name,
                    (proxy_vals[i], primary_vals[i]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=9,
                    color=GRAY,
                )

        # Identity line
        lo = min(proxy_vals.min(), primary_vals.min())
        hi = max(proxy_vals.max(), primary_vals.max())
        margin = (hi - lo) * 0.1
        ax.plot(
            [lo - margin, hi + margin],
            [lo - margin, hi + margin],
            "--",
            color=GRAY,
            linewidth=1,
            label="y = x",
        )

        # Regression line (WLS when SEs available, OLS fallback)
        if len(proxy_vals) >= 2:
            weights = _compute_weights(meta_input)
            fit = _wls_fit(proxy_vals, primary_vals, weights)
            xs = np.linspace(lo, hi, 100)
            ys = fit.intercept + fit.slope * xs
            method = "WLS" if weights is not None else "OLS"
            ax.plot(
                xs,
                ys,
                "-",
                color=PROXY_COLOR,
                linewidth=2,
                label=f"{method} (slope={fit.slope:.3f})",
            )

            fitted = fit.intercept + fit.slope * proxy_vals
            if weights is not None:
                w_mean = float(np.sum(weights * primary_vals) / np.sum(weights))
                ss_res = float(np.sum(weights * (primary_vals - fitted) ** 2))
                ss_tot = float(np.sum(weights * (primary_vals - w_mean) ** 2))
            else:
                ss_res = float(np.sum((primary_vals - fitted) ** 2))
                ss_tot = float(np.sum((primary_vals - np.mean(primary_vals)) ** 2))
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            ax.text(
                0.05,
                0.95,
                f"R² = {r_squared:.4f}\nSlope = {fit.slope:.4f}\nn = {len(proxy_vals)}",
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox={
                    "boxstyle": "round,pad=0.4",
                    "facecolor": "white",
                    "alpha": 0.8,
                },
            )

        ax.set_xlabel("Proxy Effect", fontsize=12)
        ax.set_ylabel("Primary Effect", fontsize=12)
        ax.set_title("Meta-Analysis: Primary vs Proxy Effects", fontsize=14)
        ax.legend(fontsize=10)
        clean_axes(ax)
        if report.per_proxy_results:
            proxy_key = next(iter(report.per_proxy_results))
            proxy_result = report.per_proxy_results[proxy_key]
            annotate_check_results(ax, proxy_result)
        if standalone:
            fig.tight_layout()
        return fig

    @staticmethod
    def forest_plot(
        report: ValidationReport,
        proxy_col: str | None = None,
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] = (10, 8),
    ) -> plt.Figure:
        """Forest plot of primary and proxy estimates with CIs per domain.

        Each domain shows two horizontal lines: one for primary (purple) and
        one for proxy (blue), with point estimates and confidence intervals.
        """
        proxy_col, meta_input = _resolve_meta_input(report, proxy_col)
        primary_vals, proxy_vals, primary_ses, proxy_ses, names = _extract_arrays(
            meta_input
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        n = len(primary_vals)
        if n == 0:
            ax.text(0.5, 0.5, "No domain estimates", ha="center", va="center")
            if standalone:
                fig.tight_layout()
            return fig

        # Compute CIs from stored bounds
        primary_lo = np.array([p.lower_bound for p, _ in meta_input.domain_estimates])
        primary_hi = np.array([p.upper_bound for p, _ in meta_input.domain_estimates])
        proxy_lo = np.array([q.lower_bound for _, q in meta_input.domain_estimates])
        proxy_hi = np.array([q.upper_bound for _, q in meta_input.domain_estimates])

        y_positions = np.arange(n)
        offset = 0.15

        # Primary estimates
        ax.errorbar(
            primary_vals,  # pyre-ignore[bad-argument-type]
            y_positions + offset,
            xerr=[primary_vals - primary_lo, primary_hi - primary_vals],
            fmt="s",
            color=PRIMARY_COLOR,
            markersize=7,
            capsize=3,
            elinewidth=1.5,
            label="Primary",
            alpha=0.9,
        )

        # Proxy estimates
        ax.errorbar(
            proxy_vals,  # pyre-ignore[bad-argument-type]
            y_positions - offset,
            xerr=[proxy_vals - proxy_lo, proxy_hi - proxy_vals],
            fmt="o",
            color=PROXY_COLOR,
            markersize=7,
            capsize=3,
            elinewidth=1.5,
            label="Proxy",
            alpha=0.9,
        )

        # Zero reference line
        ax.axvline(0, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.6)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel("Effect Size", fontsize=12)
        ax.set_title("Forest Plot: Primary vs Proxy by Domain", fontsize=14)
        ax.legend(fontsize=10, loc="lower right")
        ax.invert_yaxis()
        clean_axes(ax)
        if standalone:
            fig.tight_layout()
        return fig

    @staticmethod
    def bland_altman(
        report: ValidationReport,
        proxy_col: str | None = None,
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] = (8, 6),
    ) -> plt.Figure:
        """Bland-Altman plot: difference vs mean of primary and proxy effects.

        Shows systematic bias (mean difference) and limits of agreement
        (mean +/- 1.96 * SD of differences).
        """
        proxy_col, meta_input = _resolve_meta_input(report, proxy_col)
        primary_vals, proxy_vals, _, _, names = _extract_arrays(meta_input)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        n = len(primary_vals)
        if n == 0:
            ax.text(0.5, 0.5, "No domain estimates", ha="center", va="center")
            if standalone:
                fig.tight_layout()
            return fig

        means = (primary_vals + proxy_vals) / 2.0
        diffs = proxy_vals - primary_vals

        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
        upper_loa = mean_diff + 1.96 * std_diff
        lower_loa = mean_diff - 1.96 * std_diff

        ax.scatter(means, diffs, color=PRIMARY_COLOR, s=60, alpha=0.8, zorder=5)

        # Domain labels
        if meta_input.domain_names is not None:
            for i, name in enumerate(names):
                ax.annotate(
                    name,
                    (means[i], diffs[i]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=9,
                    color=GRAY,
                )

        # Mean difference
        ax.axhline(
            mean_diff,
            color=PROXY_COLOR,
            linewidth=2,
            label=f"Mean diff = {mean_diff:.4f}",
        )

        # Limits of agreement
        ax.axhline(
            upper_loa,
            color=RED,
            linestyle="--",
            linewidth=1,
            label=f"+1.96 SD = {upper_loa:.4f}",
        )
        ax.axhline(
            lower_loa,
            color=RED,
            linestyle="--",
            linewidth=1,
            label=f"-1.96 SD = {lower_loa:.4f}",
        )

        # Zero reference
        ax.axhline(0, color=GRAY, linestyle=":", linewidth=0.8, alpha=0.5)

        # Shade limits of agreement
        xlim = ax.get_xlim()
        ax.fill_between(
            list(xlim),
            lower_loa,
            upper_loa,
            alpha=0.08,
            color=PROXY_COLOR,
        )
        ax.set_xlim(xlim)

        ax.set_xlabel("Mean of Primary & Proxy", fontsize=12)
        ax.set_ylabel("Difference (Proxy - Primary)", fontsize=12)
        ax.set_title("Bland-Altman Plot", fontsize=14)
        ax.legend(fontsize=9, loc="best")
        clean_axes(ax)
        if standalone:
            fig.tight_layout()
        return fig

    @staticmethod
    def funnel_plot(
        report: ValidationReport,
        proxy_col: str | None = None,
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] = (8, 6),
    ) -> plt.Figure:
        """Funnel plot: difference (proxy - primary) vs precision (1/SE).

        Useful for detecting heterogeneity across domains.
        Points outside the funnel (95% CI bounds) indicate domains
        where proxy and primary diverge more than expected.
        """
        proxy_col, meta_input = _resolve_meta_input(report, proxy_col)
        primary_vals, proxy_vals, primary_ses, proxy_ses, names = _extract_arrays(
            meta_input
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        n = len(primary_vals)
        if n == 0:
            ax.text(0.5, 0.5, "No domain estimates", ha="center", va="center")
            if standalone:
                fig.tight_layout()
            return fig

        diffs = proxy_vals - primary_vals
        # Combined SE for the difference
        combined_ses = np.sqrt(primary_ses**2 + proxy_ses**2)
        # Use 1/SE as precision; guard against zero SE
        safe_ses = np.where(combined_ses > 0, combined_ses, np.nan)
        precision: np.ndarray = np.where(safe_ses > 0, 1.0 / safe_ses, np.nan)

        mean_diff = float(np.nanmean(diffs))

        # pyre-ignore[bad-argument-type]
        ax.scatter(diffs, precision, color=PRIMARY_COLOR, s=60, alpha=0.8, zorder=5)

        # Domain labels
        if meta_input.domain_names is not None:
            for i, name in enumerate(names):
                if np.isfinite(precision[i]):
                    ax.annotate(
                        name,
                        (diffs[i], precision[i]),
                        textcoords="offset points",
                        xytext=(6, 6),
                        fontsize=9,
                        color=GRAY,
                    )

        # Mean difference line
        ax.axvline(
            mean_diff,
            color=PROXY_COLOR,
            linewidth=2,
            label=f"Mean diff = {mean_diff:.4f}",
        )

        # Funnel bounds (95% CI)
        finite_prec = precision[np.isfinite(precision)]
        if len(finite_prec) > 0:
            y_range = np.linspace(
                max(float(finite_prec.min()) * 0.5, 0.01),
                float(finite_prec.max()) * 1.2,
                200,
            )
            se_range = 1.0 / y_range
            ax.plot(
                mean_diff - 1.96 * se_range,
                y_range,
                "--",
                color=GRAY,
                linewidth=1,
                alpha=0.7,
            )
            ax.plot(
                mean_diff + 1.96 * se_range,
                y_range,
                "--",
                color=GRAY,
                linewidth=1,
                alpha=0.7,
                label="95% bounds",
            )

        ax.set_xlabel("Difference (Proxy - Primary)", fontsize=12)
        ax.set_ylabel("Precision (1 / SE)", fontsize=12)
        ax.set_title("Funnel Plot", fontsize=14)
        ax.legend(fontsize=9, loc="best")
        clean_axes(ax)
        if standalone:
            fig.tight_layout()
        return fig

    @staticmethod
    def ci_width_comparison(
        report: ValidationReport,
        proxy_col: str | None = None,
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] = (10, 6),
    ) -> plt.Figure:
        """Compare CI widths of primary vs proxy across domains.

        A grouped bar chart showing the confidence interval width for
        each domain, for both primary and proxy estimates.
        """
        proxy_col, meta_input = _resolve_meta_input(report, proxy_col)
        _, _, _, _, names = _extract_arrays(meta_input)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        n = len(meta_input.domain_estimates)
        if n == 0:
            ax.text(0.5, 0.5, "No domain estimates", ha="center", va="center")
            if standalone:
                fig.tight_layout()
            return fig

        primary_widths = np.array(
            [p.upper_bound - p.lower_bound for p, _ in meta_input.domain_estimates]
        )
        proxy_widths = np.array(
            [q.upper_bound - q.lower_bound for _, q in meta_input.domain_estimates]
        )

        x = np.arange(n)
        bar_width = 0.35

        ax.bar(
            x - bar_width / 2,
            primary_widths,  # pyre-ignore[bad-argument-type]
            bar_width,
            color=PRIMARY_COLOR,
            alpha=0.8,
            label="Primary CI Width",
        )
        ax.bar(
            x + bar_width / 2,
            proxy_widths,  # pyre-ignore[bad-argument-type]
            bar_width,
            color=PROXY_COLOR,
            alpha=0.8,
            label="Proxy CI Width",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=10)
        ax.set_ylabel("CI Width", fontsize=12)
        ax.set_title("Confidence Interval Width Comparison", fontsize=14)
        ax.legend(fontsize=10)
        clean_axes(ax)
        if standalone:
            fig.tight_layout()
        return fig

    @staticmethod
    def residual_by_domain(
        report: ValidationReport,
        proxy_col: str | None = None,
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] = (10, 6),
    ) -> plt.Figure:
        """Residual plot: (proxy - primary) effect per domain with error bars.

        Domains are ordered by residual magnitude. Error bars show the
        combined standard error of the difference. A zero reference line
        highlights domains with systematic proxy bias.
        """
        proxy_col, meta_input = _resolve_meta_input(report, proxy_col)
        primary_vals, proxy_vals, primary_ses, proxy_ses, names = _extract_arrays(
            meta_input
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        n = len(primary_vals)
        if n == 0:
            ax.text(0.5, 0.5, "No domain estimates", ha="center", va="center")
            if standalone:
                fig.tight_layout()
            return fig

        residuals = proxy_vals - primary_vals
        combined_ses = np.sqrt(primary_ses**2 + proxy_ses**2)

        # Sort by residual for readability
        order = np.argsort(residuals)
        residuals = residuals[order]
        combined_ses = combined_ses[order]
        sorted_names = [names[i] for i in order]

        y_positions = np.arange(n)
        colors = [GREEN if r >= 0 else RED for r in residuals]

        ax.barh(
            y_positions,
            residuals,
            color=colors,
            alpha=0.7,
            height=0.6,
        )

        # Error bars
        has_errors = bool(np.any(combined_ses > 0))
        if has_errors:
            ax.errorbar(
                residuals,
                y_positions,
                xerr=combined_ses * 1.96,
                fmt="none",
                ecolor=GRAY,
                elinewidth=1,
                capsize=3,
            )

        ax.axvline(0, color=GRAY, linestyle="--", linewidth=1)

        mean_resid = float(np.mean(residuals))
        ax.axvline(
            mean_resid,
            color=ORANGE,
            linestyle="-",
            linewidth=1.5,
            alpha=0.7,
            label=f"Mean = {mean_resid:.4f}",
        )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(sorted_names, fontsize=10)
        ax.set_xlabel("Residual (Proxy - Primary)", fontsize=12)
        ax.set_title("Residuals by Domain", fontsize=14)
        ax.legend(fontsize=10, loc="best")
        clean_axes(ax)
        if standalone:
            fig.tight_layout()
        return fig

    @staticmethod
    def cumulative_effect(
        report: ValidationReport,
        proxy_col: str | None = None,
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] = (10, 6),
    ) -> plt.Figure:
        """Cumulative meta-analysis: running inverse-variance weighted average.

        Domains are added one at a time (in given order) and the running
        weighted mean of the difference (proxy - primary) is plotted with
        its 95% CI. Useful for assessing stability of the overall effect
        estimate as more domains are included.
        """
        proxy_col, meta_input = _resolve_meta_input(report, proxy_col)
        primary_vals, proxy_vals, primary_ses, proxy_ses, names = _extract_arrays(
            meta_input
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        n = len(primary_vals)
        if n == 0:
            ax.text(0.5, 0.5, "No domain estimates", ha="center", va="center")
            if standalone:
                fig.tight_layout()
            return fig

        diffs = proxy_vals - primary_vals
        combined_vars = primary_ses**2 + proxy_ses**2
        # Use inverse-variance weights; fall back to equal weights if SE=0
        has_var = bool(np.any(combined_vars > 0))

        cum_means = np.zeros(n)
        cum_lo = np.zeros(n)
        cum_hi = np.zeros(n)

        for k in range(1, n + 1):
            d_k = diffs[:k]
            if has_var:
                v_k = combined_vars[:k]
                # Guard against zero variance
                safe_v = np.where(v_k > 0, v_k, np.inf)
                weights = 1.0 / safe_v
                w_sum = float(np.sum(weights))
                if w_sum > 0:
                    cum_means[k - 1] = float(np.sum(weights * d_k)) / w_sum
                    cum_se = 1.0 / np.sqrt(w_sum)
                else:
                    cum_means[k - 1] = float(np.mean(d_k))
                    cum_se = 0.0
            else:
                cum_means[k - 1] = float(np.mean(d_k))
                cum_se = float(np.std(d_k, ddof=1)) / np.sqrt(k) if k > 1 else 0.0
            cum_lo[k - 1] = cum_means[k - 1] - 1.96 * cum_se
            cum_hi[k - 1] = cum_means[k - 1] + 1.96 * cum_se

        x = np.arange(1, n + 1)
        ax.plot(x, cum_means, "o-", color=PRIMARY_COLOR, markersize=6, linewidth=2)
        ax.fill_between(x, cum_lo, cum_hi, alpha=0.2, color=PRIMARY_COLOR)

        ax.axhline(0, color=GRAY, linestyle="--", linewidth=0.8)
        ax.axhline(
            cum_means[-1],
            color=PROXY_COLOR,
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label=f"Final = {cum_means[-1]:.4f}",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax.set_xlabel("Domains (cumulative)", fontsize=12)
        ax.set_ylabel("Weighted Mean Difference", fontsize=12)
        ax.set_title("Cumulative Meta-Analysis", fontsize=14)
        ax.legend(fontsize=10, loc="best")
        clean_axes(ax)
        if standalone:
            fig.tight_layout()
        return fig

    @staticmethod
    def sample_size_vs_effect(
        report: ValidationReport,
        proxy_col: str | None = None,
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] = (8, 6),
    ) -> plt.Figure:
        """Bubble chart: absolute difference vs domain sample size.

        Bubble size encodes the combined standard error. Useful for
        identifying whether larger domains show more or less agreement
        between primary and proxy.
        """
        proxy_col, meta_input = _resolve_meta_input(report, proxy_col)
        primary_vals, proxy_vals, primary_ses, proxy_ses, names = _extract_arrays(
            meta_input
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
        else:
            fig = cast(plt.Figure, ax.figure)
            standalone = False

        n = len(primary_vals)
        if n == 0:
            ax.text(0.5, 0.5, "No domain estimates", ha="center", va="center")
            if standalone:
                fig.tight_layout()
            return fig

        sample_sizes = np.array([p.sample_size for p, _ in meta_input.domain_estimates])
        abs_diffs = np.abs(proxy_vals - primary_vals)
        combined_ses = np.sqrt(primary_ses**2 + proxy_ses**2)

        # Bubble size proportional to SE (larger SE = larger bubble)
        if np.any(combined_ses > 0):
            sizes = 50 + 300 * (combined_ses / combined_ses.max())
        else:
            sizes = np.full(n, 100.0)

        ax.scatter(
            sample_sizes,  # pyre-ignore[bad-argument-type]
            abs_diffs,
            s=sizes,
            color=PRIMARY_COLOR,
            alpha=0.6,
            edgecolors=GRAY,
            linewidth=0.5,
        )

        # Domain labels
        if meta_input.domain_names is not None:
            for i, name in enumerate(names):
                ax.annotate(
                    name,
                    (sample_sizes[i], abs_diffs[i]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=9,
                    color=GRAY,
                )

        # Trend line
        if n >= 2:
            log_sizes = np.log1p(sample_sizes.astype(float))
            coeffs = np.polyfit(log_sizes, abs_diffs, 1)
            fit_fn = np.poly1d(coeffs)
            xs = np.linspace(log_sizes.min(), log_sizes.max(), 100)
            ax.plot(
                np.expm1(xs),
                fit_fn(xs),
                "--",
                color=PROXY_COLOR,
                linewidth=1.5,
                alpha=0.7,
                label="Trend",
            )
            ax.legend(fontsize=10)

        ax.set_xlabel("Sample Size", fontsize=12)
        ax.set_ylabel("|Proxy - Primary|", fontsize=12)
        ax.set_title("Effect Difference vs Sample Size", fontsize=14)
        clean_axes(ax)
        if standalone:
            fig.tight_layout()
        return fig


def plot_all_meta_analysis(
    report: ValidationReport,
    proxy_col: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot all meta-analysis diagnostics on a single figure.

    Combines 6 diagnostic plots in a 2x3 grid:

    1. **Scatter** -- proxy vs primary effects with OLS regression
    2. **Forest plot** -- per-domain CIs for primary and proxy
    3. **Bland-Altman** -- difference vs mean with limits of agreement
    4. **Funnel plot** -- difference vs precision for heterogeneity
    5. **Residuals by domain** -- sorted residual bar chart
    6. **Cumulative effect** -- running weighted average of differences

    Args:
        report: ValidationReport with meta_analysis_inputs populated.
        proxy_col: Which proxy to plot for. If None, uses the first available.
        figsize: Optional figure size override.

    Returns a single figure with subplots.
    """
    if not report.meta_analysis_inputs:
        raise ValueError(
            "ValidationReport has no meta_analysis_inputs. "
            "Run meta-analysis checks first (via MetaAnalysisInput)."
        )

    if proxy_col is None:
        proxy_col = next(iter(report.meta_analysis_inputs))

    methods = [
        MetaAnalysisPlots.scatter,
        MetaAnalysisPlots.forest_plot,
        MetaAnalysisPlots.bland_altman,
        MetaAnalysisPlots.funnel_plot,
        MetaAnalysisPlots.residual_by_domain,
        MetaAnalysisPlots.cumulative_effect,
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

    fig.suptitle(f"Meta-Analysis Diagnostics: {proxy_col}", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig
