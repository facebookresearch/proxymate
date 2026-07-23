# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

import html as html_mod

from IPython.display import HTML
from proxymate.orchestrator.results import ProxyValidationResult, ValidationReport
from proxymate.viz._plot_helpers import iter_all_check_results

_TABLE_STYLE = (
    "border-collapse:collapse; font-family:system-ui,-apple-system,sans-serif; "
    "font-size:13px; width:70%;"
)
_HEADER_STYLE = (
    "background:#343a40; color:#fff; padding:8px 12px; "
    "text-align:left; font-weight:600;"
)
_CELL_STYLE = "padding:6px 12px; border-bottom:1px solid #dee2e6;"
_PASS_BG = "#d4edda"
_FAIL_BG = "#f8d7da"
_PASS_SYMBOL = "PASS"
_FAIL_SYMBOL = "FAIL"
_DESC_STYLE = "font-size:11px; color:#666; font-style:italic; margin-top:2px;"


# (metric_label, metric_attr on check_result)
_CHECK_METRICS: dict[str, tuple[str, str]] = {
    # Unit-level: Precision
    "RMSE": ("RMSE", "rmse"),
    "R Squared": ("R\u00b2", "r_squared"),
    "Pearson Correlation": ("Pearson r", "correlation"),
    # Unit-level: Accuracy
    "T-Test": ("p-value", "p_value"),
    "MAPE": ("MAPE", "mape"),
    # Unit-level: Agreement
    "Lin's CCC": ("CCC", "ccc"),
    "Bland-Altman": ("% within LoA", "percent_within_loa"),
    "Directional Concordance": ("Concordance", "concordance_rate"),
    # Unit-level: Discrimination
    "Spearman Correlation": ("Spearman \u03c1", "correlation"),
    "Kendall's Tau": ("Kendall \u03c4", "tau"),
    "ROC AUC": ("ROC AUC", "roc_auc"),
    "Precision-Recall AUC": ("PR AUC", "pr_auc"),
    "Precision": ("Precision", "precision"),
    "Accuracy": ("Accuracy", "accuracy"),
    "Sensitivity": ("Sensitivity", "sensitivity"),
    "Specificity": ("Specificity", "specificity"),
    "Confusion Matrix": ("TP", "true_positives"),
    # Unit-level: Calibration
    "Calibration Ratio": ("Cal. Ratio", "calibration_ratio"),
    "Kuiper Calibration": ("Kuiper", "ecce_kuiper"),
    # Unit-level: Distributional Fidelity
    "TV Distance": ("TV Distance", "tv_distance"),
    "KS Distance": ("KS Statistic", "ks_statistic"),
    # Aggregate-level: Single Domain
    "Aggregate Bias": ("Rel. Error", "relative_absolute_error"),
    "Inferential Validity": ("Coverage", "coverage_rate"),
    "Unbiasedness": ("Norm. Bias", "normalized_bias"),
    "CI Overlap": ("Overlap", "overlap_coefficient"),
    "Relative Precision": ("SE Ratio", "se_ratio"),
    "Sign Agreement": ("Agree", "signs_agree"),
    "Effect Size Agreement": ("Score", "agreement_score"),
    # Aggregate-level: Meta-Analysis
    "Aggregate Association": ("R\u00b2", "r_squared"),
    "Effect Sensitivity": ("Slope", "slope"),
    "Directional Alignment": ("Concordance", "concordance_rate"),
    "Prediction Accuracy": ("LOO MAE", "loo_mae"),
    "Heterogeneity": ("I\u00b2", "i_squared"),
    # Representativity
    "Coverage": ("Coverage", "coverage_rate"),
    "Sample Representativity": ("Max SMD", "max_smd"),
    "Effective Sample Size": ("Efficiency", "efficiency"),
    "Weight Extremity": ("Weight Ratio", "weight_ratio"),
    "Weighting Effectiveness": ("Gap Reduction", "gap_reduction"),
}


def _rate_color(rate: float) -> str:
    if rate >= 0.8:
        return _PASS_BG
    if rate >= 0.5:
        return "#fff3cd"
    return _FAIL_BG


_SUB_PROPERTY_DESC: dict[str, str] = {
    "accuracy": "Is the proxy unbiased at the unit level? E[\u0176 - Y] = 0?",
    "precision": "How tightly does the proxy track the primary? Var(\u0176 - Y) small?",
    "agreement": "Does the proxy agree with the primary? CCC = \u03c1 \u00b7 C_b",
    "calibration": (
        "Are predicted probabilities well-calibrated? E[Y | \u0176 = q] = g(q)?"
    ),
    "discrimination": "Does the proxy preserve the relative ordering of units?",
    "distributional_fidelity": (
        "Does the proxy distribution match the primary? F_\u0176 \u2248 F_Y?"
    ),
    "aggregate_bias": (
        "Is the aggregate proxy estimate close to the primary? "
        "\u03b8_\u0176 \u2248 \u03b8_Y?"
    ),
    "inferential_validity": "Do proxy confidence intervals cover the primary estimate?",
    "unbiasedness": "Is the bootstrap proxy distribution centered on the primary?",
    "ci_coverage": "Do bootstrapped proxy CIs capture the primary estimate?",
    "ci_overlap": "Do primary and proxy confidence intervals overlap?",
    "relative_precision": "Is the proxy as precise as the primary?",
    "sign_agreement": "Do primary and proxy estimates have the same sign?",
    "effect_size_agreement": "Are estimates within each other's confidence intervals?",
    "aggregate_association": "Are proxy and primary effects correlated across domains?",
    "effect_sensitivity": (
        "Does a unit change in proxy translate to a unit change in primary?"
    ),
    "directional_alignment": (
        "Do proxy and primary effects agree in sign across domains?"
    ),
    "prediction_accuracy": "Can the proxy predict the primary effect out-of-sample?",
    "heterogeneity": "Is the proxy\u2013primary relationship stable across domains?",
    "representativity": "Does the proxy population represent the target population?",
}


def _format_metric(value: float) -> str:
    """Format a metric value for display."""
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    return f"{value:.4f}"


def display_results_table(
    proxy_result: ProxyValidationResult,
    title: str = "",
) -> HTML:
    """Styled HTML table showing all check results for a single proxy."""
    check_rows = iter_all_check_results(proxy_result)
    if not check_rows:
        return HTML(
            f"<p><em>No results to display{f' for {title}' if title else ''}.</em></p>"
        )

    rows: list[str] = []
    for sub_prop, result in check_rows:
        passed = result.thresholding_result
        bg = _PASS_BG if passed else _FAIL_BG
        result_label = _PASS_SYMBOL if passed else _FAIL_SYMBOL
        result_color = "#155724" if passed else "#721c24"

        metrics = _CHECK_METRICS.get(result.check_name)
        metric_label = metrics[0] if metrics else ""
        metric_attr = metrics[1] if metrics else ""

        # Extract raw metric from check_result
        metric_val = (
            getattr(result.check_result, metric_attr, None) if metric_attr else None
        )
        if metric_val is not None and isinstance(metric_val, (int, float)):
            metric_display = f"{metric_label} = {_format_metric(metric_val)}"
        else:
            metric_display = ""

        # Check name with description below
        check_name_escaped = html_mod.escape(result.check_name)
        name_html = f"<strong>{check_name_escaped}</strong>"
        if result.description:
            desc_escaped = html_mod.escape(result.description)
            name_html += f"<div style='{_DESC_STYLE}'>{desc_escaped}</div>"

        # Format sub-property for display
        sub_prop_display = sub_prop.replace("_", " ").title()

        rows.append(
            f"<tr style='background:{bg};'>"
            f"<td style='{_CELL_STYLE}'>{sub_prop_display}</td>"
            f"<td style='{_CELL_STYLE}'>{name_html}</td>"
            f"<td style='{_CELL_STYLE} text-align:right;'>{metric_display}</td>"
            f"<td style='{_CELL_STYLE} text-align:right;'>"
            f"{result.projection_result:.4f}</td>"
            f"<td style='{_CELL_STYLE} text-align:right;'>"
            f"{result.threshold:.4f}</td>"
            f"<td style='{_CELL_STYLE} color:{result_color}; font-weight:600;'>"
            f"{result_label}</td>"
            f"</tr>"
        )

    title_html = (
        f"<h3 style='margin:16px 0 8px;'>{html_mod.escape(title)}</h3>" if title else ""
    )
    return HTML(
        f"{title_html}"
        f"<table style='{_TABLE_STYLE}'>"
        f"<thead><tr>"
        f"<th style='{_HEADER_STYLE}'>Sub-Property</th>"
        f"<th style='{_HEADER_STYLE}'>Check</th>"
        f"<th style='{_HEADER_STYLE} text-align:right;'>Metric</th>"
        f"<th style='{_HEADER_STYLE} text-align:right;'>Score</th>"
        f"<th style='{_HEADER_STYLE} text-align:right;'>Threshold</th>"
        f"<th style='{_HEADER_STYLE}'>Result</th>"
        f"</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        f"</table>"
    )


def display_summary_table(
    proxy_result: ProxyValidationResult,
    title: str = "",
) -> HTML:
    """HTML table of sub-property pass rates."""
    summary = proxy_result.summary()
    if not summary:
        return HTML(
            f"<p><em>No summary data{f' for {title}' if title else ''}.</em></p>"
        )

    # Compute counts from the underlying results
    counts: dict[str, tuple[int, int]] = {}
    for prop, results in proxy_result.unit_level_results.items():
        passed = sum(1 for r in results if r.thresholding_result)
        counts[prop.value] = (passed, len(results))
    for prop, results in proxy_result.estimate_level_results.items():
        passed = sum(1 for r in results if r.thresholding_result)
        counts[prop.value] = (passed, len(results))
    for prop, results in proxy_result.domain_level_results.items():
        passed = sum(1 for r in results if r.thresholding_result)
        counts[prop.value] = (passed, len(results))
    if proxy_result.representativity_results:
        passed = sum(
            1 for r in proxy_result.representativity_results if r.thresholding_result
        )
        counts["representativity"] = (
            passed,
            len(proxy_result.representativity_results),
        )

    rows: list[str] = []
    total_passed = 0
    total_checks = 0
    for prop_name, rate in summary.items():
        n_passed, n_total = counts.get(prop_name, (0, 0))
        total_passed += n_passed
        total_checks += n_total
        bg = _rate_color(rate)
        display_name = html_mod.escape(prop_name.replace("_", " ").title())
        desc = _SUB_PROPERTY_DESC.get(prop_name, "")
        name_html = f"<strong>{display_name}</strong>"
        if desc:
            name_html += f"<div style='{_DESC_STYLE}'>{html_mod.escape(desc)}</div>"
        rows.append(
            f"<tr>"
            f"<td style='{_CELL_STYLE}'>{name_html}</td>"
            f"<td style='{_CELL_STYLE} text-align:right; background:{bg};'>"
            f"{rate:.0%}</td>"
            f"<td style='{_CELL_STYLE} text-align:right;'>{n_passed}</td>"
            f"<td style='{_CELL_STYLE} text-align:right;'>{n_total}</td>"
            f"</tr>"
        )

    # Footer: overall
    overall_rate = total_passed / total_checks if total_checks > 0 else 0.0
    overall_bg = _rate_color(overall_rate)
    footer = (
        f"<tr style='font-weight:600; border-top:2px solid #343a40;'>"
        f"<td style='{_CELL_STYLE}'>Overall</td>"
        f"<td style='{_CELL_STYLE} text-align:right; background:{overall_bg};'>"
        f"{overall_rate:.0%}</td>"
        f"<td style='{_CELL_STYLE} text-align:right;'>{total_passed}</td>"
        f"<td style='{_CELL_STYLE} text-align:right;'>{total_checks}</td>"
        f"</tr>"
    )

    title_html = (
        f"<h3 style='margin:16px 0 8px;'>{html_mod.escape(title)}</h3>" if title else ""
    )
    return HTML(
        f"{title_html}"
        f"<table style='{_TABLE_STYLE}'>"
        f"<thead><tr>"
        f"<th style='{_HEADER_STYLE}'>Sub-Property</th>"
        f"<th style='{_HEADER_STYLE} text-align:right;'>Pass Rate</th>"
        f"<th style='{_HEADER_STYLE} text-align:right;'>Passed</th>"
        f"<th style='{_HEADER_STYLE} text-align:right;'>Total</th>"
        f"</tr></thead>"
        f"<tbody>{''.join(rows)}{footer}</tbody>"
        f"</table>"
    )


def display_comparison_table(report: ValidationReport) -> HTML:
    """Side-by-side comparison table across proxies."""
    if report.comparison is None:
        return HTML(
            "<p><em>No comparison available (need at least 2 proxies).</em></p>"
        )

    ranking = report.comparison.ranking
    best = report.comparison.best_proxy_per_check
    check_names = sorted(ranking.keys())
    proxy_cols = sorted({col for scores in ranking.values() for col in scores})

    # Build description lookup from any proxy's results
    desc_lookup: dict[str, str] = {}
    for proxy_result in report.per_proxy_results.values():
        for _, r in iter_all_check_results(proxy_result):
            if r.check_name not in desc_lookup:
                desc_lookup[r.check_name] = r.description

    # Header
    proxy_headers = "".join(
        f"<th style='{_HEADER_STYLE} text-align:right;'>{html_mod.escape(col)}</th>"
        for col in proxy_cols
    )

    rows: list[str] = []
    for check_name in check_names:
        scores = ranking[check_name]
        best_proxy = best.get(check_name, "")
        cells: list[str] = []
        for col in proxy_cols:
            val = scores.get(col, 0.0)
            is_best = col == best_proxy
            style = f"{_CELL_STYLE} text-align:right;"
            if is_best:
                style += f" font-weight:700; background:{_PASS_BG};"
            cells.append(f"<td style='{style}'>{val:.4f}</td>")

        desc = desc_lookup.get(check_name, "")
        name_html = f"<strong>{html_mod.escape(check_name)}</strong>"
        if desc:
            name_html += f"<div style='{_DESC_STYLE}'>{html_mod.escape(desc)}</div>"

        rows.append(
            f"<tr><td style='{_CELL_STYLE}'>{name_html}</td>{''.join(cells)}</tr>"
        )

    return HTML(
        f"<h3 style='margin:16px 0 8px;'>Multi-Proxy Comparison</h3>"
        f"<table style='{_TABLE_STYLE}'>"
        f"<thead><tr>"
        f"<th style='{_HEADER_STYLE}'>Check</th>"
        f"{proxy_headers}"
        f"</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        f"</table>"
    )


def display_segment_table(report: ValidationReport) -> HTML:
    """Flat table of per-segment check results."""
    if not report.segment_results:
        return HTML("<p><em>No segment results.</em></p>")

    rows: list[str] = []
    for seg in report.segment_results:
        for proxy_col, proxy_res in seg.proxy_results.items():
            for sub_prop, result in iter_all_check_results(proxy_res):
                passed = result.thresholding_result
                bg = _PASS_BG if passed else _FAIL_BG
                result_label = _PASS_SYMBOL if passed else _FAIL_SYMBOL
                result_color = "#155724" if passed else "#721c24"

                metrics = _CHECK_METRICS.get(result.check_name)
                metric_label = metrics[0] if metrics else ""
                metric_attr = metrics[1] if metrics else ""

                metric_val = (
                    getattr(result.check_result, metric_attr, None)
                    if metric_attr
                    else None
                )
                if metric_val is not None and isinstance(metric_val, (int, float)):
                    metric_display = f"{metric_label} = {_format_metric(metric_val)}"
                else:
                    metric_display = f"{result.projection_result:.4f}"

                sub_prop_display = html_mod.escape(sub_prop.replace("_", " ").title())

                check_name_escaped = html_mod.escape(result.check_name)
                name_html = f"<strong>{check_name_escaped}</strong>"
                if result.description:
                    name_html += (
                        f"<div style='{_DESC_STYLE}'>"
                        f"{html_mod.escape(result.description)}</div>"
                    )

                group_escaped = html_mod.escape(seg.group_name)
                seg_escaped = html_mod.escape(seg.segment_name)
                proxy_escaped = html_mod.escape(proxy_col)

                rows.append(
                    f"<tr style='background:{bg};'>"
                    f"<td style='{_CELL_STYLE}'>{group_escaped}</td>"
                    f"<td style='{_CELL_STYLE}'>{seg_escaped}</td>"
                    f"<td style='{_CELL_STYLE}'>{proxy_escaped}</td>"
                    f"<td style='{_CELL_STYLE}'>{sub_prop_display}</td>"
                    f"<td style='{_CELL_STYLE}'>{name_html}</td>"
                    f"<td style='{_CELL_STYLE} text-align:right;'>"
                    f"{metric_display}</td>"
                    f"<td style='{_CELL_STYLE} color:{result_color}; font-weight:600;'>"
                    f"{result_label}</td>"
                    f"</tr>"
                )

    return HTML(
        f"<h3 style='margin:16px 0 8px;'>Segment Results</h3>"
        f"<table style='{_TABLE_STYLE}'>"
        f"<thead><tr>"
        f"<th style='{_HEADER_STYLE}'>Group</th>"
        f"<th style='{_HEADER_STYLE}'>Segment</th>"
        f"<th style='{_HEADER_STYLE}'>Proxy</th>"
        f"<th style='{_HEADER_STYLE}'>Sub-Property</th>"
        f"<th style='{_HEADER_STYLE}'>Check</th>"
        f"<th style='{_HEADER_STYLE} text-align:right;'>Metric</th>"
        f"<th style='{_HEADER_STYLE}'>Result</th>"
        f"</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        f"</table>"
    )
