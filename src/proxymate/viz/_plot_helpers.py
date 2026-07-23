# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from proxymate.orchestrator.results import ProxyValidationResult, ValidationReport
from proxymate.types.inputs import UnitLevelInput

plt.style.use("seaborn-v0_8-whitegrid")

# --- Color palette ---
GREEN = "#018571"
RED = "#e74c3c"
BLUE = "#3498db"
ORANGE = "#e67e22"
GRAY = "#95a5a6"
PRIMARY_COLOR = "#6C3CE9"
PROXY_COLOR = "#2E86DE"


def resolve_proxy_col(input: UnitLevelInput, proxy_col: str | None) -> str:
    if proxy_col is not None:
        return proxy_col
    if len(input.proxy_cols) == 1:
        return input.proxy_cols[0]
    raise ValueError("Multiple proxy columns found. Specify proxy_col explicitly.")


def clean_axes(ax: plt.Axes) -> None:
    """Remove top and right spines for a cleaner look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def resolve_proxy_result(
    report: ValidationReport, proxy_col: str
) -> ProxyValidationResult | None:
    """Look up the ProxyValidationResult for a given proxy column."""
    return report.per_proxy_results.get(proxy_col)


def annotate_check_results(
    ax: plt.Axes,
    proxy_result: ProxyValidationResult,
    sub_properties: list[str] | None = None,
) -> None:
    """Add a text box to the axes showing check results.

    Args:
        ax: Matplotlib axes to annotate.
        proxy_result: The proxy validation result containing check results.
        sub_properties: If provided, only show results whose sub_property
            value is in this list. If None, shows all results.
    """
    lines: list[str] = []
    for sub_prop, r in iter_all_check_results(proxy_result):
        if sub_properties is not None and sub_prop not in sub_properties:
            continue
        symbol = "PASS" if r.thresholding_result else "FAIL"
        lines.append(f"{r.check_name}: {r.projection_result:.3f} [{symbol}]")
    if lines:
        text = "\n".join(lines)
        ax.text(
            0.98,
            0.02,
            text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "alpha": 0.85,
            },
            family="monospace",
        )


def get_paired_arrays(
    input: UnitLevelInput, proxy_col: str
) -> tuple[np.ndarray, np.ndarray]:
    """Extract finite primary/proxy arrays from a UnitLevelInput."""
    primary = input.df[input.primary_col].to_numpy(dtype=float)
    proxy = input.df[proxy_col].to_numpy(dtype=float)
    mask = np.isfinite(primary) & np.isfinite(proxy)
    return primary[mask], proxy[mask]


def iter_all_check_results(
    proxy_result: ProxyValidationResult,
) -> list[tuple[str, Any]]:
    """Return (sub_property_name, CheckResult) for every result in a proxy."""
    items: list[tuple[str, Any]] = []
    for prop, results in proxy_result.unit_level_results.items():
        for r in results:
            items.append((prop.value, r))
    for prop, results in proxy_result.estimate_level_results.items():
        for r in results:
            items.append((prop.value, r))
    for prop, results in proxy_result.domain_level_results.items():
        for r in results:
            items.append((prop.value, r))
    for r in proxy_result.representativity_results:
        items.append(("representativity", r))
    return items
