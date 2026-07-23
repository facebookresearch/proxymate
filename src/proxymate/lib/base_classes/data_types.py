# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass


@dataclass
class Estimate:
    confidence_interval_level: float
    estimate_val: float
    lower_bound: float
    upper_bound: float
    sample_size: int
    standard_error: float

    @property
    def ci_width(self) -> float:
        """Width of the confidence interval."""
        return self.upper_bound - self.lower_bound

    @property
    def relative_error(self) -> float:
        """Relative standard error (coefficient of variation)."""
        if self.estimate_val == 0:
            return float("inf")
        return abs(self.standard_error / self.estimate_val)

    def __repr__(self) -> str:
        return (
            f"Estimate({self.estimate_val:.4f} "
            f"[{self.lower_bound:.4f}, {self.upper_bound:.4f}], "
            f"n={self.sample_size:,})"
        )

    def _repr_html_(self) -> str:
        ci_pct = self.confidence_interval_level * 100
        return (
            "\n"
            '        <div style="border:1px solid #6c757d; padding:10px; '
            'border-radius:5px; margin:5px 0;">\n'
            '            <h4 style="margin:0 0 8px 0;">Estimate</h4>\n'
            '            <table style="border-collapse:collapse;">\n'
            '                <tr><td style="padding-right:10px;"><b>Value:</b>'
            f"</td><td>{self.estimate_val:.4f}</td></tr>\n"
            '                <tr><td style="padding-right:10px;">'
            f"<b>{ci_pct:.0f}% CI:</b></td>"
            f"<td>[{self.lower_bound:.4f}, {self.upper_bound:.4f}]</td></tr>\n"
            '                <tr><td style="padding-right:10px;"><b>Std Error:</b>'
            f"</td><td>{self.standard_error:.4f}</td></tr>\n"
            '                <tr><td style="padding-right:10px;"><b>Sample Size:</b>'
            f"</td><td>{self.sample_size:,}</td></tr>\n"
            "            </table>\n"
            "        </div>\n"
            "        "
        )
