# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from proxymate.data.bootstrap import BootstrapSampler, resample_df
from proxymate.lib.base_classes.df_based_estimator_base import (
    DfBasedEstimatorBase,
    DfBasedEstimatorBaseInput,
)
from proxymate.lib.base_classes.estimator_base import EstimatorBase
from proxymate.lib.base_classes.transformation_base import TransformationBase
from proxymate.lib.segment import Segment
from proxymate.types.data import Estimate
from proxymate.types.inputs import (
    BootstrappedEstimatesInput,
    MetaAnalysisInput,
    UnitLevelInput,
)


def sql_where_to_pandas_query(where_clause: str) -> str:
    """Translate a SQL WHERE clause to a pandas query string.

    Handles the subset of SQL supported by the Segment class:
    - Comparison operators: = != > < >= <=
    - Logical operators: AND OR NOT
    - IN with parenthesized lists
    - String literals in single quotes

    Transformations:
    - ``=`` → ``==`` (except inside ``!=``, ``>=``, ``<=``, or string literals)
    - ``AND`` → ``and``, ``OR`` → ``or``, ``NOT`` → ``not``
    - ``TRUE`` → ``True``, ``FALSE`` → ``False``
    - ``IN (...)`` → ``.isin([...])`` is NOT supported by pandas query;
      instead we keep ``IN`` as-is since pandas ``.query()`` supports ``in``.
    """
    result = where_clause

    # Lowercase logical operators and boolean literals
    result = re.sub(r"\bAND\b", "and", result)
    result = re.sub(r"\bOR\b", "or", result)
    result = re.sub(r"\bNOT\b", "not", result)
    result = re.sub(r"\bTRUE\b", "True", result)
    result = re.sub(r"\bFALSE\b", "False", result)
    result = re.sub(r"\bIN\b", "in", result)

    # Replace SQL single = with == (but not !=, >=, <=)
    # Strategy: extract string literals first, do the swap, restore
    string_literals: list[str] = []

    def _save_literal(m: re.Match[str]) -> str:
        string_literals.append(m.group(0))
        return f"__STR{len(string_literals) - 1}__"

    result = re.sub(r"'[^']*'", _save_literal, result)

    result = result.replace("!=", "__NEQ__")
    result = result.replace(">=", "__GEQ__")
    result = result.replace("<=", "__LEQ__")
    result = result.replace("=", "==")
    result = result.replace("__NEQ__", "!=")
    result = result.replace("__GEQ__", ">=")
    result = result.replace("__LEQ__", "<=")

    # Restore string literals
    for i, literal in enumerate(string_literals):
        result = result.replace(f"__STR{i}__", literal)

    return result


def _compute_domain_covariance(
    df: pd.DataFrame,
    primary_col: str,
    proxy_col: str,
    treatment_col: str | None,
) -> float:
    """Compute cov(estimand_primary, estimand_proxy) within a single domain.

    For a simple mean estimand (``treatment_col is None``)::

        cov(Ȳ, Ȳ*) = cov(Y, Y*) / n

    For a two-arm ATE estimand::

        cov(ATE_primary, ATE_proxy) = cov(Y,Y*)/n_treat + cov(Y,Y*)/n_control

    where cov(Y, Y*) is estimated per-arm from paired data.
    """
    if treatment_col is None:
        paired = df[[primary_col, proxy_col]].dropna()
        if len(paired) < 2:
            return 0.0
        cov_val = float(paired[primary_col].cov(paired[proxy_col]))
        return cov_val / len(paired)

    paired = df[[primary_col, proxy_col, treatment_col]].dropna()
    if len(paired) < 2:
        return 0.0

    treat_mask = paired[treatment_col].astype(bool)
    treat_df = paired[treat_mask]
    control_df = paired[~treat_mask]

    cov_total = 0.0
    for arm_df in (treat_df, control_df):
        n_arm = len(arm_df)
        if n_arm < 2:
            continue
        arm_cov = float(arm_df[primary_col].cov(arm_df[proxy_col]))
        cov_total += arm_cov / n_arm

    return cov_total


@dataclass
class DataHandler:
    """Handles data ingestion from the standard two-table format.

    Accepts a metrics table (long format: row_id, model_name, metric, weight)
    and an entities table (row_id + segment/covariate columns), merges them
    internally, pivots from long to wide format, and exposes the merged
    DataFrame.

    The merged DataFrame has one row per ``row_id`` with columns:
    - ``primary_col`` — the primary outcome values
    - one column per proxy outcome
    - ``weight`` — from the metrics table (if present)
    - all entity columns (segment covariates, etc.)

    Bootstrap caching:
    - ``run_bootstrap()`` resamples the DataFrame N times, computes estimates
      for primary + each proxy using a shared set of resampled indices, and
      caches the results internally.
    - ``get_bootstrapped_estimates_input(proxy_col)`` returns cached
      ``BootstrappedEstimatesInput`` for a given proxy without re-computation.
    - When multiple proxies are present, all share the same bootstrap indices.
    """

    df: pd.DataFrame
    primary_col: str
    proxy_cols: list[str]
    weight_col: str | None

    # Bootstrap state (cached, not part of __init__)
    _bootstrap_samples: np.ndarray | None = field(default=None, init=False, repr=False)
    _bootstrap_seed: int | None = field(default=None, init=False, repr=False)
    _bootstrap_n_iterations: int | None = field(default=None, init=False, repr=False)
    _bootstrapped_estimates: dict[str, list[Estimate]] | None = field(
        default=None, init=False, repr=False
    )
    _primary_estimate: Estimate | None = field(default=None, init=False, repr=False)
    _proxy_estimates: dict[str, Estimate] | None = field(
        default=None, init=False, repr=False
    )
    _primary_bootstrapped_estimates: list[Estimate] | None = field(
        default=None, init=False, repr=False
    )

    # Meta-analysis state (cached, not part of __init__)
    _meta_analysis_inputs: dict[str, MetaAnalysisInput] | None = field(
        default=None, init=False, repr=False
    )
    _meta_analysis_domain_col: str | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_dataframes(
        cls,
        metrics_df: pd.DataFrame,
        entities_df: pd.DataFrame,
        primary_outcome_name: str,
        proxy_outcome_names: list[str],
        weight_col: str | None = None,
    ) -> "DataHandler":
        """Build from in-memory DataFrames matching the proxymate table schema.

        Args:
            metrics_df: Long-format metrics table with columns
                ``row_id``, ``model_name``, ``metric``, and optionally ``weight``.
            entities_df: Entities table with ``row_id`` and segment/covariate columns.
            primary_outcome_name: The ``model_name`` value for the primary outcome.
            proxy_outcome_names: The ``model_name`` values for each proxy outcome.
            weight_col: If provided, the name of the weight column in the
                output DataFrame. Weights are taken from the primary outcome rows.
        """
        all_model_names = [primary_outcome_name] + proxy_outcome_names

        # Filter metrics to only the models we care about
        filtered_metrics = metrics_df[
            metrics_df["model_name"].isin(all_model_names)
        ].copy()

        # Pivot from long to wide: row_id × model_name → metric value
        pivoted = filtered_metrics.pivot_table(
            index="row_id",
            columns="model_name",
            values="metric",
            aggfunc="first",
        ).reset_index()

        # Rename pivoted columns to standard names
        primary_col = "primary_outcome"
        rename_map: dict[str, str] = {primary_outcome_name: primary_col}
        proxy_cols: list[str] = []
        for proxy_name in proxy_outcome_names:
            col_name = f"proxy_{proxy_name}"
            rename_map[proxy_name] = col_name
            proxy_cols.append(col_name)

        pivoted = pivoted.rename(columns=rename_map)

        # Ensure numeric columns are float64 — non-numeric dtypes (STRING,
        # Decimal, etc.) break downstream numpy operations.
        numeric_cols = [primary_col] + proxy_cols
        for col in numeric_cols:
            pivoted[col] = pd.to_numeric(pivoted[col], errors="coerce")

        # Extract weights from the primary outcome rows if requested
        resolved_weight_col: str | None = None
        if weight_col is not None and "weight" in filtered_metrics.columns:
            primary_weights = (
                filtered_metrics[
                    filtered_metrics["model_name"] == primary_outcome_name
                ][["row_id", "weight"]]
                .drop_duplicates(subset=["row_id"])
                .rename(columns={"weight": weight_col})
            )
            pivoted = pivoted.merge(primary_weights, on="row_id", how="left")
            pivoted[weight_col] = pd.to_numeric(pivoted[weight_col], errors="coerce")
            resolved_weight_col = weight_col

        # Merge with entities table to get segment/covariate columns
        # Drop columns from entities that already exist in pivoted (except row_id)
        entity_cols_to_keep = [
            c for c in entities_df.columns if c not in pivoted.columns or c == "row_id"
        ]
        merged = pivoted.merge(
            entities_df[entity_cols_to_keep], on="row_id", how="inner"
        )

        return cls(
            df=merged,
            primary_col=primary_col,
            proxy_cols=proxy_cols,
            weight_col=resolved_weight_col,
        )

    def to_unit_level_input(self) -> UnitLevelInput:
        """Convert to a UnitLevelInput for the validator."""
        return UnitLevelInput(
            df=self.df,
            primary_col=self.primary_col,
            proxy_cols=self.proxy_cols,
            weight_col=self.weight_col,
        )

    def filter_segment(self, segment: Segment) -> "DataHandler":
        """Return a new DataHandler with rows filtered to the segment.

        Translates the segment's SQL WHERE clause to a pandas query.
        """
        pandas_query = sql_where_to_pandas_query(segment.get_where_clause())
        filtered_df = self.df.query(pandas_query).reset_index(drop=True)
        return DataHandler(
            df=filtered_df,
            primary_col=self.primary_col,
            proxy_cols=self.proxy_cols,
            weight_col=self.weight_col,
        )

    def apply_transformations(
        self,
        transformations: list[TransformationBase],
        suffix: str | None = None,
    ) -> "DataHandler":
        """Return a new DataHandler with transformed proxy columns added.

        For each proxy column, applies the transformation pipeline and adds
        the result as a new column. The original proxy columns are preserved.
        The transformed columns are named ``{proxy_col}_{suffix}`` where
        suffix defaults to the transformation name (e.g.,
        ``"log_transformation"``).

        Transformations apply to proxy columns only — the primary column is
        left untouched.

        Args:
            transformations: List of transformations to apply in sequence.
            suffix: Custom suffix for transformed column names. If None,
                defaults to the concatenated transformation names
                (e.g., ``"log_transformation"``).

        Returns:
            A new DataHandler with the original proxy columns plus new
            transformed proxy columns added.
        """
        if not transformations:
            return self

        # Determine suffix — sanitize to valid identifier characters only
        # (e.g., "Min-Max Scale Transformation" → "min_max_scale_transformation")
        if suffix is None:
            suffix = "_".join(
                re.sub(r"[^a-z0-9_]", "_", t.get_name().lower())
                for t in transformations
            )

        df = self.df.copy()
        new_proxy_cols = list(self.proxy_cols)

        for proxy_col in self.proxy_cols:
            new_col = f"{proxy_col}_{suffix}"
            # Copy the proxy column, apply transformations, store as new column
            df[new_col] = df[proxy_col].copy()
            for t in transformations:
                temp_df = df[[new_col]].rename(columns={new_col: proxy_col})
                temp_df = t.apply(temp_df, primary_col=proxy_col, proxy_col=proxy_col)
                df[new_col] = temp_df[proxy_col]
            new_proxy_cols.append(new_col)

        return DataHandler(
            df=df,
            primary_col=self.primary_col,
            proxy_cols=new_proxy_cols,
            weight_col=self.weight_col,
        )

    # --- Bootstrap execution and caching ---

    def run_bootstrap(
        self,
        estimator: EstimatorBase,
        n_iterations: int = 100,
        seed: int | None = None,
    ) -> None:
        """Generate bootstrap samples and compute estimates. Cache results.

        Uses pandas-based in-memory resampling. When multiple proxies are
        present, all share the same resampled indices per iteration, ensuring
        comparable bootstrap variance estimates.

        Args:
            estimator: A ``DfBasedEstimatorBase`` used to compute aggregate
                estimates.
            n_iterations: Number of bootstrap iterations.
            seed: Random seed for reproducibility.

        Raises:
            TypeError: If estimator is not a ``DfBasedEstimatorBase``.
        """
        if not isinstance(estimator, DfBasedEstimatorBase):
            raise TypeError(
                f"estimator must be a DfBasedEstimatorBase, "
                f"got {type(estimator).__name__}"
            )
        self._run_bootstrap_pandas(estimator, n_iterations, seed)

    def _run_bootstrap_pandas(
        self,
        estimator: DfBasedEstimatorBase,
        n_iterations: int,
        seed: int | None,
    ) -> None:
        """In-memory pandas-based bootstrap."""
        _logger = logging.getLogger(__name__)

        # Include additional columns required by the estimator (e.g. weight column)
        extra_cols = sorted(
            estimator.get_additional_metric_columns()
            | estimator.get_additional_entities_columns()
        )
        extra_cols = [c for c in extra_cols if c in self.df.columns]

        # Compute original primary estimate
        primary_df = (
            self.df[[self.primary_col] + extra_cols]
            .rename(columns={self.primary_col: "metric"})
            .dropna(subset=["metric"])
        )
        primary_estimate = estimator.compute_estimator(
            DfBasedEstimatorBaseInput(df=primary_df)
        )

        # Compute true proxy estimates on full (non-resampled) data
        true_proxy_estimates: dict[str, Estimate] = {}
        for proxy_col in self.proxy_cols:
            proxy_full_df = (
                self.df[[proxy_col] + extra_cols]
                .rename(columns={proxy_col: "metric"})
                .dropna(subset=["metric"])
            )
            true_proxy_estimates[proxy_col] = estimator.compute_estimator(
                DfBasedEstimatorBaseInput(df=proxy_full_df)
            )

        # Generate bootstrap samples (shared across all proxies)
        sampler = BootstrapSampler(seed=seed)
        n_rows = len(self.df)
        samples = sampler.generate_samples(n_rows, n_iterations)

        proxy_estimates: dict[str, list[Estimate]] = {
            col: [] for col in self.proxy_cols
        }
        primary_bootstrapped: list[Estimate] = []

        for i, indices in enumerate(samples):
            resampled_df = resample_df(self.df, indices)

            # Primary estimate on resampled data
            resampled_primary_df = (
                resampled_df[[self.primary_col] + extra_cols]
                .rename(columns={self.primary_col: "metric"})
                .dropna(subset=["metric"])
            )
            primary_bootstrapped.append(
                estimator.compute_estimator(
                    DfBasedEstimatorBaseInput(df=resampled_primary_df)
                )
            )

            # Proxy estimates on resampled data
            for proxy_col in self.proxy_cols:
                proxy_df = (
                    resampled_df[[proxy_col] + extra_cols]
                    .rename(columns={proxy_col: "metric"})
                    .dropna(subset=["metric"])
                )
                proxy_estimates[proxy_col].append(
                    estimator.compute_estimator(DfBasedEstimatorBaseInput(df=proxy_df))
                )

            if (i + 1) % 50 == 0:
                _logger.info("Bootstrap iteration %d/%d", i + 1, n_iterations)

        self._bootstrap_samples = samples
        self._bootstrap_seed = seed
        self._bootstrap_n_iterations = n_iterations
        self._bootstrapped_estimates = proxy_estimates
        self._primary_estimate = primary_estimate
        self._proxy_estimates = true_proxy_estimates
        self._primary_bootstrapped_estimates = primary_bootstrapped

    def get_bootstrapped_estimates_input(
        self, proxy_col: str
    ) -> BootstrappedEstimatesInput:
        """Return cached bootstrap results for a proxy.

        Can be called multiple times for different checks without re-running
        the bootstrap.

        Args:
            proxy_col: Column name of the proxy to get results for.

        Returns:
            BootstrappedEstimatesInput with primary_estimate and
            proxy_bootstrapped_estimates.

        Raises:
            ValueError: If bootstrap hasn't been run, or proxy_col not found.
        """
        if self._bootstrapped_estimates is None or self._primary_estimate is None:
            raise ValueError("Bootstrap not run. Call run_bootstrap() first.")
        if proxy_col not in self._bootstrapped_estimates:
            raise ValueError(
                f"Proxy '{proxy_col}' not found in bootstrap results. "
                f"Available: {list(self._bootstrapped_estimates.keys())}"
            )
        proxy_est = (
            self._proxy_estimates.get(proxy_col) if self._proxy_estimates else None
        )
        return BootstrappedEstimatesInput(
            primary_estimate=self._primary_estimate,
            proxy_bootstrapped_estimates=self._bootstrapped_estimates[proxy_col],
            proxy_estimate=proxy_est,
            primary_bootstrapped_estimates=self._primary_bootstrapped_estimates or [],
        )

    @property
    def is_bootstrapped(self) -> bool:
        """Check if bootstrap has been run."""
        return self._bootstrapped_estimates is not None

    def is_bootstrapped_with(self, seed: int | None, n_iterations: int) -> bool:
        """Check if bootstrap was run with matching seed and iteration count.

        Used by the validator to decide whether to reuse cache or re-run.
        """
        return (
            self._bootstrapped_estimates is not None
            and self._bootstrap_seed == seed
            and self._bootstrap_n_iterations == n_iterations
        )

    def get_proxy_estimate(self, proxy_col: str) -> Estimate:
        """Return the true (non-bootstrapped) proxy estimate computed on full data.

        Raises if bootstrap hasn't been run.
        """
        if self._proxy_estimates is None:
            raise ValueError("Bootstrap not run. Call run_bootstrap() first.")
        if proxy_col not in self._proxy_estimates:
            raise ValueError(
                f"Proxy '{proxy_col}' not found. "
                f"Available: {list(self._proxy_estimates.keys())}"
            )
        return self._proxy_estimates[proxy_col]

    # --- Meta-analysis construction and caching ---

    @property
    def is_meta_analysis_built(self) -> bool:
        """Check if meta-analysis inputs have been built."""
        return self._meta_analysis_inputs is not None

    def get_meta_analysis_input(self, proxy_col: str) -> MetaAnalysisInput:
        """Return cached MetaAnalysisInput for a proxy.

        Args:
            proxy_col: Column name of the proxy to get results for.

        Raises:
            ValueError: If meta-analysis hasn't been built, or proxy_col not found.
        """
        if self._meta_analysis_inputs is None:
            raise ValueError(
                "Meta-analysis not built. Call build_meta_analysis_input() first."
            )
        if proxy_col not in self._meta_analysis_inputs:
            raise ValueError(
                f"Proxy '{proxy_col}' not found in meta-analysis results. "
                f"Available: {list(self._meta_analysis_inputs.keys())}"
            )
        return self._meta_analysis_inputs[proxy_col]

    def build_meta_analysis_input(
        self,
        domain_col: str,
        estimator: EstimatorBase | dict[str, EstimatorBase],
        min_domain_rows: int = 2,
        treatment_col: str | None = None,
    ) -> dict[str, MetaAnalysisInput]:
        """Build per-proxy MetaAnalysisInput by grouping data on a domain column.

        Groups the data by unique values in ``domain_col``, computes paired
        (primary, proxy) estimates per domain using the estimator, and returns
        a ``MetaAnalysisInput`` per proxy column.

        Estimates are computed in-memory with pandas via a
        ``DfBasedEstimatorBase``.

        Results are cached on the DataHandler instance. Subsequent calls with
        the same ``domain_col`` reuse the cache.

        Args:
            domain_col: Column name to group by (e.g., ``"experiment_id"``).
            estimator: Estimator(s) used to compute aggregate estimates.
                A single ``EstimatorBase`` is applied uniformly to all
                domains. A ``dict[str, EstimatorBase]`` maps domain values
                (as strings) to per-domain estimators, allowing different
                metric types across experiments.
            min_domain_rows: Minimum rows per domain; domains with fewer
                rows are skipped with a warning.
            treatment_col: If provided, computes per-arm covariances for
                ATE estimands. The column must be binary (0/1 or bool),
                where 1 indicates the treatment arm. The covariance of
                the ATE estimates is::

                    cov(ATE_primary, ATE_proxy) =
                        cov(Y, Y*) / n_treat + cov(Y, Y*) / n_control

                When ``None``, assumes a simple mean estimand and computes
                ``cov(Y, Y*) / n``.

        Returns:
            Dict mapping proxy column name → MetaAnalysisInput.

        Raises:
            ValueError: If ``domain_col`` is not in the DataFrame, or no
                valid domains remain after filtering.
            TypeError: If estimator is not a supported type.
        """
        # Return cache if already built for the same domain_col
        if (
            self._meta_analysis_inputs is not None
            and self._meta_analysis_domain_col == domain_col
        ):
            return self._meta_analysis_inputs

        if domain_col not in self.df.columns:
            raise ValueError(
                f"domain_col '{domain_col}' not found in DataFrame. "
                f"Available columns: {list(self.df.columns)}"
            )

        sample_estimator: EstimatorBase = (
            next(iter(estimator.values())) if isinstance(estimator, dict) else estimator
        )
        if not isinstance(sample_estimator, DfBasedEstimatorBase):
            raise TypeError(
                f"estimator must be a DfBasedEstimatorBase, "
                f"got {type(sample_estimator).__name__}"
            )
        result = self._build_meta_analysis_pandas(
            domain_col,
            estimator,  # pyre-ignore[bad-argument-type]
            min_domain_rows,
            treatment_col,
        )

        self._meta_analysis_inputs = result
        self._meta_analysis_domain_col = domain_col
        return result

    def _build_meta_analysis_pandas(
        self,
        domain_col: str,
        estimator: DfBasedEstimatorBase | dict[str, DfBasedEstimatorBase],
        min_domain_rows: int,
        treatment_col: str | None = None,
    ) -> dict[str, MetaAnalysisInput]:
        """Build MetaAnalysisInput using in-memory pandas computation."""
        _logger = logging.getLogger(__name__)

        domain_values = sorted(self.df[domain_col].dropna().unique())

        # Per-proxy accumulators
        proxy_domain_estimates: dict[str, list[tuple[Estimate, Estimate]]] = {
            col: [] for col in self.proxy_cols
        }
        valid_domain_names: list[str] = []

        for domain_val in domain_values:
            segment = Segment(
                str(domain_val),
                f"{domain_col} = '{domain_val}'",
                [domain_col],
            )
            filtered = self.filter_segment(segment)

            if len(filtered.df) < min_domain_rows:
                _logger.warning(
                    "Domain '%s=%s' has only %d row(s) "
                    "(min_domain_rows=%d) — skipping.",
                    domain_col,
                    domain_val,
                    len(filtered.df),
                    min_domain_rows,
                )
                continue

            # Resolve estimator for this domain
            domain_name = str(domain_val)
            if isinstance(estimator, dict):
                if domain_name not in estimator:
                    _logger.warning(
                        "No estimator for domain '%s' — skipping.",
                        domain_name,
                    )
                    continue
                domain_estimator = estimator[domain_name]
            else:
                domain_estimator = estimator

            # Compute primary estimate once per domain (shared across proxies)
            extra_cols = sorted(
                domain_estimator.get_additional_metric_columns()
                | domain_estimator.get_additional_entities_columns()
            )
            extra_cols = [c for c in extra_cols if c in filtered.df.columns]

            primary_df = (
                filtered.df[[filtered.primary_col] + extra_cols]
                .rename(columns={filtered.primary_col: "metric"})
                .dropna(subset=["metric"])
            )
            primary_estimate = domain_estimator.compute_estimator(
                DfBasedEstimatorBaseInput(df=primary_df)
            )

            for proxy_col in self.proxy_cols:
                proxy_df = (
                    filtered.df[[proxy_col] + extra_cols]
                    .rename(columns={proxy_col: "metric"})
                    .dropna(subset=["metric"])
                )
                proxy_estimate = domain_estimator.compute_estimator(
                    DfBasedEstimatorBaseInput(df=proxy_df)
                )
                proxy_domain_estimates[proxy_col].append(
                    (primary_estimate, proxy_estimate)
                )

            valid_domain_names.append(str(domain_val))

        if not valid_domain_names:
            raise ValueError(
                f"No valid domains found after filtering by '{domain_col}'. "
                f"All domains had fewer than {min_domain_rows} rows."
            )

        return {
            proxy_col: MetaAnalysisInput(
                domain_estimates=proxy_domain_estimates[proxy_col],
                domain_names=valid_domain_names,
            )
            for proxy_col in self.proxy_cols
        }
