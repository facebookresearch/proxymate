# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from unittest import TestCase

import pandas as pd
from proxymate.data.data_handler import DataHandler, sql_where_to_pandas_query


class TestSqlWhereToPandasQuery(TestCase):
    def test_basic_equality(self) -> None:
        result = sql_where_to_pandas_query("col = 5")
        self.assertEqual(result, "col == 5")

    def test_inequality(self) -> None:
        result = sql_where_to_pandas_query("col != 5")
        self.assertEqual(result, "col != 5")

    def test_gte_lte(self) -> None:
        result = sql_where_to_pandas_query("col >= 5 AND col <= 10")
        self.assertEqual(result, "col >= 5 and col <= 10")

    def test_logical_operators(self) -> None:
        result = sql_where_to_pandas_query("a = 1 AND b = 2 OR c = 3")
        self.assertEqual(result, "a == 1 and b == 2 or c == 3")

    def test_not_operator(self) -> None:
        result = sql_where_to_pandas_query("NOT a = 1")
        self.assertEqual(result, "not a == 1")

    def test_boolean_literals(self) -> None:
        result = sql_where_to_pandas_query("is_active = TRUE")
        self.assertEqual(result, "is_active == True")

    def test_in_operator(self) -> None:
        result = sql_where_to_pandas_query("col IN (1, 2, 3)")
        self.assertEqual(result, "col in (1, 2, 3)")

    def test_string_literal_with_equals(self) -> None:
        result = sql_where_to_pandas_query("col = 'a=b'")
        self.assertEqual(result, "col == 'a=b'")


class TestDataHandlerFromDataframes(TestCase):
    def _make_metrics_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "row_id": ["a", "a", "b", "b", "c", "c"],
                "model_name": [
                    "primary",
                    "proxy_model",
                    "primary",
                    "proxy_model",
                    "primary",
                    "proxy_model",
                ],
                "metric": [1.0, 1.1, 2.0, 2.2, 3.0, 3.3],
            }
        )

    def _make_entities_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "row_id": ["a", "b", "c"],
                "segment": ["x", "x", "y"],
                "age": [25, 30, 35],
            }
        )

    def test_basic_pivot_merge(self) -> None:
        handler = DataHandler.from_dataframes(
            metrics_df=self._make_metrics_df(),
            entities_df=self._make_entities_df(),
            primary_outcome_name="primary",
            proxy_outcome_names=["proxy_model"],
        )
        self.assertEqual(handler.primary_col, "primary_outcome")
        self.assertEqual(handler.proxy_cols, ["proxy_proxy_model"])
        self.assertEqual(len(handler.df), 3)
        self.assertIn("primary_outcome", handler.df.columns)
        self.assertIn("proxy_proxy_model", handler.df.columns)
        self.assertIn("segment", handler.df.columns)
        self.assertIn("age", handler.df.columns)

    def test_weight_extraction(self) -> None:
        metrics = self._make_metrics_df()
        metrics["weight"] = [1.0, 1.0, 2.0, 2.0, 0.5, 0.5]
        handler = DataHandler.from_dataframes(
            metrics_df=metrics,
            entities_df=self._make_entities_df(),
            primary_outcome_name="primary",
            proxy_outcome_names=["proxy_model"],
            weight_col="w",
        )
        self.assertEqual(handler.weight_col, "w")
        self.assertIn("w", handler.df.columns)

    def test_to_unit_level_input(self) -> None:
        handler = DataHandler.from_dataframes(
            metrics_df=self._make_metrics_df(),
            entities_df=self._make_entities_df(),
            primary_outcome_name="primary",
            proxy_outcome_names=["proxy_model"],
        )
        uli = handler.to_unit_level_input()
        self.assertEqual(uli.primary_col, "primary_outcome")
        self.assertEqual(len(uli.proxy_cols), 1)

    def test_multiple_proxies(self) -> None:
        metrics = pd.DataFrame(
            {
                "row_id": ["a", "a", "a", "b", "b", "b"],
                "model_name": [
                    "primary",
                    "model1",
                    "model2",
                    "primary",
                    "model1",
                    "model2",
                ],
                "metric": [1.0, 1.1, 1.2, 2.0, 2.1, 2.2],
            }
        )
        entities = pd.DataFrame({"row_id": ["a", "b"]})
        handler = DataHandler.from_dataframes(
            metrics_df=metrics,
            entities_df=entities,
            primary_outcome_name="primary",
            proxy_outcome_names=["model1", "model2"],
        )
        self.assertEqual(len(handler.proxy_cols), 2)
        self.assertEqual(len(handler.df), 2)
