# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import List

_logger = logging.getLogger(__name__)


def create_sql_str(python_list: List[str], with_parenthesis: bool = True) -> str:
    if len(python_list) == 0:
        _logger.warning("python list is empty, returning empty string")
        return "'()'" if with_parenthesis else ""
    sql_str_list = "', '".join(python_list)
    if with_parenthesis:
        return f"('{sql_str_list}')"
    else:
        return f"'{sql_str_list}'"


def create_clause_name(
    clause_name: str, python_list: List[str], proxy_names: List[str]
) -> str:
    return (
        "CASE "
        + " ".join(
            [
                f"WHEN model_name = '{proxy}' THEN '{proxy_name}'"
                for proxy, proxy_name in zip(python_list, proxy_names)
            ]
        )
        + f" END AS {clause_name}"
    )


def transform_array_name_to_sql(name: str) -> str:
    """
    This function transforms additional columns which come from the nested array
    values to a format that can be used in SQL queries.
    """
    return name.replace("'", "").replace("[", "_").replace("]", "")
