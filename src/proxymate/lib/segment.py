# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import List, Set


class Segment:
    # Allowed operators in where clause
    _ALLOWED_OPERATORS: Set[str] = {
        ">",
        "<",
        ">=",
        "<=",
        "=",
        "!=",
        "IN",  # Comparison operators
        "AND",
        "OR",
        "NOT",  # Logical operators
        "TRUE",
        "FALSE",  # Boolean literals
        "(",
        ")",  # Parentheses for grouping
        ",",  # Comma for multiple values in IN operator
    }

    def __init__(
        self,
        name: str,
        where_clause: str,
        columns: List[str],
    ) -> None:
        """
        Initialize a Segment with a where clause string and column list.

        Args:
            name: The name of the segment (must be a valid SQL identifier)
            where_clause: A SQL WHERE clause string that represents a valid SQL
                          WHERE clause using only the allowed operators.
            columns: List of column names used in the where clause

        The where_clause should be a string that represents a valid SQL
        WHERE clause using only the allowed operators:
        - Comparison operators: > < >= <= = != IN
        - Logical operators: AND OR
        - Parentheses for grouping: ( )

        Example:
            ```python
            # Create segment with a where clause
            segment = Segment(
                "custom_segment",
                "(age > 25 AND status IN ('active', 'pending')) OR score >= 80",
                ["age", "status", "score"]
            )
            ```
        """

        self._name = name
        self._where_clause = where_clause
        self._columns = columns
        self._validate_segment_name(name)
        self._validate_where_clause(where_clause)

    @staticmethod
    def _validate_segment_name(name: str) -> None:
        """
        Validate that segment name is suitable for use as a SQL field name.

        Rules:
        - No spaces allowed
        - Must start with a letter or underscore
        - Can only contain letters, numbers, and underscores
        - Cannot be empty
        """
        if not name:
            raise ValueError("Segment name cannot be empty")

        # Check for spaces first (more specific error message)
        if " " in name:
            raise ValueError(f"Segment name '{name}' cannot contain spaces")

        # Check if name is valid SQL identifier (starts with letter/underscore,
        # contains only alphanumeric/underscore)
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            raise ValueError(
                f"Segment name '{name}' is not a valid SQL field name. "
                "It must start with a letter or underscore and contain only "
                "letters, numbers, and underscores."
            )

    def get_columns(self) -> List[str]:
        return self._columns

    def get_where_clause(self) -> str:
        return self._where_clause

    def _validate_where_clause(self, where_clause: str) -> None:
        """
        Validate that the where clause only uses allowed operators and that all columns
        used in the where_clause are included in the columns list.

        Operator validation:
        - Comparison operators: > < >= <= = != IN
        - Logical operators: AND OR NOT
        - Parentheses for grouping: ( )
        - Comma for multiple values in IN operator: ,

        String literals (text within single quotes) are allowed and not validated.

        Disallowed operators include but are not limited to:
        - LIKE, ILIKE (pattern matching)
        - BETWEEN (range check)
        - IS NULL, IS NOT NULL (null checks)
        - ANY, ALL, SOME (subquery operators)
        - CASE, WHEN, THEN, ELSE, END (conditional expressions)
        - Mathematical operators: +, -, *, /, %

        Column validation:
        - All columns referenced in the where_clause must be included in the
          columns list
        - This is a best-effort approach and may not catch all column references,
          especially in complex expressions
        """
        # Extract string literals and preprocess the clause
        modified_clause = self._extract_string_literals(where_clause)

        # Validate that the clause only uses allowed operators
        self._validate_operators(where_clause, modified_clause)

        # Validate that all columns used in the where_clause are included in
        # the columns list
        self._validate_columns(modified_clause)

    def _extract_string_literals(self, where_clause: str) -> str:
        """
        Extract string literals from the where clause and replace them with
        placeholders.

        Args:
            where_clause: The SQL WHERE clause to process

        Returns:
            A modified version of the where_clause with string literals replaced
            by placeholders
        """
        modified_clause = where_clause

        # Replace all string literals with placeholders
        # This regex matches anything between single quotes, including empty strings
        string_pattern = r"'[^']*'"
        string_matches = re.findall(string_pattern, where_clause)

        for i, match in enumerate(string_matches):
            modified_clause = modified_clause.replace(match, f"PLACEHOLDER_{i}")

        # Pre-process the clause to handle multi-character operators like !=, >=, <=
        # Replace them with single tokens to avoid treating them as separate characters
        modified_clause = modified_clause.replace("!=", " NOT_EQUALS ")
        modified_clause = modified_clause.replace(">=", " GREATER_EQUAL ")
        modified_clause = modified_clause.replace("<=", " LESS_EQUAL ")

        return modified_clause

    def _validate_operators(self, original_clause: str, modified_clause: str) -> None:
        """
        Validate that the where clause only uses allowed operators.

        Args:
            original_clause: The original SQL WHERE clause
            modified_clause: The modified clause with string literals replaced
                by placeholders

        Raises:
            ValueError: If the where clause contains disallowed operators
        """
        # Check for specific disallowed keywords
        disallowed_keywords = [
            "LIKE",
            "ILIKE",
            "BETWEEN",
            "NULL",
            "ANY",
            "ALL",
            "SOME",
            "CASE",
            "WHEN",
            "THEN",
            "ELSE",
            "END",
        ]

        for keyword in disallowed_keywords:
            pattern = r"\b" + keyword + r"\b"
            match = re.search(pattern, modified_clause.upper())
            if match:
                # Find the full pattern in the original where_clause
                # This will include the '%' characters in LIKE patterns
                context_start = max(0, match.start() - 10)
                context_end = min(len(modified_clause), match.end() + 10)
                error_context = original_clause[context_start:context_end]

                raise ValueError(
                    f"Disallowed operator '{keyword}' in where clause: "
                    f"'{error_context}'. "
                    f"Allowed operators are: "
                    f"{', '.join(sorted(self._ALLOWED_OPERATORS))}"
                )

        # Now tokenize the modified clause for operator validation
        tokens = re.findall(r"[A-Za-z0-9_\.]+|\S", modified_clause)

        # Check for disallowed operators
        for token in tokens:
            # Skip column names, values, placeholders, and whitespace
            if (
                token.isalnum()
                or token.startswith("PLACEHOLDER_")
                or token == "NOT_EQUALS"
                or token == "GREATER_EQUAL"
                or token == "LESS_EQUAL"
                or token.isspace()
                or "." in token
                or token.isdigit()
                or token in self._columns
            ):
                continue

            # Check if the token is an allowed operator
            if (
                token.upper() not in self._ALLOWED_OPERATORS
                and token not in self._ALLOWED_OPERATORS
            ):
                raise ValueError(
                    f"Disallowed operator '{token}' in where clause. "
                    f"Allowed operators are: "
                    f"{', '.join(sorted(self._ALLOWED_OPERATORS))}"
                )

    def _validate_columns(self, modified_clause: str) -> None:
        """
        Validate that all columns used in the where_clause are included in the
        columns list.

        Args:
            modified_clause: The modified clause with string literals replaced
                by placeholders

        Raises:
            ValueError: If the where clause contains columns that are not in
                the columns list
        """
        # Extract potential column references from the modified clause
        potential_columns = re.findall(r"[A-Za-z][A-Za-z0-9_]*", modified_clause)

        # Filter out SQL keywords, placeholders, and other non-column tokens
        # Include both allowed operators and our special placeholder tokens
        sql_keywords = {
            "AND",
            "OR",
            "IN",
            "NOT",
            "TRUE",
            "FALSE",
            # Special placeholder tokens we created earlier
            "NOT_EQUALS",
            "GREATER_EQUAL",
            "LESS_EQUAL",
        }
        missing_columns = []

        for col in potential_columns:
            if (
                col.upper() not in sql_keywords
                and not col.startswith("PLACEHOLDER_")
                and col not in self._columns
            ):
                missing_columns.append(col)

        if missing_columns:
            raise ValueError(
                f"The following columns are used in the where_clause but not "
                f"included in the columns list: {', '.join(missing_columns)}"
            )

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def check_segment_unique_names(cls, segments: List["Segment"]) -> None:
        unique_identifiers = {segment.name for segment in segments}
        if len(unique_identifiers) != len(segments):
            raise ValueError("Segment unique identifiers must be unique")

    @classmethod
    def get_segment_columns(cls, segments: List["Segment"]) -> List[str]:
        return list(set().union(*[segment.get_columns() for segment in segments]))


class SegmentGroup:
    def __init__(
        self,
        name: str,
        segments: List[Segment],
    ) -> None:
        # validate unique identifiers
        Segment.check_segment_unique_names(segments)
        self._name = name
        self._segments = segments

    def get_segment_columns(self) -> List[str]:
        return list(set().union(*[segment.get_columns() for segment in self._segments]))

    @property
    def segments(self) -> List[Segment]:
        return self._segments

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def check_groups_unique_identifiers(
        cls, segment_groups: List["SegmentGroup"]
    ) -> None:
        unique_identifiers = {segment_group.name for segment_group in segment_groups}
        if len(unique_identifiers) != len(segment_groups):
            raise ValueError("Segment group names must be unique")

    @classmethod
    def check_segments_unique_names_across_groups(
        cls, segment_groups: List["SegmentGroup"]
    ) -> None:
        """
        Validate that segment names are unique across all groups.
        This ensures that when we create boolean fields per segment,
        there are no naming conflicts.
        """
        all_segment_names = []
        for segment_group in segment_groups:
            for segment in segment_group.segments:
                all_segment_names.append(segment.name)

        unique_segment_names = set(all_segment_names)
        if len(unique_segment_names) != len(all_segment_names):
            raise ValueError("Segment names must be unique across all groups")
