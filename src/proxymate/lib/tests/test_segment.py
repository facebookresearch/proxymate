# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from unittest import TestCase

from proxymate.lib.segment import Segment, SegmentGroup


class TestSegment(TestCase):
    def test_valid_segment_names(self) -> None:
        """Test that valid segment names are accepted."""
        valid_names = [
            "country_US",
            "optin_status_optin",
            "user_type",
            "_private_segment",
            "segment123",
            "SEGMENT_NAME",
            "a",
            "_",
            "segment_with_numbers_123",
        ]

        for name in valid_names:
            # Should not raise any exception
            segment = Segment(name, "country = 'US'", ["country"])
            self.assertEqual(segment.name, name)

    def test_invalid_segment_names_with_spaces(self) -> None:
        """Test that segment names with spaces are rejected."""
        invalid_names_with_spaces = [
            "country US",
            "optin status",
            "user type segment",
            " leading_space",
            "trailing_space ",
            "middle space here",
            "multiple   spaces",
        ]

        for name in invalid_names_with_spaces:
            with self.assertRaises(ValueError) as context:
                Segment(name, "country = 'US'", ["country"])
            self.assertIn("cannot contain spaces", str(context.exception))

    def test_invalid_segment_names_bad_format(self) -> None:
        """Test that segment names with invalid SQL identifier format are rejected."""
        invalid_names_no_spaces = [
            "123segment",  # starts with number
            "segment-name",  # contains hyphen
            "segment.name",  # contains dot
            "segment@name",  # contains special character
            "segment+name",  # contains plus
            "segment(name)",  # contains parentheses
        ]

        for name in invalid_names_no_spaces:
            with self.assertRaises(ValueError) as context:
                Segment(name, "country = 'US'", ["country"])
            self.assertIn("not a valid SQL field name", str(context.exception))

    def test_empty_segment_name(self) -> None:
        """Test that empty segment names are rejected."""
        with self.assertRaises(ValueError) as context:
            Segment("", "country = 'US'", ["country"])
        self.assertIn("cannot be empty", str(context.exception))

    def test_segment_name_validation_error_messages(self) -> None:
        """Test that validation error messages are descriptive."""
        # Test space error message
        with self.assertRaises(ValueError) as context:
            Segment("country US", "country = 'US'", ["country"])
        self.assertEqual(
            str(context.exception), "Segment name 'country US' cannot contain spaces"
        )

        # Test invalid format error message
        with self.assertRaises(ValueError) as context:
            Segment("123invalid", "country = 'US'", ["country"])
        self.assertIn("123invalid", str(context.exception))
        self.assertIn("not a valid SQL field name", str(context.exception))
        self.assertIn("must start with a letter or underscore", str(context.exception))

    def test_segment_functionality_still_works(self) -> None:
        """Test that existing segment functionality still works after validation."""
        segment = Segment(
            "country_US",
            "country = 'US' AND status != 'inactive'",
            ["country", "status"],
        )

        self.assertEqual(segment.name, "country_US")
        self.assertEqual(segment.get_columns(), ["country", "status"])
        self.assertEqual(
            segment.get_where_clause(), "country = 'US' AND status != 'inactive'"
        )

    def test_check_segment_unique_names(self) -> None:
        """Test that segment names must be unique."""
        segment1 = Segment("country_US", "country = 'US'", ["country"])
        segment2 = Segment("age_over_18", "age > 18", ["age"])
        segment3 = Segment(
            "country_US", "country = 'US'", ["country"]
        )  # Duplicate name

        # No exception when all names are unique
        Segment.check_segment_unique_names([segment1, segment2])

        # Exception when there are duplicate names
        with self.assertRaises(ValueError) as context:
            Segment.check_segment_unique_names([segment1, segment2, segment3])
        self.assertIn("unique", str(context.exception))

    def test_get_segment_columns(self) -> None:
        """Test that get_segment_columns returns all unique columns from
        multiple segments."""
        segment1 = Segment("country_US", "country = 'US'", ["country"])
        segment2 = Segment(
            "active_US_users",
            "country = 'US' AND status = 'active'",
            ["country", "status"],
        )
        segment3 = Segment("high_score", "score > 90", ["score"])

        columns = Segment.get_segment_columns([segment1, segment2, segment3])
        # Order might vary since it's using a set internally
        self.assertEqual(set(columns), {"country", "status", "score"})

    def test_where_clause(self) -> None:
        """Test that where clause works correctly."""
        # Create segment with where clause
        segment = Segment(
            "custom_segment",
            "(age > 25 AND status IN ('active', 'pending')) OR score >= 80",
            ["age", "status", "score"],
        )

        # Check that the where clause is correctly generated
        expected_where_clause = (
            "(age > 25 AND status IN ('active', 'pending')) OR score >= 80"
        )
        self.assertEqual(segment.get_where_clause(), expected_where_clause)

    def test_where_clause_with_disallowed_operators(self) -> None:
        """Test that where clause with disallowed operators raises an error."""
        # Create segment with where clause containing disallowed operator (LIKE)

        # Check that using the where clause raises an error
        with self.assertRaises(ValueError) as context:
            _ = Segment(
                "invalid_segment",
                "age > 25 AND status ='optin' AND status LIKE '%active%'",
                ["age", "status"],
            )

        self.assertIn("Disallowed operator", str(context.exception))
        self.assertIn("%", str(context.exception))

    def test_where_clause_with_lowercase_syntax(self) -> None:
        """Test that where clause with lowercase SQL syntax is accepted."""
        # Create segment with where clause using lowercase SQL keywords
        segment = Segment(
            "lowercase_syntax",
            "age > 25 and status in ('active', 'pending') or score >= 80",
            ["age", "status", "score"],
        )

        # Check that the where clause is correctly accepted
        self.assertEqual(
            segment.get_where_clause(),
            "age > 25 and status in ('active', 'pending') or score >= 80",
        )

    def test_where_clause_with_boolean_literals(self) -> None:
        """Test that where clause with TRUE and FALSE boolean literals is accepted."""
        # Create segment with where clause using TRUE and FALSE
        segment = Segment(
            "boolean_literals",
            "is_active = TRUE AND is_verified = FALSE",
            ["is_active", "is_verified"],
        )

        # Check that the where clause is correctly accepted
        self.assertEqual(
            segment.get_where_clause(),
            "is_active = TRUE AND is_verified = FALSE",
        )

        # Test with lowercase true and false
        segment = Segment(
            "boolean_literals_lowercase",
            "is_active = true AND is_verified = false",
            ["is_active", "is_verified"],
        )

        # Check that the where clause is correctly accepted
        self.assertEqual(
            segment.get_where_clause(),
            "is_active = true AND is_verified = false",
        )

    def test_missing_column_in_columns_list(self) -> None:
        """Test what happens when a column used in where_clause is not in the
        columns list."""
        # Create segment with a column in where_clause that's not in the columns list
        with self.assertRaises(ValueError) as context:
            Segment(
                "missing_column",
                "age > 25 AND status = 'active' AND country = 'US'",
                # Intentionally omitting 'country' from the columns list
                ["age", "status"],
            )

        self.assertIn("column", str(context.exception).lower())
        self.assertIn("country", str(context.exception))


class TestSegmentGroup(TestCase):
    def test_segment_group_initialization(self) -> None:
        """Test that SegmentGroup initializes correctly."""
        segment1 = Segment("country_US", "country = 'US'", ["country"])
        segment2 = Segment("age_over_18", "age > 18", ["age"])

        group = SegmentGroup("test_group", [segment1, segment2])
        self.assertEqual(group.name, "test_group")
        self.assertEqual(len(group.segments), 2)
        self.assertEqual(group.segments[0].name, "country_US")
        self.assertEqual(group.segments[1].name, "age_over_18")

    def test_segment_group_with_duplicate_segment_names(self) -> None:
        """Test that SegmentGroup raises an error when initialized with segments
        with duplicate names."""
        segment1 = Segment("country_US", "country = 'US'", ["country"])
        segment2 = Segment("country_US", "country = 'US'", ["country"])

        with self.assertRaises(ValueError) as context:
            SegmentGroup("test_group", [segment1, segment2])
        self.assertIn("unique", str(context.exception))

    def test_get_segment_columns(self) -> None:
        """Test that get_segment_columns returns all unique columns from segments
        in the group."""
        segment1 = Segment("country_US", "country = 'US'", ["country"])
        segment2 = Segment(
            "active_US_users",
            "country = 'US' AND status = 'active'",
            ["country", "status"],
        )

        group = SegmentGroup("test_group", [segment1, segment2])
        columns = group.get_segment_columns()
        # Order might vary since it's using a set internally
        self.assertEqual(set(columns), {"country", "status"})

    def test_check_groups_unique_identifiers(self) -> None:
        """Test that segment group names must be unique."""
        segment1 = Segment("country_US", "country = 'US'", ["country"])
        segment2 = Segment("age_over_18", "age > 18", ["age"])

        group1 = SegmentGroup("group1", [segment1])
        group2 = SegmentGroup("group2", [segment2])
        group3 = SegmentGroup("group1", [segment2])  # Duplicate name

        # No exception when all names are unique
        SegmentGroup.check_groups_unique_identifiers([group1, group2])

        # Exception when there are duplicate names
        with self.assertRaises(ValueError) as context:
            SegmentGroup.check_groups_unique_identifiers([group1, group2, group3])
        self.assertIn("unique", str(context.exception))

    def test_check_segments_unique_names_across_groups(self) -> None:
        """Test that segment names must be unique across all groups."""
        segment1 = Segment("segment1", "country = 'US'", ["country"])
        segment2 = Segment("segment2", "age > 18", ["age"])
        segment3 = Segment("segment3", "status = 'active'", ["status"])
        segment4 = Segment("segment1", "score > 90", ["score"])  # Duplicate name

        group1 = SegmentGroup("group1", [segment1, segment2])
        group2 = SegmentGroup("group2", [segment3])
        group3 = SegmentGroup(
            "group3", [segment4]
        )  # Contains segment with duplicate name

        # No exception when all segment names are unique across groups
        SegmentGroup.check_segments_unique_names_across_groups([group1, group2])

        # Exception when there are duplicate segment names across groups
        with self.assertRaises(ValueError) as context:
            SegmentGroup.check_segments_unique_names_across_groups(
                [group1, group2, group3]
            )
        self.assertIn("unique", str(context.exception))
