"""
aeon-core/tests/test_temporal.py
--------------------------------
Unit tests for aeon_core.temporal: date parsing functions.
"""

import json
import numpy as np
import pytest

from aeon_core.temporal import (
    parse_date_to_decimal,
    extract_date_from_string,
    parse_dates_from_auspice_json,
)


class TestTemporalDateParsing:
    def test_parse_date_to_decimal_numeric(self):
        assert parse_date_to_decimal(2021.25) == 2021.25
        assert parse_date_to_decimal("2021.25") == 2021.25
        assert np.isnan(parse_date_to_decimal(1500.0))
        assert np.isnan(parse_date_to_decimal("unknown"))
        assert np.isnan(parse_date_to_decimal(None))

    def test_parse_date_to_decimal_iso(self):
        d_jan1 = parse_date_to_decimal("2021-01-01")
        assert abs(d_jan1 - 2021.0) < 0.01

        d_july = parse_date_to_decimal("2021-07-02")
        assert abs(d_july - 2021.5) < 0.02

        d_year = parse_date_to_decimal("2021")
        assert d_year == 2021.0

        d_slash = parse_date_to_decimal("2021/07/02")
        assert abs(d_slash - 2021.5) < 0.02

    def test_extract_date_from_string(self):
        assert abs(extract_date_from_string("isolate|2021-05-15") - 2021.37) < 0.02
        assert abs(extract_date_from_string("hCoV-19/USA/123/2021-05-15") - 2021.37) < 0.02
        assert abs(extract_date_from_string("isolate/2021.45") - 2021.45) < 0.01
        assert extract_date_from_string("isolate|2021") == 2021.0
        assert np.isnan(extract_date_from_string("just_a_name_no_date"))

    def test_parse_dates_from_auspice_json(self, tmp_path):
        mock_tree = {
            "tree": {
                "name": "NODE_ROOT",
                "children": [
                    {
                        "name": "tip_1",
                        "node_attrs": {"num_date": {"value": 2021.2}}
                    },
                    {
                        "name": "tip_2",
                        "node_attrs": {"date": {"value": "2022-01-01"}}
                    },
                    {
                        "name": "internal_node",
                        "children": [
                            {
                                "name": "tip_3|2023-06-01",
                                "node_attrs": {}
                            }
                        ]
                    }
                ]
            }
        }
        json_file = tmp_path / "mock_auspice.json"
        with open(json_file, 'w') as f:
            json.dump(mock_tree, f)

        dates = parse_dates_from_auspice_json(json_file)
        assert len(dates) == 3
        assert dates["tip_1"] == 2021.2
        assert abs(dates["tip_2"] - 2022.0) < 0.01
        assert abs(dates["tip_3|2023-06-01"] - 2023.41) < 0.02
