import pandas as pd

from src.analysis.new_openings import load_analysis_config, property_jumps


def test_property_jumps_counts_rows_and_sums_values():
    df = pd.DataFrame(
        {
            "base_property_id": ["a"] * 3 + ["a"] * 15 + ["b"] * 2,
            "year": [2013] * 3 + [2014] * 15 + [2013, 2014],
            "number_of_seats": [10] * 18 + [100, 500],
        }
    )
    est = property_jumps(df, None, 10)
    assert est[["base_property_id", "year", "gain"]].values.tolist() == [
        ["a", 2014, 12.0]
    ]

    seats = property_jumps(df, "number_of_seats", 300)
    assert seats[["base_property_id", "year", "gain"]].values.tolist() == [
        ["b", 2014, 400.0]
    ]


def test_new_openings_config_is_complete():
    p = load_analysis_config()["new_openings"]
    assert p["sensor_radius_m"] > 0 and 0 < p["min_year_completeness"] <= 1
    assert set(p["attractor_divisions"]) >= {"42", "45"}
