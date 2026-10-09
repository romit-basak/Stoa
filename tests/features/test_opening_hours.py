import pytest

from src.features.opening_hours import parse_opening_hours


def hours(mask, day):
    """Open hours (0-23) on a weekday (0 = Monday)."""
    return [h for h in range(24) if mask[day * 24 + h]]


def test_always_open():
    assert parse_opening_hours("24/7").all()


def test_weekdays_and_saturday():
    m = parse_opening_hours("Mo-Fr 09:00-17:00; Sa 10:00-14:00")
    assert hours(m, 0) == list(range(9, 17))
    assert hours(m, 4) == list(range(9, 17))
    assert hours(m, 5) == list(range(10, 14))
    assert hours(m, 6) == []


def test_half_hour_boundaries_use_the_middle_of_the_hour():
    m = parse_opening_hours("Mo 09:30-10:00")
    assert hours(m, 0) == [9]  # open at 09:30
    m = parse_opening_hours("Mo 09:31-10:00")
    assert hours(m, 0) == []


def test_past_midnight_spills_into_next_day():
    m = parse_opening_hours("Fr 22:00-02:00; Sa 10:00-14:00")
    assert hours(m, 4) == [22, 23]
    assert hours(m, 5) == [0, 1, 10, 11, 12, 13]


def test_sunday_night_wraps_to_monday():
    m = parse_opening_hours("Su 20:00-03:00")
    assert hours(m, 6) == [20, 21, 22, 23]
    assert hours(m, 0) == [0, 1, 2]


def test_later_rule_replaces_earlier_days():
    m = parse_opening_hours("Mo-Su 08:00-20:00; Su off")
    assert hours(m, 6) == []
    assert hours(m, 5) == list(range(8, 20))


def test_lists_wraps_and_split_ranges():
    m = parse_opening_hours("Fr-Mo 10:00-12:00,13:00-15:00")
    for day in (4, 5, 6, 0):
        assert hours(m, day) == [10, 11, 13, 14]
    assert hours(m, 2) == []


def test_comma_used_between_rules():
    m = parse_opening_hours("Mo-Fr 08:00-18:00, Sa 09:00-12:00")
    assert hours(m, 5) == [9, 10, 11]


def test_public_holidays_are_ignored_not_fatal():
    m = parse_opening_hours("Mo-Fr,PH 09:00-17:00; PH off")
    assert hours(m, 0) == list(range(9, 17))


def test_open_end_and_no_day():
    assert hours(parse_opening_hours("17:00+"), 3) == list(range(17, 24))
    assert hours(parse_opening_hours("07:00-19:00"), 6) == list(range(7, 19))


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        float("nan"),
        "sunrise-sunset",
        "Jan-Mar Mo 09:00-12:00",
        '"by appointment"',
        "Mo 25:99-26:00",
    ],
)
def test_unsupported_forms_are_unknown(value):
    assert parse_opening_hours(value) is None
