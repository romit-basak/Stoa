from datetime import date

from src.ingest.weather import years_to_fetch

START = date(2009, 1, 1)


def test_fetches_everything_the_first_time():
    assert years_to_fetch(START, date(2026, 10, 2), set()) == list(range(2009, 2027))


def test_refetches_only_the_last_two_years():
    have = set(range(2009, 2027))
    assert years_to_fetch(START, date(2026, 10, 2), have) == [2025, 2026]


def test_fills_gaps_and_new_years():
    have = set(range(2009, 2026)) - {2015}
    assert years_to_fetch(START, date(2026, 1, 3), have) == [2015, 2025, 2026]


def test_full_refetches_all():
    have = set(range(2009, 2027))
    assert years_to_fetch(START, date(2026, 10, 2), have, full=True) == list(
        range(2009, 2027)
    )
