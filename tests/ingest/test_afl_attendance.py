from src.ingest.afl_attendance import parse_season

ROW = (
    "<td>Thu 02-Apr-2015 7:50 PM (7:50 PM) <b>Att: </b>83,493 <b>Venue:</b> "
    '<a href="../venues/mcg.html">M.C.G.</a></td>'
)
NO_CROWD = (
    "<td>Sat 30-May-2020 7:40 PM <b>Venue:</b> "
    '<a href="../venues/docklands.html">Docklands</a></td>'
)
TOTAL = "<b>Tot Att: </b>359,470 (39,941)</td>"


def test_parse_season_reads_games_and_skips_round_totals():
    rows = parse_season(ROW + TOTAL + NO_CROWD, 2015)
    assert rows[0] == {
        "date": "2015-04-02",
        "local_time": "7:50 PM",
        "melbourne_time": "7:50 PM",
        "attendance": "83493",
        "venue": "M.C.G.",
        "season": "2015",
    }
    assert rows[1]["attendance"] == "" and rows[1]["venue"] == "Docklands"
    assert len(rows) == 2
