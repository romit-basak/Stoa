# Observed egress at Melbourne Park

The first step of the simulation in [getting_started.md](getting_started.md): the observed
series that the walkers and JuPedSim are compared with.

`src/simulation/observed.py` takes each concert at Rod Laver Arena and builds the counts at
sensors 161, 136, 123, 124 and 75 in 5-minute blocks and by direction, from 2.5 to 5 hours
after the advertised start (no end times are published). Settings are in
`configs/simulation.yaml`; outputs go to `data/simulation/melbourne_park/`
(`observed_egress.parquet` and `observed_summary.csv`).

```bash
.venv/bin/python -m src.simulation.observed --area melbourne_park
```

## Results

On the archive to 11 November (8 of the 14 concerts listed):

| Night | Concert | Total at 161 | Peak block at 161 | Peak count at 161 | Total at 136 |
|---|---|---|---|---|---|
| 25 Oct | Hans Zimmer Live | 634 | 23:10 | 261 | 22 |
| 26 Oct | Hilary Duff | 994 | 22:50 | 307 | 199 |
| 27 Oct | Hilary Duff | 1,477 | 23:00 | 452 | 335 |
| 28 Oct | Lily Allen | 1,448 | 22:25 | 485 | 99 |
| 29 Oct | Lily Allen | 1,271 | 22:25 | 406 | 106 |
| 2 Nov | 5 Seconds of Summer | 1,028 | 23:15 | 272 | 177 |
| 8 Nov | Daniel Caesar | 32 | 21:50 | 4 | 3 |
| 9 Nov | Daniel Caesar | 35 | 21:30 | 4 | 5 |

- Sensor 161 carries the crowd (1,000 to 1,500 a night); 136 sees about a tenth to a
  quarter of it, peaking in the same block or 5 minutes later. 123 and 75 see 13 to 66
  people and 124 almost none, so most of the crowd leaves the route after 161. 161 is the
  main validation sensor and 136 the second; peak delays at the others are noise.
- 27 October matches the 1,417 people at 161 between 22:50 and 23:30 noted in
  getting_started.md.
- The Daniel Caesar nights show almost no crowd; their entries in `configs/events.yaml`
  need checking before they're used.
- 25 October and 28 October had another event at Melbourne Park the same night.
- None of the five sensors are among the eight that skip minutes :00 to :04.

## Next

Walkers fitted to these nights, then JuPedSim on the same area, both compared with these
series at sensors 161 and 136.
