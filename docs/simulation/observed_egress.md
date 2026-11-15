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

## Walkers on 27 October

`src/simulation/walkers.py` follows steps 1 and 2 of getting_started.md. Walkers leave
from the graph nodes within 60 m of Rod Laver Arena, towards Flinders Street, Richmond,
Jolimont and Parliament stations, on 20 noisy shortest paths per destination. Walkers per
route are fitted with non-negative least squares to the hourly counts at 22:00 and 23:00
at sensors 161, 123, 124 and 75; 136 is left out as the test. Each walker's time at its
first sensor is drawn from sensor 161's event-exit minute profile, and later sensors
follow at 1.3 m/s. Settings are under `walkers:` in `configs/simulation.yaml`.

```bash
.venv/bin/python -m src.simulation.walkers --area melbourne_park --night 2026-10-27
```

| Sensor | In the fit | Observed | Simulated | Observed peak | Simulated peak | Correlation |
|---|---|---|---|---|---|---|
| 161 | Yes | 1,477 | 1,468 | 23:00 (452) | 23:00 (335) | 0.87 |
| 75 | Yes | 38 | 19 | 23:10 (9) | 23:10 (4) | 0.46 |
| 123 | Yes | 66 | 0 | 23:00 (28) | none | |
| 124 | Yes | 0 | 19 | none | 23:00 (4) | |
| 136 | No (test) | 335 | 0 | 23:00 (91) | none | |

- At 161 the walkers match the total and the peak block. The simulated peak is flatter,
  as getting_started.md expects from average profiles.
- No sampled route passes 136, 123 or most of the riverside paths: shortest paths on
  distance alone don't follow the river path the crowd uses after 161, so the test sensor
  can't be checked yet.
- Walkers on routes that pass no fitted sensor stay at the prior, so the total crowd
  (about 2,200 in the 23:00 hour) is rough.
- The minute profile is pooled from the same archive, including this night, so the
  timing at 161 is not an independent test.

## Next

1. A route cost with path type, slope, shade and crossings, so routes follow the river
   path, then check sensor 136.
2. Run the other concert nights, fitting on some and testing on others.
3. The same area in JuPedSim (microsim.py), with the walkers as input.
