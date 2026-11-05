# Modelling plan

Decisions from the 5 November 2026 discussion on the model phase, and the order of work from here. The flow-model code is in progress on the `romit/flow-model` branch, so the sections on selection, gates, the registry and the training DAG describe planned work; its implementation details (commands, config keys, search spaces) will be in `docs/models/flow_model.md`. Experiment tracking conventions are in [experiment_tracking.md](experiment_tracking.md).

## What the course asks for

The course's model development guidelines ask for: code that loads the pipeline's versioned data; training several architectures and selecting the best on validation metrics; validation on data the model never trained on; bias detection by slicing (Fairlearn or TFMA) with mitigation; hyperparameter tuning with a documented search space; experiment tracking (MLflow); sensitivity analysis (SHAP or LIME); a final selection made after the bias check; a push to a registry; and CI/CD that trains, validates, checks bias, alerts and can roll back. The plan below covers each of these.

## The four models

All four predict the same thing: the pedestrian count on a segment in a given hour. Each is trained and scored with whole sensors held out (5 folds grouped by sensor, core rule 2), so every prediction we score comes from a model that never saw that sensor. Each held-out sensor's first 28 days of counts are kept back for calibration and not scored.

| # | Model | Architecture | Inputs | Output |
|---|---|---|---|---|
| 1 | Betweenness baseline | A ratio, nothing fitted | `met_betweenness_1600` for the segment; the training fold's mean count and mean betweenness | Training mean count × (segment betweenness ÷ training mean betweenness), scaled to the sensor's first 28 days |
| 2 | Space Syntax baseline | Ridge regression (`alpha=1.0`) on log counts | The `ang_*`, `met_*` and `junctions_*` network metrics at 400, 800 and 1600 m, plus hour and weekday | A count per hour, scaled to the sensor's first 28 days |
| 3 | Flow model, Melbourne-rich | `HistGradientBoostingRegressor`, Poisson loss | Every feature `src/features/schema.py` allows: time, weather, calendars, slope, shade, network metrics, amenities, transit, events, and the Melbourne-only layers (CLUE jobs, City businesses, canopy, footpath survey slope) | Predicted count, plus low and high values from two quantile models (0.1 and 0.9) |
| 4 | Flow model, portable | The same as 3 | Only features `schema.py` marks as portable: OSM, terrain, weather, calendars | The same as 3 |

Models 1 and 2 are the baseline ladder. They don't feed models 3 and 4; they set the bar those models have to clear. Their network metrics are also inputs to models 3 and 4, so the learned models can only win if the extra features help. The gap between models 3 and 4 shows what Melbourne's own data adds. Model 4 is the one used for NYC and Boston.

The ladder in `PROJECT_CONTEXT.md` has a third rung, betweenness on a graph with hand-set edge costs. It's deferred for now.

### How this differs from the bias model

`src/bias/slice_errors.py` trains the same kind of model (HGBR, Poisson loss, the same folds and features) as a stand-in, so bias slicing could start before the flow model existed. It uses fixed hand-picked settings, predicts a single value, isn't compared with any baseline, and is never saved or served. Once the flow model exists, bias slicing runs on the selected model's out-of-fold predictions and the stand-in retires.

## Model selection

Three gradient-boosting families compete, each with Poisson loss, on the same folds and the same feature matrix, for each variant:

- `hgb`: scikit-learn `HistGradientBoostingRegressor`
- `lgbm`: LightGBM, `objective="poisson"`
- `xgb`: XGBoost, `objective="count:poisson"`

Optuna (TPE sampler) can tune each family; the search spaces are in `configs/models.yaml`. Selection runs in two steps:

1. Rank candidates by `mae_log_cv`, the mean absolute error on log counts across the held-out-sensor folds. This is the team's headline success criterion.
2. For every candidate within a small tolerance of the best, compute WAPE per slice (precinct, time of day, rain, era, volume tier, and the other slices in `configs/bias.yaml`) with Fairlearn's `MetricFrame`. Pick the one whose worst slice is closest to its overall error.

The selected model gets its quantile models and is scored once on the held-out precincts (University of Melbourne edge, Carlton, North Melbourne), uncalibrated and calibrated on one or two of each precinct's sensors, with the UniMelb-edge results split at the Metro Tunnel opening (30 November 2025). Holdout scores are reported and never used to choose. SHAP values for the selected model go into the run as the sensitivity analysis.

## Gates, registry and rollback

A model reaches the registry only if:

- its `mae_log_cv` is at least 15% lower than `baseline_space_syntax` on the same data version (the run's `stoa.dvc.processed` tag), and
- its 80% intervals cover 75 to 85% of held-out hours (`coverage80_cv`).

The registry uses MLflow aliases, because MLflow 3 deprecates stages. `champion` is the served version and `previous` the one it replaced; rollback moves `champion` back to `previous`. If the learned model doesn't beat Space Syntax, the weeks 5 to 6 checkpoint in CLAUDE.md applies.

## Tracking and orchestration

Every run goes through `src.models.tracking.run`, so it carries the git commit, the DVC data version and the training frame's spec. All flow-model runs go to the `stoa-flow` experiment, with run names like `hgb_rich`, `lgbm_portable` and `baseline_space_syntax`, and metrics named `wape_cv`, `mae_cv`, `rmse_cv`, `mae_log_cv`, `coverage80_cv` and `wape_fold0` to `wape_fold4`.

Training has its own DAG, `stoa_train`, run monthly or by hand: tuning (optional), candidates and baselines, selection, the gate, and registration. The daily `stoa_pipeline` doesn't train. Model outputs go to `data/models/`, which DVC doesn't track, so training never changes a data snapshot.

## Tier 2: the GNN

A spatio-temporal graph neural network is the first Tier 2 model. It joins the `stoa-flow` experiment as one more candidate, with the same folds, gates and selection, and is adopted only if it beats the selected gradient-boosted model on held-out sensors. Two things make that hard. Scoring on unseen sensors makes it a problem of filling the gaps between about 150 points, where GNNs often struggle. And NYC has only two days of totals a year, so there is no time series to run it on for the cross-city test.

An LSTM isn't planned. It forecasts from a sensor's own recent counts, so it can't predict segments without a sensor, other cities, or what-if scenarios. Its one use, live forecasting at sensors during an event, is real-time streaming, which is Tier 3.

## Simulation

The simulation has two layers, and neither is a learned model.

### Route-sampling walkers (our code)

The flow model's segment counts don't say where people start and end, which way they go, or when within the hour they pass. Its segment predictions are also made independently of each other, so people aren't conserved at junctions. The walkers supply both:

1. Origins (entrances, tram and train stops, car parks, event gates) and destinations (shops, cafés, toilets, offices), weighted by attraction and whether they're open that hour.
2. Route choice on a cost (distance, slope, shade, crossings), with some randomness so walkers don't all take the shortest path.
3. Trip tables fitted so the walkers' summed traffic on each segment matches the flow model's predicted counts, with standard methods for estimating trips from link counts, such as iterative proportional fitting.
4. Departure times within the hour from `minute_profiles.parquet` (by volume tier, day type and hour type, including event phases), and direction splits from the counts' `direction_1` and `direction_2`.

The flow model's three outputs are used like this:

| Output | Use |
|---|---|
| Predicted count | The fitting target for the main scenario |
| Low and high (0.1 and 0.9) | Weights in the fit, so segments with wide intervals count less; and the spread of a set of runs, each drawing its targets within the intervals |

Each run in that set draws one citywide day factor plus local noise. Setting every segment to its high value at once would describe a day much rarer than one in ten, because the intervals are computed segment by segment. The intervals also cover hourly uncertainty only; bunching within the hour adds more.

What-if scenarios edit the graph, recompute the network metrics nearby, re-predict counts with the flow model, and re-route the walkers. The sanity scenario required before deploy (closing a path lowers its flow and raises flow on the alternatives) falls out of the walkers, which conserve people.

### Microsimulation (an existing open-source simulator)

For limited outdoor approach areas only, the walkers' entry points, times and routes feed an existing open-source pedestrian simulator, not one we write. The candidates in the scoping document are JuPedSim and Vadere. Both ship several built-in movement models (social-force and speed-based models among them), and we'd pick one and calibrate it rather than build our own. Setup on one area is scheduled for week 7 (9 to 15 November).

Before choosing, check each simulator's licence against the rule that code dependencies be permissive (MIT, BSD, Apache), since the API image may be distributed. If the best simulator has a copyleft licence, one option is to run it as a separate service outside the distributed image; that is a team decision.

Movement from one time step to the next comes from the simulator's rules: each person follows a route at a speed that falls as density rises. Time-series data checks those rules rather than driving them:

| Data | What it checks |
|---|---|
| Per-minute counts at Melbourne sensors | When walkers pass each sensor: peak timing, peak sharpness, and the delay between sensors along an approach route. Event egress (for example after a Marvel Stadium match) is the strongest test. A few settings, such as departure spread after a match and walking speed, are fitted on some events and tested on others. |
| Jülich controlled experiments | Speed against density, and flow through bottlenecks (availability and licence still to verify) |
| Stanford Drone trajectories | Route choice. CC BY-NC-SA 3.0: validation only, and no model fit to it ships (core rule 5) |

The per-minute archive starts in late October 2026, so few events are covered yet, and some sensors report only every 5 minutes. We observe crowds only at fixed sensor points, so the simulation can be checked on when and how many people pass each sensor, not on where everyone is at once. Simulation output is planning guidance for outdoor approach areas, never a crowd-safety result (core rule 7).

## Order of work

1. Finish the flow-model branch: candidates, selection after the bias check, gates, aliases, the `stoa_train` DAG, tests, and one full run on real data.
2. Point the bias report at the selected model's out-of-fold predictions.
3. Run the calibration curve (calibrate on 3, 5, 10 and 20 sensors) and the precinct transfer test with the selected model.
4. Run the portable model on NYC's 114 count locations.
5. Week 7: check the simulators' licences, choose one, and set up one outdoor area; build the walker layer.
6. Week 9: validate the microsimulation against the experiment data and the per-minute event counts.
7. Tier 2, once Tier 1 is solid: the GNN as a `stoa-flow` candidate.

## Open decisions

- Which simulator (JuPedSim, Vadere or another), after the licence check, and how to run it if its licence is copyleft.
- Whether to add the hand-weighted ladder rung later.
- The tolerance used in step 2 of model selection.
