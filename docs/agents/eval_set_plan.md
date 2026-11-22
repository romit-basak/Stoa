# Building the agent evaluation set

The intake agent's evaluation set (`tests/agent_eval/`, format in [llm_agents.md](llm_agents.md), "Evaluation") needs about 100 to 150 cases. Writing them all by hand is slow, so most come from data whose correct answers are already known, and people write only the hard cases.

No public dataset fits as it stands: none has our sites' places, our tools or our rules. The ones below help in two ways. Some screen models for general tool use and spatial reasoning. Others supply real event text whose answers come from structured columns.

## Datasets

| Dataset | What it covers | How we'd use it | Licence |
|---|---|---|---|
| [BFCL](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) (Berkeley Function Calling Leaderboard) | Tool calling. Its multi-turn files include `miss_param` (a required argument is missing) and `miss_func` (no tool fits) | Screening models on asking instead of inventing, and on calling tools in the right order | Apache 2.0 |
| [GeoBenchX](https://github.com/Solirinai/GeoBenchX) | 202 multi-step GIS tasks over 23 tools, some deliberately unsolvable | Screening spatial tool use and declining impossible requests | To check in the repo |
| [MapEval](https://arxiv.org/abs/2501.00316), API subset | 300 map questions answered through map tools | Screening general spatial reasoning with tools | No dataset licence found, and the data comes from Google Maps. Internal comparison only |
| [NYC Permitted Event Information](https://data.cityofnewyork.us/City-Government/NYC-Permitted-Event-Information/tvpp-9vvx) | Real permitted events: name, type, start and end times, and locations such as "EAST 17 STREET between PARK AVENUE SOUTH… \| South \| Full Street Closure" | Real event text with known answers (below) | NYC Open Data, no use restrictions |

BFCL, GeoBenchX and MapEval measure general ability. They help choose a model ([build_plan.md](build_plan.md), Part 2, step 4) but stay outside our acceptance set.

The NYC events don't conflict with the rule never to train on NYC. That rule covers the NYC DOT pedestrian counts behind the flow model's transfer test, a different dataset for a different model. Check `docs/data_licenses.md` and add a data card before the events are pulled.

## Where the cases come from

| Source | Cases | Human effort |
|---|---|---|
| NYC permitted events | about 40 | Review the wording |
| Melbourne: `configs/events.yaml` and the AFL fixtures | about 30 | Review the wording |
| Generated from the gazetteer (Melbourne, NEU Boston) | about 60 | Spot-check a sample |
| Hand-written hard cases | 20 to 30 | Write them, about 5 per person |

### Real events with known answers

- **NYC.** Each permitted event becomes a description in a planner's words, for example "We're closing the south side of East 17th between Park Ave South and Broadway from 8 to 6." The expected fields and closed segments come from the row's columns, so the labels cost nothing. The street segments still need matching to our NYC graph, by code.
- **Melbourne.** The 63 sourced events in `configs/events.yaml` and the AFL fixtures already give each event's venue, date, start time and crowd. Each becomes a description with its answers known. Some leave fields out on purpose, so the expected behaviour is to ask.

### Generated place cases

For questions like "which gate is to the north?", cases are generated with labels that are right by construction:

1. Code picks a venue and a target from the gazetteer, such as the gate whose bearing is north, and records the expected tool calls and place IDs.
2. A template turns them into a sentence: "close the {direction} gate at {venue} after the match".
3. An LLM rewords the sentence. A check confirms the new wording still names the same places and direction. The label never comes from the LLM.
4. Ambiguity is generated on purpose: two gates facing north (the agent should ask), aliases, landmarks ("the gate by the station"), and requests no tool can serve.

The rewording uses a different model from the one being scored, because a model tends to do better on its own phrasing.

### Hand-written hard cases

These are the cases generation can't produce well:
- local nicknames ("the Kenmore side");
- corrections over several turns ("no, the other one");
- traps: crowd numbers before any run, safety guarantees, setting an adjustment's strength;
- messy real phrasing.

## Reporting

The hand-written cases are scored and reported on their own as well as in the total. They're the closest to real users, and the generated cases shouldn't hide how the model does on them.
