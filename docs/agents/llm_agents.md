# LLM agents

Stoa uses a language model in two places: an intake interface that turns a planner's description into structured inputs and narrates finished results (Tier 1), and a planning assistant that runs scenarios by calling deterministic tools (Tier 2). Neither exists in code yet. This document is the design to build against; schemas marked "proposed" need the team's sign-off before they're frozen in `src/agents/`.

For how the agents fit with the models, see [modelling_plan.md](../models/modelling_plan.md). Background is in `docs/PROJECT_CONTEXT.md` sections 6.3 and 9.

## Rules every LLM call follows

These come from CLAUDE.md's core rules and apply to both agents.

1. The LLM never computes, estimates or invents a count, a route, a location, or a claim about the network (rule 1). It reads text and writes text. Every number and network claim in its output must trace to a model output or a graph query, or the output is blocked.
2. Missing values are asked for or filled with a visibly labelled default, and every field records where it came from: the user's text, a default, or public data (rule 3).
3. An ambiguous place goes to the map for the user to confirm. The LLM never picks one silently (rule 4).
4. Local knowledge becomes a labelled scenario edit whose strength the user picks from fixed levels. The LLM never sets the value (rule 11).
5. Wording: forecasts and simulations are planning guidance, never a crowd-safety guarantee (rule 7). Amenity what-ifs are labelled rough guidance, and exposure scores are visible impressions, not customers (rule 8). Uncertainty is shown with every forecast.
6. Every result the LLM narrates can also be produced without it: the form-based path and the template narration below give the same specs and the same numbers.
7. The LLM is called for intake and for the final narration only, never on each map interaction, so cost and latency stay low.
8. A customer's text and uploaded data stay scoped to that customer's site (rule 6). No personal data is collected.

## Where the agents sit

```
Planner's text ──► INTAKE AGENT ──► event spec + adjustment specs (each field sourced)
                     │   asks for missing required fields
                     │   sends ambiguous places to the map for confirmation
                     ▼
                Schema validation (src/agents/*_spec.py)  ◄── or the same specs from a form
                     ▼
     Deterministic code: spec → event features, graph edits, adjustments
                     ▼
          Flow model → walkers → microsimulation   (no LLM here)
                     ▼
                Results bundle (numbers, segment IDs, intervals, caveats)
                     ▼
     NARRATION (LLM) ──► grounding check ──► shown to the user
                     └─ fails the check ──► template narration instead
```

The planning assistant (Tier 2) replaces the single pass above with a loop. It reads the user's request, calls tools that edit and rerun the scenario, and narrates the tools' results through the same grounding check.

## Part 1: the intake interface (Tier 1)

### Event spec (proposed)

The event spec matches the shape of `configs/events.yaml`, so an event typed in by a planner and an event from the hand-kept log produce the same features. Required fields follow the event-planner flow in `PROJECT_CONTEXT.md` section 9.

| Field | Type | Required | Notes |
|---|---|---|---|
| `site` | site id | yes | Set by the map the user is working on, not by the LLM |
| `kind` | one of the kinds in `configs/events.yaml` | yes | football, cricket, concert, and so on |
| `venue` | OSM object | yes | Confirmed on the map. Named venues resolve through the `venues` table in `configs/events.yaml` |
| `date` | local date | yes | |
| `gates_open`, `start`, `end` | local times | yes | Relative times ("90 minutes before") are resolved by code from `start`, not by the LLM |
| `expected_crowd` | integer, or a category such as `sellout` | yes | A category needs the venue's capacity from public data; if there is none, ask |
| `gates` | list of OSM objects or map points | yes | Confirmed on the map |
| `arrival_modes` | shares by mode | no | Default: labelled site default |
| `parking`, `rideshare` | map points | no | |
| `closures`, `barriers` | segments | no | Confirmed on the map. Become graph edits |
| `fan_zones`, `vendors` | map points | no | Food trucks, merchandise |

Every field is stored as `{value, source, evidence}`. `source` is `user_text`, `default` or `public_data`. `evidence` is the exact span of the user's text the value came from, or the name of the default or dataset. A value with `source: user_text` and no matching span in the text fails validation, which catches the model inventing a value.

### Adjustment spec (proposed)

A local-knowledge statement such as "fans avoid the underpass after dark" becomes:

| Field | Notes |
|---|---|
| `statement` | The user's words, verbatim |
| `segments` | The segments it applies to, confirmed on the map |
| `conditions` | When it applies: hours, day type, event phase or weather, from a fixed vocabulary |
| `direction` | `fewer` or `more` people |
| `strength` | `slight`, `moderate` or `strong`, chosen by the user in the UI. The LLM output has no strength field |
| `effect` | Looked up by code from the level and a table of typical effect sizes measured in Melbourne (closures, new station entrances, event days). Not set by the LLM or the user |

The adjustment multiplies the base prediction on the matching segments and hours. It's shown as a labelled assumption with a with/without comparison, and later counts at the site update it through calibration. The effect-size table still has to be derived from the data.

### Conversation flow

1. The user describes the event. The LLM returns the specs through structured output (a tool call or a JSON schema), never free text that code then parses.
2. Code validates the output against the schema and the evidence spans. If validation fails, the LLM gets the errors and one more try; after that the form opens with the valid fields filled in.
3. For each missing required field, the agent asks one short question. Missing optional fields get a labelled default.
4. Each place mention becomes a list of candidate map objects from a deterministic lookup (OSM names, the venue table, nearby segments). The user confirms one on the map. The LLM never picks a candidate.
5. The user reviews the spec, with each field's source shown, and runs the scenario.

### Narration and the grounding check

After a run, code builds a results bundle: per-segment and per-period predictions with intervals, differences between scenarios, hotspots, the labels and caveats that apply, and the segment and place names involved. The LLM writes a short summary from the bundle only.

The grounding checker (`src/agents/grounding.py`, deterministic) then:

- pulls every number from the text and matches it to a bundle value, allowing only the rounding the template uses;
- matches every place, segment and comparison ("busier than", "the main route") to a bundle entry or a graph query result;
- checks the required caveats are present and no banned wording appears (safety guarantees, "customers" for exposure, unlabelled amenity what-ifs).

Anything that fails blocks the text, and the user sees the template narration built from the same bundle. Every block is logged for monitoring.

### Fallbacks

The form-based input produces the same specs, and the template narration produces the same numbers. A scenario never depends on the LLM being available, and the cached replay mode for demos needs no live calls.

## Part 2: the planning assistant (Tier 2)

The assistant lets a planner work in plain language: "What happens if we close Jersey Street and open gate C after the match?" It plans the steps, calls tools, and narrates the differences. It's Tier 2, so it starts only after the intake interface works.

### Tools (proposed)

Every tool is a deterministic function that already exists for the API or the app. The assistant can only do what these tools do.

| Tool | What it does |
|---|---|
| `get_scenario` | The current scenario: event spec, adjustments, edits |
| `find_places(text)` | Candidate map objects for a place mention, for the user to confirm |
| `edit_graph(...)` | Close or reopen a segment, add an entrance or exit, add a barrier (on confirmed objects only) |
| `set_event_fields(...)` | Change event spec fields, with the same sourcing and validation as intake |
| `propose_adjustment(...)` | Draft an adjustment spec; the UI then asks the user for its strength |
| `run_prediction(scenario)` | Run the flow model on the scenario |
| `run_walkers(scenario)` | Run the walker simulation; microsimulation only on the user's request, because of its cost |
| `compare(a, b)` | Differences between two scenarios' results, as a results bundle |
| `query_graph(...)` | Routes, distances and nearby segments, for network claims |
| `explain(segment, hour)` | Precomputed SHAP contributions for a prediction |

### Guardrails

- Tools that change the scenario show the change and wait for the user's confirmation before running.
- The assistant can't set an adjustment's strength or confirm a place; those actions only come from the UI.
- Every narration goes through the same grounding check, against the tool results of that turn.
- A turn has a budget of tool calls and of model runs. Microsimulation never starts without an explicit request.
- Each turn's tool calls are logged with the scenario, so any answer can be replayed without the LLM.

## Evaluation

The evaluation set lives in `tests/agent_eval/`, one JSON line per case:

```json
{"id": "marvel-001", "site": "melbourne", "text": "Saturday's game at Marvel, 7:25 start, gates open 90 minutes before...",
 "expected": {"kind": "football", "venue": "marvel", "date": "...", "start": "19:25", "gates_open": "17:55"},
 "expected_questions": ["expected_crowd"], "expected_places": ["marvel"]}
```

Good sources for cases are past AFL games at Marvel Stadium and the events in `configs/events.yaml`, written the way a planner would describe them, plus cases with missing fields, ambiguous places and contradictions. The cases need writing by a person; they're due in week 6 of the schedule.

| Measure | Target (team criteria, October 2026) |
|---|---|
| Field-level extraction accuracy | at least 90% |
| Location placement accuracy | reported |
| Grounding violations that reach the user | zero |

Further checks:
- adjustment parsing (segments, conditions, direction), and the Metro Tunnel test: enter the opening as a planner statement and check whether predictions after the opening improve;
- for the planning assistant (Tier 2), task success on scripted requests, correct tool calls, and the same grounding rule.

## Monitoring

Per the MLOps plan: how often users correct intake fields, grounding blocks per narration, tool-call and token cost per scenario, and latency. A rise in corrections or blocks on one field points to a prompt or schema problem.

## Files to create

| Path | Contents |
|---|---|
| `src/agents/event_spec.py` | Event spec schema and validation (the CLAUDE.md interface) |
| `src/agents/adjustment_spec.py` | Adjustment spec schema and the strength-to-effect lookup |
| `src/agents/intake.py` | The intake conversation |
| `src/agents/places.py` | Candidate lookup for place mentions |
| `src/agents/grounding.py` | The grounding checker |
| `src/agents/narrate.py` | Results bundle, LLM narration and template narration |
| `src/agents/tools.py` | Planning-assistant tools (Tier 2) |
| `tests/agent_eval/` | The evaluation set and its scorer |

## Open decisions

- Which LLM provider and model, and how prompts and responses are logged without storing customer data beyond its site.
- Freezing the event spec and adjustment spec fields above.
- The effect-size table behind slight, moderate and strong.
- Whether document upload (event plans, traffic management PDFs) stays Tier 2.
