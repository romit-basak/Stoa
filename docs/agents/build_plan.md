# Building the LLM agent: model choice and plan

This plan builds the intake agent designed in [llm_agents.md](llm_agents.md), and the tool loop the planning assistant will reuse. It covers which model to use, the infrastructure to build, and how to get the model's behaviour right, especially its handling of places ("I'm closing the gate to the north"). How the evaluation set is built is in [eval_set_plan.md](eval_set_plan.md).

The model choice below is a recommendation. The provider and model stay an open decision for the team (llm_agents.md, "Open decisions").

## The approach

The model turns language into structured requests. Code does everything that needs knowledge of the site or arithmetic. For "the gate to the north", the model produces a place query (`kind: gate`, `relation: north_of`, `anchor: venue`), and code finds which gates match, from a precomputed table of the site's places. That's what core rules 1 and 4 require anyway: the LLM never invents a location, and the user confirms every place on the map.

With the geometry in code, an off-the-shelf model is enough, and the model choice comes down to cost and scores on the evaluation set. Fine-tuning can't teach a model where the gates are on a site it hasn't seen, so it's a last resort (step 5 of Part 2).

## Which model

Prices checked on 21 November 2026. Per-conversation costs assume about 5 model calls per intake conversation, about 25,000 input and 2,000 output tokens in all; they're rough.

| Option | Price | Per conversation | Notes |
|---|---|---|---|
| Gemini 3.8 Flash, Gemini API | $0.75 in, $3.75 out per million tokens until 31 December 2026, then twice that | about 2.5 cents (5 cents from January) | The strongest tool use out of the box |
| Gemini 3.1 Flash-Lite, Gemini API | $0.25 in, $1.50 out | about 1 cent | Worth trying on the evaluation set |
| Gemma 4 (26B or 31B), Gemini API | Free; no paid tier | 0 | Function calling supported. Rate-limited, no tuning, and free-tier prompts may be used to improve Google's products |
| Gemma 4 26B, managed on Vertex | $0.15 in, $0.60 out | about 0.5 cents | Experimental; function calling supported |
| Our own Gemma on Cloud Run, one L4 GPU | About $0.67 an hour for the GPU, plus at least 4 vCPU and 16 GiB | A fixed cost while it's up | Scales to zero, but loading the model makes cold starts slow |

- **Hosted beats self-hosted at our volume.** Our own GPU only pays off at around 30 or more conversations an hour, around the clock. A project with demos is far below that.
- **Free Gemma suits development and demos, not customer text.** The free tier's data terms conflict with core rule 6 for anything a real customer types. Production use needs a paid tier.
- **Fine-tuning Gemma means hosting it.** The hosted Gemma doesn't take tuned weights, so a fine-tuned model brings back the fixed GPU cost.

**Recommendation:** develop on the Gemini API, keep the model name in `configs/agents.yaml`, and let the evaluation set choose among Gemini Flash, Flash-Lite and hosted Gemma 4. Pick the cheapest model that meets the targets.

## Part 1: infrastructure

Build in this order.

1. **Freeze the schemas.** Pydantic models in `src/agents/event_spec.py` and `src/agents/adjustment_spec.py`, with the fields in llm_agents.md. Every field is `{value, source, evidence}`, and `evidence` must be an exact span of the user's text. A validator rejects any `user_text` value without a matching span, which catches invented values.
2. **The site gazetteer.** A precomputed table per site of every place a user might name: gates and entrances, venues, streets, stations, plazas, bridges and car parks. It's built from OSM, the venues in `configs/events.yaml`, and hand-added entries. For each place, code stores:
   - its bearing and distance from the venue centre;
   - its side of the venue, by compass and by the street grid. Melbourne's central grid is turned about 20° from true north, so the two can differ;
   - the streets it's next to;
   - aliases ("Gate 3", "the Swan St gate", "the Kenmore side").

   It's versioned with the graph, like the OSM features.
3. **`resolve_place`.** The tool takes a structured query, never coordinates: the user's words (`mention`), a `kind`, and optionally a `relation` (`north_of`, `near`, `between`) with an `anchor`. Code scores the gazetteer and returns ranked candidates, each with a reason ("bearing 12° from the venue centre, on Jersey St"). Directions use tolerance windows. When one candidate clearly wins, the UI pre-selects it for one-click confirmation. When several are close, the agent asks. Nothing is applied until the user confirms it on the map (rule 4).
4. **The action tools.** For Tier 1:
   - `get_scenario`;
   - `resolve_place`;
   - `set_event_fields`;
   - `edit_graph` (close or reopen a segment, add an entrance);
   - `propose_adjustment`;
   - `run_prediction`;
   - `compare`.

   Arguments can only be IDs that an earlier tool call in the same session returned. The orchestrator rejects any other ID, so the model can't invent a segment. Tools that change the scenario return a pending edit, which the UI shows for confirmation before anything runs.
5. **The orchestrator.** A function-calling loop on the `google-genai` SDK, with:
   - a cap on steps and on model runs per turn;
   - scenario state kept on the server, not in the prompt;
   - a thin adapter so Flash, Flash-Lite and Gemma swap through the config. Hosted Gemma and Gemini use the same SDK and function-call format.
6. **The grounding checker and template narration** (`src/agents/grounding.py`, `src/agents/narrate.py`), as in llm_agents.md. Every number and place in a narration must match a tool result, required caveats must be present and banned wording absent. A failed check falls back to the template narration.
7. **Operations.**
   - The API key goes in Secret Manager, with a budget alert on the Gemini project.
   - Each turn is traced: prompt, tool calls, tokens, cost, the grounding result and any user corrections. MLflow tracing works, since MLflow already runs on the VM.
   - The demo uses cached replay, so it never depends on a live call.

## Part 2: model behaviour

1. **The evaluation set comes first.** How it's built is in [eval_set_plan.md](eval_set_plan.md). The cases cover:
   - field extraction;
   - place resolution: directions, relations, aliases, landmarks;
   - ambiguity, where the right behaviour is to ask;
   - tool choice and order;
   - traps: counts asked for before any run, safety guarantees, setting an adjustment's strength;
   - corrections over several turns.
2. **Metrics**, with the team's targets:
   - field-level accuracy of at least 90%;
   - place accuracy;
   - asking when it should, and not asking when it shouldn't;
   - correct tool sequences;
   - zero grounding violations reaching the user;
   - cost and latency per conversation.

   Each evaluation run is logged to MLflow, like the flow model, and a change to the prompt or model must not make it worse.
3. **Context and prompts.** This is the main lever:
   - a system prompt with the rules;
   - a short site summary added to each session (the venue, its gates and their sides, the main streets), so the model knows what can be named;
   - tool descriptions that spell out the relation vocabulary;
   - 5 to 10 worked examples.

   Most place errors are fixed here, in the gazetteer, or in how `resolve_place` scores candidates, not in the model.
4. **Model comparison.** Run the same evaluation on Flash, Flash-Lite and hosted Gemma 4, and record the result as the model choice, with its costs.
5. **Fine-tuning, only if a gap remains** that prompts and tools can't close. Generate dialogues from the gazetteer, the same way as the generated evaluation cases but on other sites or places, so training and evaluation don't overlap. Fit a LoRA on a small Gemma 4 (E4B or 12B) and host it on Cloud Run. Before starting:
   - check the Gemini API terms on using its outputs as training data;
   - check Gemma's licence for distributing tuned weights (core rule 5);
   - budget for the GPU.

## Scope

The intake agent, with closures in the event spec, is Tier 1. A free-form "close this, then compare" loop is the planning assistant, Tier 2. The tools and orchestrator above serve both, so they're built once, and the Tier 1 path (describe, confirm on the map, run, narrate) works first.

## Sources

- [Gemini Developer API pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [What's new in Gemini 3.8 Flash](https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.8)
- [Run Gemma with the Gemini API](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api)
- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Gemma 4 26B A4B on Google Cloud](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/maas/google/gemma-4-26b-a4b-it)
- [Cloud Run pricing](https://cloud.google.com/run/pricing)
- [Cloud Run GPU support](https://docs.cloud.google.com/run/docs/configuring/services/gpu)
- [Gemini API rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)
