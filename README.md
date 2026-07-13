# bot-gpt-nano

Kriegspiel bot that asks an OpenAI model to choose the next action from the bot's private game state.

## What it does

- registers as a listed Kriegspiel bot
- syncs its supported rulesets with the API on startup
- runs one bot process per bot identity/model instance
- polls assigned games from the main process and runs one lightweight runner thread per active game
- does not create waiting lobby games by default
- can join another bot's waiting lobby game with 1% probability while still under its active-game cap
- builds a stateless prompt from a file-backed ruleset summary, private FEN, ruleset-specific public state, recent scorecard turns, legal actions, and retry feedback
- asks an OpenAI model for the top ranked next actions in compact strict JSON
- validates the model output against the server-provided legal actions
- honors explicit server-reported ply caps before asking the model; current
  bot-vs-bot LLM game caps are backend-enforced completed-turn limits
- checks OpenAI availability with a tiny cached preflight call before joining a new bot-vs-bot game
- skips the join if OpenAI is unavailable or out of quota
- still falls back safely if the model response itself is malformed

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python bot.py --register
python bot.py
```

The bot uses dedicated prompt summaries in `ruleset_summaries/*.md`, derived from the canonical `ks-content/rules` docs.

By default the registration email is `bot-gpt-nano@kriegspiel.org`.

## Multiple Model Instances

Use separate env and state files when running one independent OpenAI bot per
model:

```bash
python bot.py \
  --env-file instances/gpt55.env \
  --state-file instances/gpt55-state.json \
  --register

python bot.py \
  --env-file instances/gpt55.env \
  --state-file instances/gpt55-state.json
```

Each instance env must have its own Kriegspiel bot identity and `OPENAI_MODEL`.
`ks-deploy bot-instance-bootstrap bot-gpt-nano ...` renders this shape for
production instances.

By default the bot does not create open lobby games on its own. That behavior is controlled with:

- `KRIEGSPIEL_AUTO_CREATE_LOBBY_GAME=true|false`
- `KRIEGSPIEL_AUTO_CREATE_RULE_VARIANT=berkeley|berkeley_any|cincinnati|wild16|rand|english|crazykrieg`
- `KRIEGSPIEL_AUTO_CREATE_PLAY_AS=white|black|random`
- `KRIEGSPIEL_SUPPORTED_RULE_VARIANTS=berkeley,berkeley_any,cincinnati,wild16,rand,english,crazykrieg`
- `KRIEGSPIEL_MAX_ACTIVE_GAMES_BEFORE_CREATE=1`
- `KRIEGSPIEL_ACTIVE_GAME_DISCOVERY_LIMIT=100`
- `LLM_BOT_MAX_CONCURRENT_MODEL_CALLS=5`
- `KRIEGSPIEL_LLM_BOT_TIER=T2|T3|T4|T5`
- `KRIEGSPIEL_BOT_GAME_PICK_PROBABILITY=0.001` optional explicit join
  probability override; if unset, tier defaults are T2 `0.0010`, T3 `0.0005`,
  T4 `0.0002`, and T5 `0.0001`
- `KRIEGSPIEL_AUTO_CREATE_COOLDOWN_SECONDS=3600|10800|21600`
- `KRIEGSPIEL_RESIGN_AFTER_MOVE_NUMBER=256` fallback used only when the server
  omits an LLM bot limit field

Existing production env files with the old default `KRIEGSPIEL_SUPPORTED_RULE_VARIANTS=berkeley,berkeley_any` are treated as stale defaults and expanded to all supported rulesets.

Bot-vs-bot play is also enabled by default:

- the bot samples open waiting games at most once every 10 minutes
- it will only consider games created by another bot
- it samples that decision at most once every 10 minutes
- it will try to join one using the configured/tiered budget probability on
  that scan
- it uses the same 1-active-game cap for intentional bot-vs-bot joins
- it keeps the local cooldown even when no join candidate is found, matching backend bot-join limits and avoiding tight lobby scans

Assigned active games are handled by per-game runner threads inside the same
process. The main loop discovers active games with
`KRIEGSPIEL_ACTIVE_GAME_DISCOVERY_LIMIT`, handles lobby create/join policy, and
starts missing runners. Existing runners are not stopped only because a later
capped discovery response omits them; each runner exits when its own game-state
poll reports completion or unavailability. OpenAI calls across all game runners
are bounded by `LLM_BOT_MAX_CONCURRENT_MODEL_CALLS`, which defaults to `5`.
Backend polling, lobby scans, sleeps, and fallback move selection do not hold
that provider-call gate.

Optional human-lobby creation is still disabled by default for individual model
instances. If an operator enables one selected model instance as the random
tier representative, the built-in create cooldown defaults to T2 hourly, T3
every 3 hours, and T4 every 6 hours; `KRIEGSPIEL_AUTO_CREATE_COOLDOWN_SECONDS`
overrides that cadence.

OpenAI prompting defaults:

- system prompt carries a ruleset-specific summary from `ruleset_summaries/*.md` and the overall Kriegspiel scene
- user prompt is stateless and uses the same compact turn keys as the Haiku bot for private FEN, public material/reserves, at least the last 10 scorecard turns when available, legal actions, and retry feedback
- the bot asks for exactly the top 10 ranked candidate actions by default when 10 legal actions exist
- if a batch fails, it asks the model for the next batch of compact move candidates
- defaults can be tuned with:
  - `OPENAI_MODEL=gpt-5.4-nano`
  - `OPENAI_MAX_PROMPT_TURNS=10` (values below 10 are clamped to 10)
  - `OPENAI_MODEL_BATCH_SIZE=10`
  - `OPENAI_MAX_BATCHES_PER_TURN=5`
  - `OPENAI_PREFLIGHT_SUCCESS_TTL_SECONDS=60`
  - `OPENAI_PREFLIGHT_FAILURE_TTL_SECONDS=15`
  - `OPENAI_INPUT_USD_PER_MILLION_TOKENS=0.20`
  - `OPENAI_CACHED_INPUT_USD_PER_MILLION_TOKENS=0.02`
  - `OPENAI_OUTPUT_USD_PER_MILLION_TOKENS=1.25`

## Test

```bash
python -m unittest discover -s tests
```

## systemd

A production host can run the bot as a service with `deploy/kriegspiel-gpt-nano-bot.service`.
