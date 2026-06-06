# bot-gpt-nano

Kriegspiel bot that asks an OpenAI model to choose the next action from the bot's private game state.

## What it does

- registers as a listed Kriegspiel bot
- syncs its supported rulesets with the API on startup
- polls assigned games from the live API
- does not create waiting lobby games by default
- can join another bot's waiting lobby game with 0.1% probability while still under its active-game cap
- builds a stateless prompt from a file-backed ruleset summary, private FEN, ruleset-specific public state, recent scorecard turns, legal actions, and retry feedback
- asks an OpenAI model for the top ranked next actions in strict JSON
- validates the model output against the server-provided legal actions
- resigns instead of asking the model once the server-reported move number reaches 256
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

By default the bot does not create open lobby games on its own. That behavior is controlled with:

- `KRIEGSPIEL_AUTO_CREATE_LOBBY_GAME=true|false`
- `KRIEGSPIEL_AUTO_CREATE_RULE_VARIANT=berkeley|berkeley_any|cincinnati|wild16|rand|english|crazykrieg`
- `KRIEGSPIEL_AUTO_CREATE_PLAY_AS=white|black|random`
- `KRIEGSPIEL_SUPPORTED_RULE_VARIANTS=berkeley,berkeley_any,cincinnati,wild16,rand,english,crazykrieg`
- `KRIEGSPIEL_MAX_ACTIVE_GAMES_BEFORE_CREATE=1`
- `KRIEGSPIEL_RESIGN_AFTER_MOVE_NUMBER=256`

Existing production env files with the old default `KRIEGSPIEL_SUPPORTED_RULE_VARIANTS=berkeley,berkeley_any` are treated as stale defaults and expanded to all supported rulesets.

Bot-vs-bot play is also enabled by default:

- the bot samples open waiting games at most once per minute
- it will only consider games created by another bot
- it samples that decision at most once per minute
- it will try to join one with 0.1% probability on that minute check
- it uses the same 1-active-game cap for intentional bot-vs-bot joins
- it keeps the local cooldown even when no join candidate is found, matching backend bot-join limits and avoiding tight lobby scans

OpenAI prompting defaults:

- system prompt carries a ruleset-specific summary from `ruleset_summaries/*.md` and the overall Kriegspiel scene
- user prompt is stateless and carries private FEN, ruleset-specific public material/reserves, at least the last 10 scorecard turns when available, legal actions, and retry feedback
- the bot asks for exactly the top 10 ranked candidate actions by default when 10 legal actions exist
- if a batch fails, it asks the model for the next batch of candidates
- defaults can be tuned with:
  - `OPENAI_MODEL=gpt-5.4-nano`
  - `OPENAI_MAX_PROMPT_TURNS=10` (values below 10 are clamped to 10)
  - `OPENAI_MODEL_BATCH_SIZE=10`
  - `OPENAI_MAX_BATCHES_PER_TURN=5`
  - `OPENAI_PREFLIGHT_SUCCESS_TTL_SECONDS=60`
  - `OPENAI_PREFLIGHT_FAILURE_TTL_SECONDS=15`

## Test

```bash
python -m unittest discover -s tests
```

## systemd

A production host can run the bot as a service with `deploy/kriegspiel-gpt-nano-bot.service`.
