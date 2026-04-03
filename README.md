# bot-gpt-nano

Kriegspiel bot that asks an OpenAI model to choose the next action from the bot's private game state.

## What it does

- registers as a listed Kriegspiel bot
- polls assigned games from the live API
- can keep one open human-joinable lobby game advertised when under its active-game cap
- builds a prompt from the current rule variant, private FEN, legal actions, and private scoresheet
- asks an OpenAI model for the next action in strict JSON
- validates the model output against the server-provided legal actions
- falls back safely if the model response is invalid or the API call fails

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python bot.py --register
python bot.py
```

The bot reads the live rules text from the sibling `content` repository:

- `content/rules/berkeley.md`
- `content/rules/README.md`

By default the registration email is `bot-gpt-nano@kriegspiel.org`.

By default the bot also keeps one open lobby game available for humans to join,
but only while it has fewer than 5 active games. That behavior is controlled with:

- `KRIEGSPIEL_AUTO_CREATE_LOBBY_GAME=true|false`
- `KRIEGSPIEL_AUTO_CREATE_RULE_VARIANT=berkeley|berkeley_any`
- `KRIEGSPIEL_AUTO_CREATE_PLAY_AS=white|black|random`
- `KRIEGSPIEL_MAX_ACTIVE_GAMES_BEFORE_CREATE=5`

## Test

```bash
python -m unittest discover -s tests
```

## systemd

A production host can run the bot as a service with `deploy/kriegspiel-gpt-nano-bot.service`.
