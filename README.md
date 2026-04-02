# bot-gpt-nano

Kriegspiel bot that asks an OpenAI model to choose the next action from the bot's private game state.

## What it does

- registers as a listed Kriegspiel bot
- polls assigned games from the live API
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

## Test

```bash
python -m unittest discover -s tests
```

## systemd

A production host can run the bot as a service with `deploy/kriegspiel-gpt-nano-bot.service`.
