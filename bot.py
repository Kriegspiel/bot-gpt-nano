"""Kriegspiel bot driven by an OpenAI model.

The bot keeps the game API as the source of truth:
- the server provides the bot's private board state and legal actions
- the model recommends one action in strict JSON
- the bot validates that action locally before sending it back to the API

If the model output is malformed, stale, or unavailable, the bot falls back to a
deterministic legal action so the service remains playable.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

BASE_DIR = Path(__file__).resolve().parent
STATE_PATH = BASE_DIR / ".bot-state.json"
ENV_PATH = BASE_DIR / ".env"
CONTENT_DIR = BASE_DIR.parent / "content" / "rules"
DEFAULT_TIMEOUT_SECONDS = 20
ACTION_SCHEMA_NAME = "kriegspiel_next_action"
DEFAULT_MAX_ACTIVE_GAMES_BEFORE_CREATE = 5


def load_env_file(path: str | Path = ENV_PATH) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def base_url() -> str:
    return os.environ.get("KRIEGSPIEL_API_BASE", "http://localhost:8000").rstrip("/")


def openai_base_url() -> str:
    return os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")


def openai_timeout_seconds() -> float:
    raw = os.environ.get("OPENAI_TIMEOUT_SECONDS", "45").strip()
    try:
        return max(5.0, float(raw))
    except ValueError:
        return 45.0


def auth_headers() -> dict[str, str]:
    token = os.environ.get("KRIEGSPIEL_BOT_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def save_token(token: str) -> None:
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    state["token"] = token
    STATE_PATH.write_text(json.dumps(state, indent=2))


def maybe_restore_token() -> None:
    if os.environ.get("KRIEGSPIEL_BOT_TOKEN"):
        return
    if STATE_PATH.exists():
        token = json.loads(STATE_PATH.read_text()).get("token")
        if token:
            os.environ["KRIEGSPIEL_BOT_TOKEN"] = token


def register_bot() -> None:
    response = requests.post(
        f"{base_url()}/api/auth/bots/register",
        headers={"X-Bot-Registration-Key": os.environ["KRIEGSPIEL_BOT_REGISTRATION_KEY"]},
        json={
            "username": os.environ.get("KRIEGSPIEL_BOT_USERNAME", "gptnano"),
            "display_name": os.environ.get("KRIEGSPIEL_BOT_DISPLAY_NAME", "GPT Nano"),
            "owner_email": os.environ.get("KRIEGSPIEL_BOT_OWNER_EMAIL", "bot-gpt-nano@kriegspiel.org"),
            "description": os.environ.get(
                "KRIEGSPIEL_BOT_DESCRIPTION",
                "Model-driven Kriegspiel bot that chooses moves using GPT nano model.",
            ),
            "listed": True,
        },
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    save_token(payload["api_token"])
    print(json.dumps(payload, indent=2))


def get_json(path: str) -> dict[str, Any]:
    response = requests.get(f"{base_url()}{path}", headers=auth_headers(), timeout=DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def post_json(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.post(
        f"{base_url()}{path}",
        headers=auth_headers(),
        json=payload or {},
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def auto_create_enabled() -> bool:
    raw = os.environ.get("KRIEGSPIEL_AUTO_CREATE_LOBBY_GAME", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def max_active_games_before_create() -> int:
    raw = os.environ.get("KRIEGSPIEL_MAX_ACTIVE_GAMES_BEFORE_CREATE", str(DEFAULT_MAX_ACTIVE_GAMES_BEFORE_CREATE)).strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_MAX_ACTIVE_GAMES_BEFORE_CREATE


def create_payload() -> dict[str, str]:
    return {
        "rule_variant": os.environ.get("KRIEGSPIEL_AUTO_CREATE_RULE_VARIANT", "berkeley_any").strip() or "berkeley_any",
        "play_as": os.environ.get("KRIEGSPIEL_AUTO_CREATE_PLAY_AS", "random").strip() or "random",
        "time_control": "rapid",
        "opponent_type": "human",
    }


def active_games(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [game for game in games if game.get("state") == "active"]


def waiting_games(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [game for game in games if game.get("state") == "waiting"]


def should_create_lobby_game(games: list[dict[str, Any]]) -> bool:
    if not auto_create_enabled():
        return False
    if waiting_games(games):
        return False
    return len(active_games(games)) < max_active_games_before_create()


def maybe_create_lobby_game(games: list[dict[str, Any]]) -> bool:
    if not should_create_lobby_game(games):
        return False

    created = post_json("/api/game/create", create_payload())
    print(f"created lobby game {created['game_id']} ({created['game_code']})")
    return True


def strip_frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return text.strip()
    parts = text.split("\n---\n", 1)
    return parts[1].strip() if len(parts) == 2 else text.strip()


def _variant_any_rule_excerpt() -> str:
    readme = (CONTENT_DIR / "README.md").read_text()
    lines = []
    for line in readme.splitlines():
        if line.startswith("| Berkeley + Any"):
            lines.append(line.strip())
    return "\n".join(lines)


def load_rules_text(rule_variant: str) -> str:
    berkeley = strip_frontmatter((CONTENT_DIR / "berkeley.md").read_text())
    if rule_variant == "berkeley_any":
        any_excerpt = _variant_any_rule_excerpt()
        appendix = (
            "\n\nAdditional variant note for Berkeley + Any:\n"
            "The current game allows the special 'Any?' action. "
            "If the referee says there are pawn captures, the player must make one.\n"
        )
        if any_excerpt:
            appendix += f"\nRules comparison excerpt:\n{any_excerpt}\n"
        return berkeley + appendix
    return berkeley


def summarize_scoresheet_turns(scoresheet: dict[str, Any], *, max_turns: int) -> str:
    viewer_color = scoresheet.get("viewer_color", "unknown")
    turns = scoresheet.get("turns") if isinstance(scoresheet.get("turns"), list) else []
    recent_turns = turns[-max_turns:]
    lines = [f"Viewer color: {viewer_color}", f"Recent turns kept: {len(recent_turns)}"]

    for turn in recent_turns:
        turn_number = turn.get("turn")
        lines.append(f"Turn {turn_number}:")
        for color in ("white", "black"):
            entries = turn.get(color) if isinstance(turn.get(color), list) else []
            if not entries:
                continue
            for entry in entries:
                normalized = normalize_scoresheet_entry(entry)
                if normalized:
                    lines.append(f"- {color}: {normalized}")
    return "\n".join(lines)


def normalize_scoresheet_entry(entry: Any) -> str:
    if isinstance(entry, str):
        return entry.strip()
    if not isinstance(entry, dict):
        return ""

    prompt = str(entry.get("prompt") or "").strip()
    messages = entry.get("messages") if isinstance(entry.get("messages"), list) else []
    cleaned_messages = [str(item).strip() for item in messages if str(item).strip()]
    if not cleaned_messages:
        message = str(entry.get("message") or "").strip()
        if message:
            cleaned_messages = [message]

    text = " | ".join(dict.fromkeys(cleaned_messages))
    move_uci = str(entry.get("move_uci") or "").strip()

    if move_uci:
        text = f"[{move_uci}] {text}" if text else f"[{move_uci}]"
    if prompt and text:
        return f"{prompt}: {text}"
    return text or prompt


def build_prompt(state: dict[str, Any]) -> str:
    rule_variant = state.get("rule_variant", "berkeley_any")
    scoresheet = state.get("scoresheet") if isinstance(state.get("scoresheet"), dict) else {}
    allowed_moves = state.get("allowed_moves") if isinstance(state.get("allowed_moves"), list) else []
    possible_actions = state.get("possible_actions") if isinstance(state.get("possible_actions"), list) else []
    max_prompt_turns = int(os.environ.get("OPENAI_MAX_PROMPT_TURNS", "24"))

    rules_text = load_rules_text(rule_variant)
    scoresheet_text = summarize_scoresheet_turns(scoresheet, max_turns=max_prompt_turns)

    return (
        "You are a strong Kriegspiel player.\n"
        "Choose the single best next action for the current player using only the private information below.\n"
        "Do not invent moves. Use only the provided legal actions.\n"
        "If you choose action=move, the uci value must be one of allowed_moves exactly.\n"
        "If you choose action=ask_any, set uci to null.\n"
        "Return valid JSON only.\n\n"
        f"Rule variant: {rule_variant}\n"
        f"Your color: {state.get('your_color')}\n"
        f"Game state: {state.get('state')}\n"
        f"Turn: {state.get('turn')}\n"
        f"Move number: {state.get('move_number')}\n"
        f"Private board FEN: {state.get('your_fen')}\n"
        f"Possible actions: {json.dumps(possible_actions)}\n"
        f"Allowed moves: {json.dumps(allowed_moves)}\n\n"
        "Private scoresheet:\n"
        f"{scoresheet_text}\n\n"
        "Rules:\n"
        f"{rules_text}\n"
    )


def action_schema() -> dict[str, Any]:
    return {
        "name": ACTION_SCHEMA_NAME,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action": {"type": "string", "enum": ["move", "ask_any"]},
                "uci": {"type": ["string", "null"]},
                "reason": {"type": "string"},
            },
            "required": ["action", "uci", "reason"],
        },
        "strict": True,
    }


def openai_enabled() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def call_openai(prompt: str) -> dict[str, Any]:
    api_key = os.environ["OPENAI_API_KEY"].strip()
    model = os.environ.get("OPENAI_MODEL", "gpt-5-nano").strip()
    response = requests.post(
        f"{openai_base_url()}/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": ACTION_SCHEMA_NAME,
                    "schema": action_schema()["schema"],
                    "strict": True,
                }
            },
        },
        timeout=openai_timeout_seconds(),
    )
    response.raise_for_status()
    return response.json()


def extract_response_text(payload: dict[str, Any]) -> str:
    output = payload.get("output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
        if chunks:
            return "\n".join(chunks)

    for key in ("output_text", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("No text found in OpenAI response payload")


def parse_model_decision(payload: dict[str, Any]) -> dict[str, Any]:
    text = extract_response_text(payload)
    try:
        decision = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        decision = json.loads(match.group(0))

    if not isinstance(decision, dict):
        raise ValueError("Model response must decode to an object")
    return decision


def normalize_decision(decision: dict[str, Any], state: dict[str, Any]) -> dict[str, Any] | None:
    possible_actions = state.get("possible_actions") if isinstance(state.get("possible_actions"), list) else []
    allowed_moves = state.get("allowed_moves") if isinstance(state.get("allowed_moves"), list) else []

    action = str(decision.get("action") or "").strip().lower()
    uci = decision.get("uci")

    if action == "move":
        if "move" not in possible_actions or not isinstance(uci, str):
            return None
        normalized_uci = uci.strip().lower()
        if normalized_uci not in allowed_moves:
            return None
        return {"action": "move", "uci": normalized_uci}

    if action == "ask_any":
        if "ask_any" not in possible_actions:
            return None
        return {"action": "ask_any", "uci": None}

    return None


def fallback_decision(state: dict[str, Any]) -> dict[str, Any] | None:
    allowed_moves = state.get("allowed_moves") if isinstance(state.get("allowed_moves"), list) else []
    possible_actions = state.get("possible_actions") if isinstance(state.get("possible_actions"), list) else []

    if allowed_moves:
        ordered = sorted(allowed_moves)
        center_bias = ["d2d4", "e2e4", "d7d5", "e7e5"]
        for preferred in center_bias:
            if preferred in ordered:
                return {"action": "move", "uci": preferred}
        return {"action": "move", "uci": ordered[0]}
    if "ask_any" in possible_actions:
        return {"action": "ask_any", "uci": None}
    return None


def choose_action(state: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    if not openai_enabled():
        return fallback_decision(state), "fallback_no_openai_key"

    try:
        prompt = build_prompt(state)
        raw_response = call_openai(prompt)
        decision = normalize_decision(parse_model_decision(raw_response), state)
        if decision is not None:
            return decision, "model"
    except (KeyError, ValueError, json.JSONDecodeError, requests.RequestException) as exc:
        print(f"model selection failed: {exc}", file=sys.stderr, flush=True)

    return fallback_decision(state), "fallback"


def maybe_play_game(game_id: str) -> bool:
    state = get_json(f"/api/game/{game_id}/state")
    if state.get("state") != "active" or state.get("turn") != state.get("your_color"):
        return False
    metadata = get_json(f"/api/game/{game_id}")
    if isinstance(metadata.get("rule_variant"), str):
        state["rule_variant"] = metadata["rule_variant"]

    decision, source = choose_action(state)
    if decision is None:
        return False

    if decision["action"] == "move":
        result = post_json(f"/api/game/{game_id}/move", {"uci": decision["uci"]})
        print(f"{game_id}: {source} move {decision['uci']} -> {result['announcement']}")
        return bool(result.get("move_done"))

    result = post_json(f"/api/game/{game_id}/ask-any")
    print(f"{game_id}: {source} ask-any -> {result['announcement']}")
    return bool(result.get("move_done"))


def run_loop(poll_seconds: float) -> None:
    while True:
        try:
            mine = get_json("/api/game/mine")
            games = mine.get("games", [])
            maybe_create_lobby_game(games)
            for game in active_games(games):
                maybe_play_game(game["game_id"])
        except requests.RequestException as exc:
            print(f"poll failed: {exc}", file=sys.stderr, flush=True)
        time.sleep(poll_seconds)


def main() -> None:
    load_env_file()
    maybe_restore_token()

    parser = argparse.ArgumentParser(description="Run the Kriegspiel GPT Nano bot.")
    parser.add_argument("--register", action="store_true", help="Register the bot and persist the returned token.")
    parser.add_argument("--poll-seconds", type=float, default=3.0, help="Seconds between /api/game/mine polls.")
    args = parser.parse_args()

    if args.register:
        register_bot()
        return

    if not os.environ.get("KRIEGSPIEL_BOT_TOKEN"):
        raise SystemExit("KRIEGSPIEL_BOT_TOKEN is missing. Run with --register first.")
    if not openai_enabled():
        print("OPENAI_API_KEY is missing; running in fallback mode.", file=sys.stderr, flush=True)

    run_loop(args.poll_seconds)


if __name__ == "__main__":
    main()
