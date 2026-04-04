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
from functools import lru_cache
import json
import logging
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
DEFAULT_MAX_ACTIVE_GAMES_BEFORE_CREATE = 1
BOT_JOIN_COOLDOWN_SECONDS = 60
BOT_GAME_PICK_PROBABILITY = 0.1
DEFAULT_MODEL_BATCH_SIZE = 10
DEFAULT_MAX_MODEL_BATCHES_PER_TURN = 5
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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


def bot_username() -> str:
    return os.environ.get("KRIEGSPIEL_BOT_USERNAME", "").strip().lower()


def load_state() -> dict[str, Any]:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def save_token(token: str) -> None:
    state = load_state()
    state["token"] = token
    save_state(state)


def maybe_restore_token() -> None:
    if os.environ.get("KRIEGSPIEL_BOT_TOKEN"):
        return
    if STATE_PATH.exists():
        token = load_state().get("token")
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
            "supported_rule_variants": supported_rule_variants(),
        },
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    save_token(payload["api_token"])
    logger.debug("%s", json.dumps(payload, indent=2))


def get_json(path: str) -> dict[str, Any]:
    response = requests.get(f"{base_url()}{path}", headers=auth_headers(), timeout=DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def get_public_user(username: str) -> dict[str, Any]:
    response = requests.get(f"{base_url()}/api/user/{username}", headers=auth_headers(), timeout=DEFAULT_TIMEOUT_SECONDS)
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


def supported_rule_variants() -> list[str]:
    raw = os.environ.get("KRIEGSPIEL_SUPPORTED_RULE_VARIANTS", "berkeley,berkeley_any")
    variants: list[str] = []
    for item in raw.split(","):
        value = item.strip()
        if value in {"berkeley", "berkeley_any"} and value not in variants:
            variants.append(value)
    return variants or ["berkeley", "berkeley_any"]


def active_games(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [game for game in games if game.get("state") == "active"]


def waiting_games(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [game for game in games if game.get("state") == "waiting"]


def open_bot_lobby_candidates(open_games: list[dict[str, Any]], *, profile_lookup=None) -> list[dict[str, Any]]:
    profile_lookup = profile_lookup or get_public_user
    own_username = bot_username()
    candidates = []
    for game in open_games:
        creator_username = str(game.get("created_by") or "").strip()
        if not creator_username:
            continue
        if str(game.get("rule_variant") or "").strip() not in supported_rule_variants():
            continue
        creator_username_lower = creator_username.lower()
        if creator_username_lower == own_username:
            continue

        try:
            profile = profile_lookup(creator_username)
        except requests.RequestException:
            continue

        is_bot = bool(profile.get("is_bot")) or str(profile.get("role") or "").strip().lower() == "bot"
        if not is_bot:
            continue
        candidates.append(game)
    return candidates


def has_own_waiting_game(open_games: list[dict[str, Any]]) -> bool:
    own_username = bot_username()
    for game in open_games:
        created_by = str(game.get("created_by") or "").strip().lower()
        if created_by and created_by == own_username:
            return True
    return False


def can_attempt_bot_join(now: float | None = None) -> bool:
    current = time.time() if now is None else now
    last_attempt = load_state().get("last_bot_game_join_attempt_at", 0)
    try:
        last_attempt = float(last_attempt)
    except (TypeError, ValueError):
        last_attempt = 0
    return current - last_attempt >= BOT_JOIN_COOLDOWN_SECONDS


def record_bot_join_attempt(now: float | None = None) -> None:
    state = load_state()
    state["last_bot_game_join_attempt_at"] = time.time() if now is None else now
    save_state(state)


def should_join_bot_lobby_game(games: list[dict[str, Any]]) -> bool:
    return len(active_games(games)) < max_active_games_before_create()


def choose_bot_game_to_join(open_games: list[dict[str, Any]], *, rng: random.Random = random) -> dict[str, Any] | None:
    candidates = open_bot_lobby_candidates(open_games)
    if not candidates:
        return None
    if rng.random() >= BOT_GAME_PICK_PROBABILITY:
        return None
    return rng.choice(candidates)


def maybe_join_bot_lobby_game(games: list[dict[str, Any]], *, rng: random.Random = random) -> bool:
    if not should_join_bot_lobby_game(games):
        return False
    if not can_attempt_bot_join():
        return False

    open_games = get_json("/api/game/open").get("games", [])
    candidate = choose_bot_game_to_join(open_games, rng=rng)
    if not candidate:
        return False

    game_code = candidate.get("game_code")
    if not isinstance(game_code, str) or not game_code.strip():
        return False

    record_bot_join_attempt()
    joined = post_json(f"/api/game/join/{game_code.strip()}")
    logger.debug("joined bot lobby game %s (%s)", joined["game_id"], joined["game_code"])
    return True


def should_create_lobby_game(games: list[dict[str, Any]]) -> bool:
    if not auto_create_enabled():
        return False
    if waiting_games(games):
        return False
    return len(active_games(games)) < max_active_games_before_create()


def maybe_create_lobby_game(games: list[dict[str, Any]]) -> bool:
    if not should_create_lobby_game(games):
        return False

    open_games = get_json("/api/game/open").get("games", [])
    if has_own_waiting_game(open_games):
        return False

    created = post_json("/api/game/create", create_payload())
    logger.debug("created lobby game %s (%s)", created["game_id"], created["game_code"])
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


@lru_cache(maxsize=4)
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


def summarize_scoresheet_turns(scoresheet: dict[str, Any], *, max_turns: int) -> list[str]:
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
    return lines


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


def model_batch_size() -> int:
    raw = os.environ.get("OPENAI_MODEL_BATCH_SIZE", str(DEFAULT_MODEL_BATCH_SIZE)).strip()
    try:
        return max(1, min(20, int(raw)))
    except ValueError:
        return DEFAULT_MODEL_BATCH_SIZE


def max_model_batches_per_turn() -> int:
    raw = os.environ.get("OPENAI_MAX_BATCHES_PER_TURN", str(DEFAULT_MAX_MODEL_BATCHES_PER_TURN)).strip()
    try:
        return max(1, min(20, int(raw)))
    except ValueError:
        return DEFAULT_MAX_MODEL_BATCHES_PER_TURN


def extract_recent_referee_items(scoresheet: dict[str, Any], *, limit: int = 8) -> list[str]:
    turns = scoresheet.get("turns") if isinstance(scoresheet.get("turns"), list) else []
    lines: list[str] = []
    for turn in turns[-limit:]:
        turn_number = turn.get("turn")
        for color in ("white", "black"):
            entries = turn.get(color) if isinstance(turn.get(color), list) else []
            for entry in entries:
                normalized = normalize_scoresheet_entry(entry)
                if normalized:
                    lines.append(f"Turn {turn_number} {color}: {normalized}")
    return lines[-limit:]


def build_system_prompt(rule_variant: str) -> str:
    rules_text = load_rules_text(rule_variant)
    return (
        "You are a strong Kriegspiel player.\n"
        "Use only the provided private information and legal actions.\n"
        "Do not invent moves. Do not suggest illegal actions.\n"
        "Return unique candidate actions ordered strictly from best to worse priority.\n"
        "Candidate 1 must be your best choice, candidate 2 your next-best choice, and so on.\n"
        "Prioritize strategically strong, tactically sound moves that are robust under uncertainty.\n"
        "Do not explain the rules. Do not include prose outside the JSON schema.\n\n"
        "Rules and setting:\n"
        f"Rule variant: {rule_variant}\n"
        f"{rules_text}\n"
    )


def build_user_prompt(
    state: dict[str, Any],
    *,
    feedback: list[str] | None = None,
    exclude_actions: list[dict[str, Any]] | None = None,
) -> str:
    scoresheet = state.get("scoresheet") if isinstance(state.get("scoresheet"), dict) else {}
    allowed_moves = state.get("allowed_moves") if isinstance(state.get("allowed_moves"), list) else []
    possible_actions = state.get("possible_actions") if isinstance(state.get("possible_actions"), list) else []
    max_prompt_turns = int(os.environ.get("OPENAI_MAX_PROMPT_TURNS", "12"))
    scoresheet_text = summarize_scoresheet_turns(scoresheet, max_turns=max_prompt_turns)
    recent_referee = extract_recent_referee_items(scoresheet, limit=6)
    feedback = (feedback or [])[-4:]
    exclude_actions = exclude_actions or []
    exclusion_lines = [format_action(item) for item in exclude_actions[-12:]]
    target_count = min(model_batch_size(), max(len(allowed_moves), 1 if "ask_any" in possible_actions else 0))
    payload = {
        "your_color": state.get("your_color"),
        "game_state": state.get("state"),
        "turn": state.get("turn"),
        "move_number": state.get("move_number"),
        "private_board_fen": state.get("your_fen"),
        "possible_actions": possible_actions,
        "allowed_moves": allowed_moves,
        "recent_scoresheet_lines": scoresheet_text[-12:],
        "recent_referee_items": recent_referee,
        "feedback_this_turn": feedback,
        "already_tried_this_turn": exclusion_lines,
        "target_count": target_count,
    }
    return (
        "Current private state JSON follows.\n"
        "Return exactly target_count unique candidates when possible, ordered from best to worse priority.\n"
        "If action=move, uci must be one of allowed_moves exactly.\n"
        "If action=ask_any, uci must be null.\n\n"
        f"{json.dumps(payload, separators=(',', ':'), ensure_ascii=True, sort_keys=True)}"
    )


def action_schema() -> dict[str, Any]:
    return {
        "name": ACTION_SCHEMA_NAME,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "candidates": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "action": {"type": "string", "enum": ["move", "ask_any"]},
                            "uci": {"type": ["string", "null"]},
                            "reason": {"type": "string"},
                        },
                        "required": ["action", "uci", "reason"],
                    },
                }
            },
            "required": ["candidates"],
        },
        "strict": True,
    }


def openai_enabled() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def call_openai(prompt: Any) -> dict[str, Any]:
    api_key = os.environ["OPENAI_API_KEY"].strip()
    model = os.environ.get("OPENAI_MODEL", "gpt-5.4-nano").strip()
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


def normalize_ranked_decisions(payload: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None]] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        decision = normalize_decision(candidate, state)
        if decision is None:
            continue
        key = (decision["action"], decision["uci"])
        if key in seen:
            continue
        seen.add(key)
        normalized.append(decision)
    return normalized


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


def fallback_ranked_actions(state: dict[str, Any]) -> list[dict[str, Any]]:
    decision = fallback_decision(state)
    return [decision] if decision is not None else []


def choose_ranked_actions(
    state: dict[str, Any],
    *,
    feedback: list[str] | None = None,
    exclude_actions: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    if not openai_enabled():
        return fallback_ranked_actions(state), "fallback_no_openai_key"

    try:
        prompt = [
            {"role": "system", "content": build_system_prompt(state.get("rule_variant", "berkeley_any"))},
            {"role": "user", "content": build_user_prompt(state, feedback=feedback, exclude_actions=exclude_actions)},
        ]
        raw_response = call_openai(prompt)
        decisions = normalize_ranked_decisions(parse_model_decision(raw_response), state)
        if decisions:
            return decisions, "model"
    except (KeyError, ValueError, json.JSONDecodeError, requests.RequestException) as exc:
        logger.warning("model selection failed: %s", exc)

    return fallback_ranked_actions(state), "fallback"


def format_action(decision: dict[str, Any]) -> str:
    if decision["action"] == "ask_any":
        return "ask_any"
    return str(decision["uci"])


def maybe_play_game(game_id: str) -> bool:
    metadata = get_json(f"/api/game/{game_id}")
    feedback: list[str] = []
    tried_actions: list[dict[str, Any]] = []
    acted = False

    for _batch in range(max_model_batches_per_turn()):
        state = get_json(f"/api/game/{game_id}/state")
        if state.get("state") != "active" or state.get("turn") != state.get("your_color"):
            return acted
        if isinstance(metadata.get("rule_variant"), str):
            state["rule_variant"] = metadata["rule_variant"]

        decisions, source = choose_ranked_actions(state, feedback=feedback, exclude_actions=tried_actions)
        if not decisions:
            return acted

        batch_success = False
        for decision in decisions:
            tried_actions.append(decision)
            if decision["action"] == "move":
                result = post_json(f"/api/game/{game_id}/move", {"uci": decision["uci"]})
                acted = acted or bool(result.get("move_done"))
                logger.debug("%s: %s move %s -> %s", game_id, source, decision["uci"], result["announcement"])
                if result.get("move_done"):
                    feedback = [
                        f"Move complete: {result.get('announcement')}",
                        f"Opponent/new turn: {result.get('turn')}",
                    ]
                    tried_actions = []
                    batch_success = True
                    break
                feedback.append(f"Rejected move {decision['uci']}: {result.get('announcement')}")
                continue

            result = post_json(f"/api/game/{game_id}/ask-any")
            acted = acted or bool(result.get("move_done"))
            logger.debug("%s: %s ask-any -> %s", game_id, source, result["announcement"])
            feedback = [f"Ask-any result: {result.get('announcement')}"]
            tried_actions = []
            batch_success = True
            break

        if not batch_success:
            feedback.append("Top-ranked batch failed; provide the next best legal candidates.")
            continue

    return acted


def run_loop(poll_seconds: float) -> None:
    while True:
        try:
            mine = get_json("/api/game/mine")
            games = mine.get("games", [])
            maybe_create_lobby_game(games)
            maybe_join_bot_lobby_game(games)
            for game in active_games(games):
                maybe_play_game(game["game_id"])
        except requests.RequestException as exc:
            logger.warning("poll failed: %s", exc)
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
        logger.warning("OPENAI_API_KEY is missing; running in fallback mode.")

    run_loop(args.poll_seconds)


if __name__ == "__main__":
    main()
