from __future__ import annotations

import unittest

import bot


class BotTests(unittest.TestCase):
    def test_normalize_decision_accepts_legal_move(self) -> None:
        state = {"possible_actions": ["move", "ask_any"], "allowed_moves": ["e2e4", "d2d4"]}
        decision = bot.normalize_decision({"action": "move", "uci": "E2E4"}, state)
        self.assertEqual(decision, {"action": "move", "uci": "e2e4"})

    def test_normalize_decision_rejects_illegal_move(self) -> None:
        state = {"possible_actions": ["move"], "allowed_moves": ["e2e4"]}
        decision = bot.normalize_decision({"action": "move", "uci": "a2a4"}, state)
        self.assertIsNone(decision)

    def test_normalize_decision_accepts_ask_any_when_available(self) -> None:
        state = {"possible_actions": ["ask_any"], "allowed_moves": []}
        decision = bot.normalize_decision({"action": "ask_any", "uci": None}, state)
        self.assertEqual(decision, {"action": "ask_any", "uci": None})

    def test_extract_response_text_reads_nested_output(self) -> None:
        payload = {"output": [{"content": [{"text": "{\"action\":\"move\",\"uci\":\"e2e4\",\"reason\":\"center\"}"}]}]}
        self.assertEqual(
            bot.extract_response_text(payload),
            "{\"action\":\"move\",\"uci\":\"e2e4\",\"reason\":\"center\"}",
        )

    def test_fallback_prefers_center_moves(self) -> None:
        state = {"possible_actions": ["move"], "allowed_moves": ["a2a3", "e2e4", "h2h3"]}
        decision = bot.fallback_decision(state)
        self.assertEqual(decision, {"action": "move", "uci": "e2e4"})


if __name__ == "__main__":
    unittest.main()
