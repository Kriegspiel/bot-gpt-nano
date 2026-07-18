# Release Notes

These notes summarize the bot runtime release history reconstructed from the
current repository state. Add a new section at the top for runtime,
deployment-facing, or user-visible bot behavior changes. Test-only and
docs-only changes do not need entries unless they affect operator workflow.

## Shared OpenAI Monthly Cap

- **Provider Cap**: enforce a shared `$18` cap per UTC calendar month across
  GPT Nano and every other direct OpenAI bot process on the host.
- **Strict Accounting**: reserve a conservative upper-bound cost before each
  request, settle from returned token usage, and consume the full reservation
  when the provider does not return trustworthy usage.
- **Fallback**: stop paid OpenAI calls, report the provider unavailable, and
  continue active games with legal local fallback moves when the cap is spent.

## Non-Billable OpenAI Availability

- **Idle Cost Fix**: replace the periodic Responses API `Ping` generation with
  authenticated `GET /v1/models/{model}` metadata checks.
- **Availability**: keep the existing readiness cache and backend reporting
  behavior without consuming input, output, or reasoning tokens while idle.

## Tiered Bot Join Budgets

- **Lobby Policy**: derive bot-vs-bot join probability from
  `KRIEGSPIEL_LLM_BOT_TIER` by default: T2 `0.0010`, T3 `0.0005`, T4
  `0.0002`, and T5 `0.0001`.
- **Operator Override**: allow `KRIEGSPIEL_BOT_GAME_PICK_PROBABILITY` or
  `BOT_GAME_PICK_PROBABILITY` to override the tier default per instance.

## Current Runtime Baseline

- **Bot Identity**: `llm_gptnano`, the OpenAI GPT-Nano model bot.
- **Rulesets**: supports `berkeley`, `berkeley_any`, `cincinnati`, `wild16`,
  `rand`, `english`, and `crazykrieg`, with legacy two-ruleset configs expanded
  to the full supported set.
- **Runtime Shape**: runs one process per bot identity or model instance, with
  one lightweight runner thread per active game and a configurable shared model
  call cap that defaults to 5 concurrent calls.
- **Lobby Policy**: does not create human lobby games by default, can join a
  compatible bot-created waiting game using its tier probability on a
  ten-minute scan, and checks OpenAI availability before joining new
  bot-vs-bot games.
- **Move Policy**: builds compact stateless prompts from private game state,
  ruleset summaries, public state, recent scorecard turns, legal actions, and
  retry feedback, then validates ranked JSON model output before playing.
