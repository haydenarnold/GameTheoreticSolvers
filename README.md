# GLEE Competition Agent

A game-theoretically grounded agent for the [GLEE Competition](https://glee-competition.com) (NeurIPS 2026 IAB workshop track): bargaining, negotiation, and persuasion.

## Layout

- `agent/bargaining/` — Rubinstein alternating-offer backward induction (finite & infinite horizon), Bayesian belief tracking over an opponent's hidden discount factor.
- `agent/negotiation/` — Faratin-et-al time-dependent concession tactics, revealed-preference bound tightening on the opponent's hidden reservation value.
- `agent/persuasion/` — Kamenica-Gentzkow-optimal disclosure (calibrated bluffing probability), online trust adaptation for both seller and buyer.
- `agent/rhetoric/` — LLM-composed message text (via `litellm`), style-matched to the opponent's inferred evidence-vs-intuition leaning and moral-foundation emphasis. Never affects numeric decisions.
- `agent/dispatch.py` / `agent/main.py` — wiring into `glee_sdk.GleeClient.run()`.
- `agent/tests/` — offline unit tests for the three solvers' math (no network needed).
- `simple_agent.py` — the SDK's own baseline example, kept for reference.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run the offline tests

```bash
.venv/bin/python -m pytest agent/tests -v
```

## Run the agent against the live competition

Needs a GLEE API key from your dashboard at glee-competition.com, and an LLM provider key for whichever `LLM_MODEL` you pick (litellm model string).

```bash
export GLEE_API_KEY=glee_...
export LLM_MODEL=gemini/gemini-2.5-flash     # or gpt-5-mini, claude-sonnet-5, ...
export GEMINI_API_KEY=...                     # whichever key LLM_MODEL needs

.venv/bin/python -m agent.main --max-games 3 --concurrency 3
```

Start with a small `--max-games` for a smoke test — watch the logs for `valid: false` rejections (a schema mismatch) before running unbounded.
