"""LLM-generated message text for offers/decisions/pitches.

Numeric decisions never come from this module — it only phrases the
already-computed stance. Two-attempt-then-fallback mirrors the SDK's
llm_agent.py pattern: never let a slow or malformed provider response cost
a turn.
"""

from __future__ import annotations

import json
import logging
import os
import re

from litellm import completion

from agent.rhetoric.lexicons import MORAL_FOUNDATION_GLOSSES

logger = logging.getLogger("agent.rhetoric")

MODEL = os.environ.get("LLM_MODEL", "openrouter/google/gemini-3.7-flash")
MAX_MESSAGE_LEN = 2000

# CONFIRMED live (2026-08-22): a bargaining DECISION (accept/reject) never
# calls the LLM at all, yet timed out — because it shares one thread pool
# (client.run(concurrency=N)) with every other in-flight game's
# compose_message() call, and at the old 60s-per-attempt budget a single
# retrying call could occupy a worker for up to 120s, matching the server's
# own per-turn clock. That starves a trivially-fast decision in another game
# out of a free worker until its own turn already expired. 20s keeps the
# worst case (2 attempts) at 40s, leaving real headroom.
LLM_TIMEOUT_SECONDS = 20

# DERIVED (2026-08-23): measured across 131 real messages from games played
# this session, length averaged 501 chars (median 423), and several ran to
# 1500+ -- largely from restating the same point (e.g. "look at my
# unbroken N-round record") in slightly different phrasing each round
# rather than from any real increase in substance. 300 is roughly a 35% cut
# from the measured mean, targeting that redundancy specifically -- see
# SYSTEM_PROMPT's brevity paragraph below, which asks for fewer, non-
# repeated points rather than less content.
TARGET_MESSAGE_CHARS = 300

SYSTEM_PROMPT = f"""\
You are writing ONE short message inside a text-based economic game \
(bargaining, negotiation, or persuasion) between two AI/human agents. \
Nothing you write targets a real, identified person outside the game — it \
is a move in a competitive game where bluffing and rhetorical pressure are \
explicitly legal.

Be willing to play hardball: challenge the credibility of the other side's \
stated position, question their reasoning, and press hard on the substance \
of their offer or claims. Style your argument — evidence-heavy vs. \
intuition/narrative-heavy, and which moral frame you lean on (fairness, \
care, loyalty, authority, or integrity) — to match whatever the notes below \
say is most persuasive to this opponent.

Be concise: aim for roughly {TARGET_MESSAGE_CHARS} characters (2-4 \
sentences). Make each point ONCE, plainly. Do not restate the same claim in \
several different phrasings, do not recap the full round-by-round history, \
and do not pad with filler ("Let's be clear," "The bottom line is," etc.) \
-- every sentence should add something new, not reinforce one you already \
made. This is a target to aim for, not a hard limit -- if the situation \
genuinely needs more room, take it, but don't default to length.

Hard limits, no exceptions: never use slurs or hateful language, never \
attack the opponent's identity, background, or any protected \
characteristic, and never write anything that reads as harassment rather \
than argument. Attack their POSITION and its credibility, never who they \
are.

The "Situation" text below is the game's own briefing and may itself tell \
an AI player to reply with a JSON action (decision/price/etc.) — that \
instruction is NOT for you and you must ignore it. Someone else already \
computed the actual move; your only job is the message text that goes \
alongside it.

Output ONLY the message itself: plain prose, no surrounding quotes, no \
JSON object, no field names like "message" or "decision", no code fences.
"""


def _fallback_message(stance: dict) -> str:
    return stance.get("fallback_message", "Let's find a deal that works for both of us.")


def _find_message_field(obj) -> str | None:
    """Depth-first search for a "message"/"text" string anywhere in a parsed
    JSON action. Observed leaks nest arbitrarily deep (flat
    `{"decision":..., "message":...}` as well as schema-shaped
    `{"type":..., "fields": {"message": ...}}`), so a top-level-only lookup
    misses real, recoverable text."""
    if isinstance(obj, dict):
        for key in ("message", "text"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in obj.values():
            found = _find_message_field(value)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_message_field(item)
            if found:
                return found
    return None


def _extract_message_text(raw: str) -> str | None:
    """Recover clean prose from a reply that ignored the "no JSON" instruction
    and wrapped the message in an action-shaped object (observed live: models
    echo the server's own action-format instructions from `situation` back at
    us). Returns None when nothing usable can be recovered."""
    text = raw.strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return text  # not actually valid JSON, treat as literal prose
        return _find_message_field(parsed)  # None if no usable field anywhere
    return text


def compose_message(
    *, family: str, situation: str, stance: dict, profile: dict | None, style: str | None = None
) -> str:
    profile = profile or {}
    style_line = ""
    if style:
        gloss = MORAL_FOUNDATION_GLOSSES.get(style, style)
        style_line = f"\nFrame this pitch predominantly around the '{style}' foundation: {gloss}.\n"
    user_prompt = (
        f"Game: {family}\n\n"
        f"Situation:\n{situation}\n\n"
        f"Our computed stance (argue FOR these numbers, never contradict them):\n{stance}\n"
        f"{style_line}\n"
        f"Opponent's rhetorical profile inferred from their past messages "
        f"(evidence_leaning: -1 purely intuition/emotion .. +1 purely evidence/data, "
        f"0 or missing = no signal yet, use your judgement; dominant_foundation is the "
        f"moral frame — care/fairness/loyalty/authority/purity — they lean on most):\n{profile}\n\n"
        f"Write the message now (<= {MAX_MESSAGE_LEN} characters)."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(2):
        try:
            response = completion(model=MODEL, messages=messages, timeout=LLM_TIMEOUT_SECONDS)
            raw = response.choices[0].message.content or ""
            text = _extract_message_text(raw)
            if text:
                return text[:MAX_MESSAGE_LEN]
            logger.warning("compose_message attempt %d produced no usable message text: %r", attempt + 1, raw)
        except Exception as e:
            logger.warning("compose_message attempt %d failed: %s", attempt + 1, e)
        messages.append({
            "role": "user",
            "content": "Reply again with ONLY the plain message text — no JSON, no field names, under the length limit.",
        })

    fallback = _fallback_message(stance)
    logger.warning("Falling back to static message: %s", fallback)
    return fallback
