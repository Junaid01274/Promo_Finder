"""Optional second pass: an LLM re-picks promos from the rule shortlist.

The model never gets to make up timestamps. It sees numbered beats and answers
with beat numbers; the code turns those back into times, snaps them so no line
is cut mid-sentence, and throws the pick out if it breaks the 20-60s rule.
Every rejection is logged, because that log is the honest answer to "where
did the model get it wrong".
"""

from __future__ import annotations

import json

from .candidates import (MAX_LEN, MIN_LEN, Promo, Scorer, Segment, beats_between,
                         valid_ends, valid_starts)
from .describe import describe, quote, segment_text
from .llm import LLM, LLMError, parse_json
from .parser import Episode, fmt

SYSTEM = """You choose promo clips for vertical microdrama ads on TikTok, Reels and Shorts.
A promo is a 20-60 second clip cut from the show. A good one:
1. hooks a scrolling stranger in the first 3 seconds (a slap, a reveal, a shocking line),
2. cuts at an unresolved moment so the viewer needs to know what happens next,
3. makes sense to someone who has never seen the show.
You refer to moments only by the [index] numbers you are given. Never invent dialogue,
names or timestamps. Reply with a single JSON object and nothing else."""

TASK = """Return up to {k} promos for this episode, best first, in this exact shape:
{{"promos": [{{
  "candidate": "C2" or null,
  "segments": [[first_index, last_index], ...],
  "hook": "what happens in the first 3 seconds",
  "cliffhanger": "the unresolved moment it cuts on",
  "why": "one line on why it works",
  "hook_score": 1-10, "cliff_score": 1-10, "standalone_score": 1-10,
  "spoiler_risk": "low" | "medium" | "high"
}}]}}
Rules:
- To use a shortlist candidate unchanged, give its id and leave "segments" empty.
- Otherwise give segments as beat index ranges from THIS episode. Total length must be 20-60s.
- One segment is normal. Use two only for a deliberate cold open (peak moment first).
- Don't start on a reply to a question the viewer never heard.
- If nothing here works as a promo, return {{"promos": []}}."""


def _beat_table(ep: Episode) -> str:
    rows = []
    for b in ep.beats:
        if b.is_line:
            tone = f" ({b.tone})" if b.tone else ""
            body = f"{b.speaker}{tone}: \"{b.text}\""
        else:
            text = b.text if len(b.text) <= 120 else b.text[:117] + "..."
            body = f"{b.kind.upper()}: {text}"
        rows.append(f"[{b.index}] {fmt(b.start)}-{fmt(b.end)} {body}")
    return "\n".join(rows)


def _candidate_table(cands: list[Promo]) -> str:
    rows = []
    for i, p in enumerate(cands, 1):
        segs = " + ".join(f"ep{s.episode} {fmt(s.start)}-{fmt(s.end)}" for s in p.segments)
        rows.append(f"C{i} ({p.kind}, {p.length:.1f}s, rule score {p.total:.2f}): {segs}")
        if any(s.episode != cands[0].segments[0].episode for s in p.segments):
            rows.append(segment_text(p))   # show text from the other episode
    return "\n".join(rows)


def _snap(ep: Episode, i: int, j: int) -> Segment | None:
    if not (0 <= i <= j < len(ep.beats)):
        return None
    starts, ends = valid_starts(ep), valid_ends(ep)
    s_raw = ep.beats[i].start
    e_raw = max(b.end for b in ep.beats[i: j + 1])
    s = max((x for x in starts if x <= s_raw + 0.01), default=None)
    e = min((x for x in ends if x >= e_raw - 0.01), default=None)
    if s is None or e is None or e <= s:
        return None
    return Segment(ep.number, s, e, beats_between(ep, s, e))


def judge_episode(llm: LLM, ep: Episode, cands: list[Promo], scorer: Scorer,
                  show_title: str, k: int = 3, rejections: list | None = None) -> list[Promo]:
    user = (f"Show: {show_title}\nEpisode {ep.number}. Logline: {ep.logline}\n\n"
            f"Beats:\n{_beat_table(ep)}\n\n"
            f"Shortlist from the rule-based pass (keep, adjust or ignore):\n"
            f"{_candidate_table(cands) or '(none)'}\n\n" + TASK.format(k=k))
    rejections = rejections if rejections is not None else []

    try:
        data = parse_json(llm.complete(SYSTEM, user))
        picks = data.get("promos", []) if isinstance(data, dict) else []
    except (LLMError, ValueError) as e:
        rejections.append({"episode": ep.number, "reason": f"call/parse failed: {e}"})
        return []

    out = []
    for n, pick in enumerate(picks[:k]):
        if not isinstance(pick, dict):
            continue
        promo = None
        cid = pick.get("candidate")
        segs = pick.get("segments") or []
        if cid and not segs:
            try:
                promo = cands[int(str(cid).lstrip("Cc")) - 1]
                promo = Promo(promo.kind, promo.segments)
            except (ValueError, IndexError):
                rejections.append({"episode": ep.number, "pick": n, "reason": f"unknown candidate {cid}",
                                   "raw": pick})
                continue
        else:
            built = []
            for pair in segs:
                try:
                    i, j = int(pair[0]), int(pair[1])
                except (TypeError, ValueError, IndexError):
                    built = None
                    break
                seg = _snap(ep, i, j)
                if seg is None:
                    built = None
                    break
                built.append(seg)
            if not built:
                rejections.append({"episode": ep.number, "pick": n,
                                   "reason": "segment indices out of range or unsnappable", "raw": pick})
                continue
            kind = "single" if len(built) == 1 else "cold_open"
            promo = Promo(kind, built)

        if not (MIN_LEN - 0.5 <= promo.length <= MAX_LEN + 0.5):
            rejections.append({"episode": ep.number, "pick": n,
                               "reason": f"length {promo.length:.1f}s outside 20-60s", "raw": pick})
            continue

        scorer.score(promo)
        describe(promo)                      # fills rule-based text as a fallback
        promo.source = "llm"
        promo.hook = str(pick.get("hook") or promo.hook)
        promo.cliffhanger = str(pick.get("cliffhanger") or promo.cliffhanger)
        promo.why = str(pick.get("why") or promo.why)
        llm_scores = {}
        for key in ("hook_score", "cliff_score", "standalone_score"):
            try:
                llm_scores[key] = max(1, min(10, int(pick.get(key, 5))))
            except (TypeError, ValueError):
                llm_scores[key] = 5
        promo.scores["llm"] = llm_scores
        promo.scores["llm_mean"] = round(sum(llm_scores.values()) / 30, 3)
        risk = str(pick.get("spoiler_risk", "")).lower()
        if risk in ("medium", "high"):
            promo.flags.append(f"spoiler risk: {risk}")
        # blend so the two passes stay comparable across episodes
        promo.total = round(0.5 * promo.total + 0.5 * promo.scores["llm_mean"], 4)
        out.append(promo)
    return out


RANK_SYSTEM = """You are the growth lead for a microdrama app deciding which promo clips to post
first on TikTok/Reels to bring in new viewers who have never heard of the show.
Reply with a single JSON object and nothing else."""


def rank_with_llm(llm: LLM, promos: list[Promo], show_title: str) -> list[tuple[Promo, str]] | None:
    rows = []
    for i, p in enumerate(promos, 1):
        rows.append(f"P{i}: eps {p.episodes}, {p.length:.0f}s. Hook: {p.hook} | "
                    f"Cuts on: {p.cliffhanger} | {p.why}")
    user = (f"Show title: {show_title}\n\nCandidate promos:\n" + "\n".join(rows) +
            "\n\nOrder them by which to post first. Consider: the hook, the cliffhanger, "
            "whether it sells the show's premise to a stranger, and whether it gives away "
            "a late twist. Return {\"order\": [{\"id\": \"P3\", \"reason\": \"one line\"}, ...]} "
            "covering every candidate exactly once.")
    try:
        data = parse_json(llm.complete(RANK_SYSTEM, user))
        order = data.get("order", [])
    except (LLMError, ValueError):
        return None
    seen, out = set(), []
    for item in order:
        try:
            idx = int(str(item.get("id", "")).lstrip("Pp")) - 1
        except (ValueError, AttributeError):
            continue
        if 0 <= idx < len(promos) and idx not in seen:
            seen.add(idx)
            out.append((promos[idx], str(item.get("reason", ""))))
    for idx, p in enumerate(promos):          # anything the model dropped goes last
        if idx not in seen:
            out.append((p, "not ranked by model"))
    return out
