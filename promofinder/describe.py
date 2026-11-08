"""Plain-language hook / cliffhanger / reason for rule-based picks.

When an LLM is plugged in it rewrites these. Without one, an editor still gets
something readable: the actual line that opens the clip, the line it cuts on,
and the two or three signals that pushed it up the list.
"""

from __future__ import annotations

import re

from .candidates import Promo
from .parser import Beat, fmt
from .signals import HOT_ACTION, is_cut_off, is_open_question, is_threat


def _name(speaker: str) -> str:
    return " ".join(w.capitalize() for w in speaker.split())


def quote(b: Beat | None, limit: int = 110) -> str:
    if b is None:
        return ""
    text = b.text if len(b.text) <= limit else b.text[: limit - 3].rstrip() + "..."
    if b.is_line:
        return f'{_name(b.speaker)}: "{text}"'
    return text


def _hook_kind(b: Beat | None) -> str:
    if b is None:
        return "a slow open"
    if not b.is_line:
        if HOT_ACTION.search(b.text):
            return "a physical moment"
        return "a visual beat"
    if is_threat(b):
        return "a threat"
    if is_open_question(b):
        return "a loaded question"
    if "!" in b.text:
        return "an outburst"
    return "a sharp line"


def _cliff_kind(p: Promo, b: Beat | None) -> str:
    if b is None:
        return "an open moment"
    if b.is_line and is_open_question(b):
        return "a question nobody answers"
    if b.is_line and is_threat(b):
        return "a threat left hanging"
    if b.is_line and is_cut_off(b):
        return "a line that breaks off"
    if not b.is_line and HOT_ACTION.search(b.text):
        return "the moment before the hit lands"
    return "an unresolved beat"


def describe(p: Promo) -> Promo:
    hook_b = getattr(p, "_hook_beat", None)
    cliff_b = getattr(p, "_cliff_beat", None)
    p.hook = quote(hook_b)
    p.cliffhanger = quote(cliff_b)

    s = p.scores
    bits = [f"opens on {_hook_kind(hook_b)}", f"cuts on {_cliff_kind(p, cliff_b)}"]
    if s["standalone"] >= 0.8:
        speakers = {b.speaker for b in p.all_beats() if b.is_line}
        bits.append(f"{len(speakers)} speakers, no backstory needed")
    elif s["standalone"] < 0.55:
        bits.append("leans on names/plot a new viewer won't know")
    if p.kind == "stitch":
        a, b = p.segments[0].episode, p.segments[-1].episode
        bits.append(f"joins the set-up in ep {a} to the payoff in ep {b}")
    elif p.kind == "cold_open":
        bits.append("flashes the peak first because the scene starts slow")
    p.why = "; ".join(bits).capitalize() + "."
    return p


def segment_text(p: Promo) -> str:
    """Human-readable transcript of a promo, used in reports and LLM prompts."""
    out = []
    for i, seg in enumerate(p.segments, 1):
        out.append(f"-- segment {i}: ep {seg.episode} {fmt(seg.start)} to {fmt(seg.end)}")
        for b in seg.beats:
            if b.kind == "camera":
                continue
            out.append(f"   {fmt(b.start)} {quote(b, 160)}")
    return "\n".join(out)


def premise_words(show_title: str) -> set[str]:
    """The title of a microdrama is its pitch. Pull the content words out of it."""
    stop = {"i", "a", "an", "the", "he", "she", "was", "is", "my", "his", "her", "and",
            "of", "to", "in", "on", "for", "with", "by", "at", "it", "be", "me"}
    words = {w.lower() for w in re.findall(r"[A-Za-z']+", show_title)} - stop
    stems = set()
    for w in words:
        stems.add(w)
        for suf in ("ied", "ed", "ing", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                stems.add(w[: -len(suf)])
    return stems
