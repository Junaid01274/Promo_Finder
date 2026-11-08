"""Generate promo candidates and score them with the rule-based signals.

Three shapes of promo are tried:

  single     one continuous stretch from one episode (most promos are this)
  stitch     the tail of episode N joined to a stretch of episode N+1, for
             set-ups that only pay off after the episode break
  cold_open  a 2-5 second flash of the episode's peak moment, then the run-up
             to it, for scenes where the strongest beat is not at the start
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parser import Beat, Episode, fmt
from .signals import (
    ARRIVAL, CONTINUATION_START, INTIMACY, RESOLUTION_WORDS, STING_HINT, CALM_ACTION,
    beat_heat, is_cut_off, is_open_question, is_reaction, is_threat,
)

MIN_LEN, MAX_LEN = 20.0, 60.0
HOOK_WINDOW = 3.0
WEIGHTS = {"hook": 0.32, "cliff": 0.30, "intensity": 0.18, "standalone": 0.20}


@dataclass
class Segment:
    episode: int
    start: float
    end: float
    beats: list[Beat]

    @property
    def length(self) -> float:
        return self.end - self.start

    def as_dict(self) -> dict:
        return {"episode": self.episode, "start": fmt(self.start), "end": fmt(self.end),
                "seconds": round(self.length, 2)}


@dataclass
class Promo:
    kind: str
    segments: list[Segment]
    scores: dict = field(default_factory=dict)
    total: float = 0.0
    hook: str = ""
    cliffhanger: str = ""
    why: str = ""
    flags: list[str] = field(default_factory=list)
    source: str = "rules"     # rules | llm

    @property
    def length(self) -> float:
        return sum(s.length for s in self.segments)

    @property
    def episodes(self) -> list[int]:
        return sorted({s.episode for s in self.segments})

    def all_beats(self) -> list[Beat]:
        return [b for s in self.segments for b in s.beats]

    def key(self) -> str:
        return "|".join(f"{s.episode}:{s.start:.2f}-{s.end:.2f}" for s in self.segments)


# ---------------------------------------------------------------- boundaries

def _lines(ep: Episode) -> list[Beat]:
    return [b for b in ep.beats if b.is_line]


_BOUNDS: dict[tuple[int, str], list[float]] = {}


def valid_starts(ep: Episode) -> list[float]:
    """Beat start times that don't fall in the middle of someone's line, or
    more than half a second into an action (no starting halfway through a slap)."""
    key = (id(ep), "s")
    if key not in _BOUNDS:
        lines = _lines(ep)
        actions = [b for b in ep.beats if b.kind == "action"]
        out = set()
        for b in ep.beats:
            s = b.start
            if any(l.start < s - 0.05 and s < l.end - 0.05 for l in lines):
                continue
            if any(a.start < s - 0.5 and s < a.end - 0.3 for a in actions):
                continue
            out.add(round(s, 2))
        _BOUNDS[key] = sorted(out)
    return _BOUNDS[key]


def valid_ends(ep: Episode) -> list[float]:
    """Beat end times that don't cut a line off mid-sentence."""
    key = (id(ep), "e")
    if key not in _BOUNDS:
        lines = _lines(ep)
        out = set()
        for b in ep.beats:
            e = b.end
            if any(l.start < e - 0.05 and e < l.end - 0.05 for l in lines):
                continue
            out.add(round(e, 2))
        out.add(round(ep.duration, 2))
        _BOUNDS[key] = sorted(out)
    return _BOUNDS[key]


def beats_between(ep: Episode, start: float, end: float) -> list[Beat]:
    return [b for b in ep.beats if b.start >= start - 0.01 and b.start < end - 0.01]


# ---------------------------------------------------------------- scoring parts

class Scorer:
    def __init__(self, episodes: list[Episode], use_sting_hints: bool = True,
                 premise: set[str] | None = None):
        self.eps = {e.number: e for e in episodes}
        self.use_sting_hints = use_sting_hints
        self.premise = premise or set()
        self.heat: dict[tuple[int, int], float] = {}
        for e in episodes:
            for b in e.beats:
                self.heat[(e.number, b.index)] = beat_heat(b, use_sting_hints)

    def h(self, b: Beat) -> float:
        return self.heat[(b.episode, b.index)]

    # The first 3 seconds -------------------------------------------------
    def hook(self, seg: Segment) -> tuple[float, Beat | None]:
        early = [b for b in seg.beats if b.start < seg.start + HOOK_WINDOW]
        if not early:
            return 0.0, None
        best = max(early, key=self.h)
        score = self.h(best)
        content = [b for b in early if b.kind != "camera"]
        if not content:
            score *= 0.7                      # nothing but a camera move up top
        # a second strong beat in the opening seconds helps (line + slap, etc.)
        others = sorted((self.h(b) for b in content if b is not best), reverse=True)
        if others:
            score += 0.15 * others[0]
        first_line = next((b for b in seg.beats if b.is_line), None)
        if first_line and first_line.start < seg.start + HOOK_WINDOW:
            if is_open_question(first_line) and not is_reaction(first_line):
                score += 0.05
        return min(1.0, score), best

    # The last few seconds ------------------------------------------------
    def cliff(self, seg: Segment) -> tuple[float, Beat | None]:
        ep = self.eps[seg.episode]
        tail = [b for b in seg.beats if b.end > seg.end - 4.5 and b.kind != "camera"]
        lines_tail = [b for b in tail if b.is_line]
        last_line = lines_tail[-1] if lines_tail else None
        score = 0.0
        content = [b for b in seg.beats if b.kind != "camera"]
        anchor = last_line or (tail[-1] if tail else None) or (content[-1] if content else None)

        if last_line:
            open_ = 0.0
            if is_open_question(last_line):
                open_ = 0.12 if is_reaction(last_line) else 0.25
            elif is_cut_off(last_line):
                open_ = 0.15
            if is_threat(last_line):
                open_ = max(open_, 0.25) + 0.5 * min(open_, 0.25)
            score += open_
            if RESOLUTION_WORDS.search(last_line.text):
                score -= 0.3
        if tail:
            score += 0.3 * max(self.h(b) for b in tail)
            if all(CALM_ACTION.search(b.text) for b in tail if not b.is_line) and \
               not any(self.h(b) > 0.5 for b in tail):
                score -= 0.15
            # Frozen danger or an arrival on screen is a cliffhanger even with no line:
            # a weapon stopped inches from a face, figures emerging from smoke.
            visual = [b for b in tail if b.kind in ("action", "sfx")]
            if any(self.h(b) >= 0.5 for b in visual) or any(ARRIVAL.search(b.text) for b in visual):
                score += 0.15
        # A big moment just before the tail (an explosion 6s from the end) still hangs.
        recent = [b for b in seg.beats if seg.end - 9 < b.end <= seg.end - 4.5 and b.kind != "camera"]
        if recent:
            score += 0.1 * max(self.h(b) for b in recent)

        # What the viewer is denied: the beat that would have come next.
        after = [b for b in ep.beats if b.start >= seg.end - 0.05 and
                 b.start < seg.end + 4.0 and b.kind != "camera"]
        if after:
            score += 0.25 * max(self.h(b) for b in after[:3])
        elif seg.end >= ep.duration - 6:
            score += 0.15                      # the episode's own ending

        if self.use_sting_hints and any(
                b.kind == "sfx" and STING_HINT.search(b.text) and b.start > seg.end - 4
                for b in seg.beats):
            score += 0.1
        return max(0.0, min(1.0, score)), anchor

    # The middle -------------------------------------------------------------
    def intensity(self, beats: list[Beat], start: float, end: float) -> float:
        content = [b for b in beats if b.kind != "camera"]
        if not content:
            return 0.0
        heats = sorted((self.h(b) for b in content), reverse=True)
        top = heats[: max(1, len(heats) // 2)]
        score = sum(top) / len(top)
        # dead air: longest stretch with no line, action or sound
        cover = sorted((b.start, b.end) for b in content)
        gap, cursor = 0.0, start
        for s, e in cover:
            gap = max(gap, s - cursor)
            cursor = max(cursor, e)
        gap = max(gap, end - cursor)
        if gap > 4.0:
            score -= min(0.3, 0.05 * (gap - 4.0))
        return max(0.0, min(1.0, score))

    # Would a stranger follow it? ---------------------------------------------
    def standalone(self, promo: Promo) -> float:
        score = 1.0
        beats = promo.all_beats()
        lines = [b for b in beats if b.is_line]
        first = promo.segments[0]
        ep = self.eps[first.episode]

        first_line = next((b for b in first.beats if b.is_line), None)
        if first_line and CONTINUATION_START.match(first_line.text):
            score -= 0.25
        prev = [b for b in ep.beats if b.is_line and b.end <= first.start + 0.05
                and b.end > first.start - 1.5]
        if prev and is_open_question(prev[-1]):
            score -= 0.2                       # opens on an answer to an unheard question

        words = [w for b in lines for w in re.findall(r"[A-Za-z']+", b.text)]
        if words:
            caps = 0
            for b in lines:
                toks = re.findall(r"[A-Za-z']+", b.text)
                for i, w in enumerate(toks[1:], 1):
                    if w[0].isupper() and w not in {"I", "I'm", "I'll", "I've", "God", "Father",
                                                    "Mother", "Your", "Grace", "Lady", "Lord"}:
                        caps += 1
            score -= min(0.3, 3.0 * caps / len(words))

        long_lines = sum(1 for b in lines if len(b.text.split()) > 16)
        score -= min(0.3, 0.08 * long_lines)

        speakers = {b.speaker for b in lines}
        if len(speakers) > 4:
            score -= 0.1 * (len(speakers) - 4)

        if promo.kind != "single":
            a, b = promo.segments[0], promo.segments[-1]
            sa = {x.speaker for x in a.beats if x.is_line}
            sb = {x.speaker for x in b.beats if x.is_line}
            score -= 0.05 if (sa & sb) else 0.15
        return max(0.0, min(1.0, score))

    def length_fit(self, seconds: float) -> float:
        if 25 <= seconds <= 45:
            return 1.0
        if seconds < 25:
            return 0.7 + 0.3 * max(0.0, (seconds - MIN_LEN) / 5.0)
        return 1.0 - 0.3 * min(1.0, (seconds - 45) / 15.0)

    # Put it together ---------------------------------------------------------
    def score(self, promo: Promo) -> Promo:
        first, last = promo.segments[0], promo.segments[-1]
        hook, hook_beat = self.hook(first)
        cliff, cliff_beat = self.cliff(last)
        beats = promo.all_beats()
        inten = sum(self.intensity(s.beats, s.start, s.end) * s.length for s in promo.segments)
        inten /= max(1e-6, promo.length)
        alone = self.standalone(promo)
        fit = self.length_fit(promo.length)

        # Small nudge for clips that say what the title promises ("married", "duke"...)
        words = {w.lower() for b in beats if b.is_line for w in re.findall(r"[A-Za-z']+", b.text)}
        echo = len({s for s in self.premise for w in words if w.startswith(s)})
        # Every cut is a place a cold viewer can get lost; continuous clips win ties.
        cut_tax = 0.04 * (len(promo.segments) - 1)

        promo.scores = {"hook": round(hook, 3), "cliff": round(cliff, 3),
                        "intensity": round(inten, 3), "standalone": round(alone, 3),
                        "length_fit": round(fit, 3), "title_echo": echo}
        promo.total = round(fit * sum(WEIGHTS[k] * promo.scores[k] for k in WEIGHTS)
                            + 0.02 * min(3, echo) - cut_tax, 4)
        promo._hook_beat = hook_beat          # used by describe.py
        promo._cliff_beat = cliff_beat

        flags = []
        if any(INTIMACY.search(b.text) for b in beats):
            flags.append("intimate content: check platform ad rules")
        if hook < 0.35:
            flags.append("weak first 3 seconds")
        if cliff < 0.3:
            flags.append("ending may feel resolved")
        promo.flags = flags
        return promo


# ---------------------------------------------------------------- generation

def _segment(ep: Episode, s: float, e: float) -> Segment:
    return Segment(ep.number, s, e, beats_between(ep, s, e))


def single_windows(ep: Episode, scorer: Scorer, lo=MIN_LEN, hi=MAX_LEN) -> list[Promo]:
    starts, ends = valid_starts(ep), valid_ends(ep)
    out = []
    for s in starts:
        for e in ends:
            if e - s < lo:
                continue
            if e - s > hi:
                break
            seg = _segment(ep, s, e)
            if not any(b.is_line for b in seg.beats):
                continue
            out.append(scorer.score(Promo("single", [seg])))
    return out


def _sub_windows(ep: Episode, lo: float, hi: float, region: tuple[float, float]) -> list[Segment]:
    starts, ends = valid_starts(ep), valid_ends(ep)
    segs = []
    for s in starts:
        if s < region[0] or s > region[1]:
            continue
        for e in ends:
            if e - s < lo:
                continue
            if e - s > hi:
                break
            seg = _segment(ep, s, e)
            if any(b.is_line for b in seg.beats):
                segs.append(seg)
    return segs


def _text_overlap(a: Segment, b: Segment) -> float:
    wa = {w.lower() for x in a.beats if x.is_line for w in re.findall(r"[a-z']+", x.text, re.I)}
    wb = {w.lower() for x in b.beats if x.is_line for w in re.findall(r"[a-z']+", x.text, re.I)}
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def stitches(ep_a: Episode, ep_b: Episode, scorer: Scorer, keep: int = 12) -> list[Promo]:
    """Tail of episode N + a stretch of N+1. Only tried for back-to-back episodes.

    A must run into the last quarter of N and B must start in the first 40% of
    N+1, otherwise the join skips too much story for a stranger to follow.
    """
    a_segs = [s for s in _sub_windows(ep_a, 8, 32, (ep_a.duration * 0.4, ep_a.duration))
              if s.end >= ep_a.duration * 0.75]
    b_segs = _sub_windows(ep_b, 10, 40, (0, ep_b.duration * 0.4))

    def a_rank(seg):
        hk, _ = scorer.hook(seg)
        return hk + scorer.intensity(seg.beats, seg.start, seg.end)

    def b_rank(seg):
        cl, _ = scorer.cliff(seg)
        return cl + 0.5 * scorer.intensity(seg.beats, seg.start, seg.end)

    a_top = sorted(a_segs, key=a_rank, reverse=True)[:keep]
    b_top = sorted(b_segs, key=b_rank, reverse=True)[:keep]
    out = []
    for a in a_top:
        for b in b_top:
            total = a.length + b.length
            if total < MIN_LEN or total > MAX_LEN:
                continue
            if _text_overlap(a, b) > 0.35:     # N+1 often replays N's last moment
                continue
            sa = {x.speaker for x in a.beats if x.is_line}
            sb = {x.speaker for x in b.beats if x.is_line}
            if not (sa & sb):                  # different storyline: never join them
                continue
            out.append(scorer.score(Promo("stitch", [a, b])))
    return out


def cold_opens(ep: Episode, singles: list[Promo], scorer: Scorer, keep: int = 10) -> list[Promo]:
    """Flash the peak beat first, then play the run-up to it."""
    starts, ends = valid_starts(ep), valid_ends(ep)
    weak_hook = [p for p in singles if p.scores["hook"] < 0.5 and p.scores["cliff"] >= 0.35]
    weak_hook.sort(key=lambda p: p.total, reverse=True)
    out = []
    for p in weak_hook[:keep]:
        seg = p.segments[0]
        # the flash has to come from the back half, or it just repeats the opening
        peaks = [b for b in seg.beats if b.kind in ("line", "action")
                 and b.start > seg.start + seg.length * 0.5 and b.end - b.start <= 5
                 and scorer.h(b) >= 0.55 and not is_reaction(b)]
        for peak in sorted(peaks, key=scorer.h, reverse=True)[:2]:
            fs = max((s for s in starts if s <= peak.start + 0.01), default=None)
            fe = min((e for e in ends if e >= peak.end - 0.01), default=None)
            if fs is None or fe is None or not (1.5 <= fe - fs <= 6):
                continue
            flash = _segment(ep, fs, fe)
            body = seg
            if flash.length + body.length > MAX_LEN:
                continue
            out.append(scorer.score(Promo("cold_open", [flash, body])))
    return out


# ---------------------------------------------------------------- de-duplication

def overlap(p: Promo, q: Promo) -> float:
    inter = 0.0
    for a in p.segments:
        for b in q.segments:
            if a.episode == b.episode:
                inter += max(0.0, min(a.end, b.end) - max(a.start, b.start))
    return inter / max(1e-6, min(p.length, q.length))


def non_max_suppress(promos: list[Promo], limit: int, max_overlap: float = 0.35) -> list[Promo]:
    kept: list[Promo] = []
    for p in sorted(promos, key=lambda x: x.total, reverse=True):
        if all(overlap(p, k) <= max_overlap for k in kept):
            kept.append(p)
        if len(kept) >= limit:
            break
    return kept
