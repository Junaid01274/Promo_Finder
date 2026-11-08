"""End-to-end run: parse -> candidates -> (optional LLM) -> rank -> report."""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .candidates import (Promo, Scorer, cold_opens, non_max_suppress, overlap,
                         single_windows, stitches)
from .describe import describe, premise_words
from .judge import judge_episode, rank_with_llm
from .llm import LLM
from .parser import Episode, load_show


@dataclass
class RunResult:
    show_title: str
    episodes: list[Episode]
    per_episode: dict[int, list[Promo]]
    ranking: list[tuple[Promo, float, str]]
    meta: dict = field(default_factory=dict)
    rejections: list = field(default_factory=list)


def find_candidates(eps: list[Episode], scorer: Scorer, per_ep: int = 3) -> dict[int, list[Promo]]:
    by_ep: dict[int, list[Promo]] = {}
    numbers = [e.number for e in eps]
    lookup = {e.number: e for e in eps}
    for e in eps:
        singles = single_windows(e, scorer)
        pool = list(singles)
        pool += cold_opens(e, singles, scorer)
        nxt = e.number + 1
        if nxt in lookup:
            pool += stitches(e, lookup[nxt], scorer)
        by_ep[e.number] = pool
    return by_ep


def post_first_score(p: Promo, n_eps: int, first_ep: int, premise: set[str]) -> tuple[float, str]:
    """Which promo goes out first. Rules, written down so they can be argued with."""
    words = {w.lower() for b in p.all_beats() if b.is_line
             for w in re.findall(r"[A-Za-z']+", b.text)}
    hits = sorted({w for w in words if any(w.startswith(s) for s in premise)})
    premise_boost = 1 + 0.04 * min(2, len(hits))
    ep = p.segments[0].episode
    funnel = 1.0 - 0.25 * (ep - first_ep) / max(1, n_eps - 1)
    safety = 0.6 if any("intimate" in f for f in p.flags) else 1.0
    score = p.total * premise_boost * funnel * safety

    s = p.scores
    def word(x):
        return "strong" if x >= 0.75 else "decent" if x >= 0.5 else "weak"
    reasons = [f"{word(s['hook'])} hook ({s['hook']:.2f}), {word(s['cliff'])} cliffhanger ({s['cliff']:.2f})"]
    if hits:
        reasons.append("says what the title promises: " + ", ".join(f"'{h}'" for h in hits[:3]))
    if ep - first_ep <= max(3, n_eps // 10):
        reasons.append(f"ep {ep}, so a new viewer lands near the start")
    elif ep - first_ep > n_eps * 0.6:
        reasons.append(f"ep {ep} of {n_eps}: may give away a twist")
    if s["standalone"] < 0.6:
        reasons.append("needs a little context")
    if safety < 1:
        reasons.append("intimate content, held back")
    return round(score, 4), "; ".join(reasons)


def run(show_dir: str | Path, llm: LLM | None = None, per_ep: int = 3, top_n: int = 20,
        use_sting_hints: bool = True, shortlist: int = 6, episodes: list[int] | None = None,
        progress=print, workers: int = 6) -> RunResult:
    t0 = time.time()
    show_dir = Path(show_dir)
    show_title = show_dir.name.strip()
    eps = load_show(show_dir)
    if episodes:
        keep = set(episodes)
        eps = [e for e in eps if e.number in keep]
    premise = premise_words(show_title)
    scorer = Scorer(eps, use_sting_hints=use_sting_hints, premise=premise)

    pools = find_candidates(eps, scorer)
    n_candidates = sum(len(v) for v in pools.values())
    t_rules = time.time() - t0

    per_episode: dict[int, list[Promo]] = {}
    rejections: list = []
    shortlists = {}
    for e in eps:
        top = non_max_suppress(pools[e.number], limit=max(per_ep, shortlist))
        for p in top:
            describe(p)
        shortlists[e.number] = top
        per_episode[e.number] = top[:per_ep]

    if llm is not None:
        # Episodes are independent, so ask about several at once.
        def ask(e):
            local: list = []
            picked = judge_episode(llm, e, shortlists[e.number][:shortlist], scorer, show_title,
                                   k=per_ep, rejections=local)
            return e.number, picked, local

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(ask, e) for e in eps]
            for done, fut in enumerate(as_completed(futures), 1):
                n, picked, local = fut.result()
                rejections.extend(local)
                progress(f"  {done}/{len(eps)} episodes judged by {llm.label}")
                if picked:                     # otherwise keep the rule picks
                    per_episode[n] = non_max_suppress(picked, limit=per_ep, max_overlap=0.6) \
                        if len(picked) > 1 else picked
        per_episode = dict(sorted(per_episode.items()))
        rejections.sort(key=lambda r: r.get("episode", 0))

    first_ep = eps[0].number if eps else 1
    everything = [p for ps in per_episode.values() for p in ps]
    scored = [(p, *post_first_score(p, len(eps), first_ep, premise)) for p in everything]
    scored.sort(key=lambda x: x[1], reverse=True)
    # one promo per episode in the posting list, so the feed isn't five cuts of one scene
    ranking, used = [], set()
    for p, s, why in scored:
        if p.segments[0].episode in used or any(overlap(p, q) > 0.2 for q, _, _ in ranking):
            continue
        used.add(p.segments[0].episode)
        ranking.append((p, s, why))
        if len(ranking) >= top_n:
            break

    if llm is not None and ranking:
        progress(f"  final ranking pass with {llm.label}")
        reordered = rank_with_llm(llm, [p for p, _, _ in ranking], show_title)
        if reordered:
            lookup = {id(p): (s, why) for p, s, why in ranking}
            ranking = [(p, lookup[id(p)][0], f"{reason} [rules: {lookup[id(p)][1]}]")
                       for p, reason in reordered]

    meta = {
        "show": show_title,
        "episodes": len(eps),
        "beats": sum(len(e.beats) for e in eps),
        "candidates_scored": n_candidates,
        "rules_seconds": round(t_rules, 2),
        "total_seconds": round(time.time() - t0, 2),
        "use_sting_hints": use_sting_hints,
        "llm": llm.label if llm else None,
    }
    if llm is not None:
        meta.update({"llm_calls": llm.calls, "llm_cache_hits": llm.cache_hits,
                     "approx_tokens_in": llm.chars_in // 4,
                     "approx_tokens_out": llm.chars_out // 4,
                     "llm_rejections": len(rejections)})
    return RunResult(show_title, eps, per_episode, ranking, meta, rejections)
