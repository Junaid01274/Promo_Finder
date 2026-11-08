"""Offline tests. Run with `python -m pytest tests` or `python tests/test_promofinder.py`.

They use a two-episode sample show written for this repo (tests/fixtures), so they
run without Kuku's input pack and without any API key.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from promofinder.candidates import Scorer, non_max_suppress, single_windows, valid_ends, valid_starts
from promofinder.judge import judge_episode, rank_with_llm
from promofinder.parser import load_show, to_seconds
from promofinder.pipeline import run

SHOW = ROOT / "tests" / "fixtures" / "The Contract Bride"


class FakeLLM:
    """Stands in for a model. Returns whatever JSON the test queues up."""
    label = "fake:model"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = self.cache_hits = self.chars_in = self.chars_out = 0

    def complete(self, system, user):
        self.calls += 1
        return self.replies.pop(0) if self.replies else '{"promos": []}'


def test_parser_reads_every_beat():
    eps = load_show(SHOW)
    assert [e.number for e in eps] == [1, 2]
    ep1 = eps[0]
    assert len(ep1.beats) == 24
    assert ep1.duration == to_seconds("01:12.00")
    line = next(b for b in ep1.beats if b.speaker == "LILA")
    assert line.tone == "smug, gloating"
    assert line.text.startswith("He chose me months ago")


def test_boundaries_never_cut_a_line():
    for ep in load_show(SHOW):
        lines = [b for b in ep.beats if b.is_line]
        for s in valid_starts(ep):
            assert not any(l.start < s - 0.05 and s < l.end - 0.05 for l in lines)
        for e in valid_ends(ep):
            assert not any(l.start < e - 0.05 and e < l.end - 0.05 for l in lines)


def test_candidates_respect_length_and_find_the_slap():
    eps = load_show(SHOW)
    scorer = Scorer(eps)
    picks = non_max_suppress(single_windows(eps[0], scorer), 3)
    assert picks
    for p in picks:
        assert 20 <= p.length <= 60
    # the slap at 00:38 should sit inside at least one of the top picks
    assert any(p.segments[0].start <= 38.0 <= p.segments[0].end for p in picks)


def test_llm_picks_are_validated():
    eps = load_show(SHOW)
    scorer = Scorer(eps)
    ep = eps[0]
    cands = non_max_suppress(single_windows(ep, scorer), 3)
    reply = json.dumps({"promos": [
        {"candidate": "C1", "segments": [], "hook": "h", "cliffhanger": "c", "why": "w",
         "hook_score": 9, "cliff_score": 8, "standalone_score": 7, "spoiler_risk": "low"},
        {"candidate": None, "segments": [[0, 2]], "hook": "too short"},
        {"candidate": None, "segments": [[5, 999]], "hook": "made-up index"},
    ]})
    rejections = []
    out = judge_episode(FakeLLM([reply]), ep, cands, scorer, "The Contract Bride", rejections=rejections)
    assert len(out) == 1 and out[0].source == "llm"
    assert out[0].scores["llm"] == {"hook_score": 9, "cliff_score": 8, "standalone_score": 7}
    reasons = " | ".join(r["reason"] for r in rejections)
    assert "outside 20-60s" in reasons and "out of range" in reasons


def test_garbage_from_the_model_falls_back_to_rules():
    res = run(SHOW, llm=FakeLLM(["sorry, I can't help with that", "not json either"]),
              progress=lambda *_: None)
    assert all(res.per_episode[e] for e in (1, 2))      # rules picks fill the gap
    assert any("parse failed" in r["reason"] for r in res.rejections)


def test_rank_with_llm_handles_missing_ids():
    eps = load_show(SHOW)
    scorer = Scorer(eps)
    cands = non_max_suppress(single_windows(eps[0], scorer), 3)
    reply = json.dumps({"order": [{"id": "P2", "reason": "best"}, {"id": "P9", "reason": "bogus"}]})
    ranked = rank_with_llm(FakeLLM([reply]), cands, "The Contract Bride")
    assert ranked[0][0] is cands[1]
    assert len(ranked) == len(cands)                      # nothing dropped


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
