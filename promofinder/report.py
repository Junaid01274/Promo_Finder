"""Write a run to disk: JSON for machines, CSV for a sheet, Markdown for people."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .candidates import Promo
from .describe import segment_text
from .parser import fmt


def promo_dict(p: Promo) -> dict:
    return {
        "kind": p.kind,
        "source": p.source,
        "segments": [s.as_dict() for s in p.segments],
        "total_seconds": round(p.length, 2),
        "hook": p.hook,
        "cliffhanger": p.cliffhanger,
        "why": p.why,
        "score": p.total,
        "scores": p.scores,
        "flags": p.flags,
    }


def seg_str(p: Promo) -> str:
    return " + ".join(f"Ep {s.episode} {fmt(s.start)}-{fmt(s.end)}" for s in p.segments)


def write(result, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = {
        "meta": result.meta,
        "ranking": [dict(rank=i, post_score=s, reason=why, **promo_dict(p))
                    for i, (p, s, why) in enumerate(result.ranking, 1)],
        "per_episode": {str(ep): [promo_dict(p) for p in ps]
                        for ep, ps in result.per_episode.items()},
    }
    (out / "promos.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    with open(out / "promos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "pick", "kind", "segments", "seconds", "hook", "cliffhanger",
                    "why", "score", "hook_s", "cliff_s", "intensity_s", "standalone_s", "flags"])
        for ep, ps in result.per_episode.items():
            for i, p in enumerate(ps, 1):
                s = p.scores
                w.writerow([ep, i, p.kind, seg_str(p), round(p.length, 1), p.hook,
                            p.cliffhanger, p.why, p.total, s["hook"], s["cliff"],
                            s["intensity"], s["standalone"], "; ".join(p.flags)])

    lines = [f"# Post-first ranking: {result.show_title}", "",
             f"Run: {result.meta.get('llm') or 'rules only'}, "
             f"{result.meta['episodes']} episodes, {result.meta['candidates_scored']:,} "
             f"candidate clips scored in {result.meta['rules_seconds']}s.", "",
             "| # | Segments | Len | Hook (first 3s) | Cuts on | Why post it |",
             "|---|---|---|---|---|---|"]
    for i, (p, s, why) in enumerate(result.ranking, 1):
        cell = lambda t: t.replace("|", "/").replace("\n", " ")
        lines.append(f"| {i} | {seg_str(p)} | {p.length:.0f}s | {cell(p.hook)} | "
                     f"{cell(p.cliffhanger)} | {cell(why)} |")
    lines += ["", "## Transcripts of the ranked promos", ""]
    for i, (p, s, why) in enumerate(result.ranking, 1):
        lines += [f"### {i}. {seg_str(p)} ({p.kind}, {p.length:.0f}s)", "",
                  f"{p.why}", "", "```", segment_text(p), "```", ""]
    (out / "ranking.md").write_text("\n".join(lines), encoding="utf-8")

    lines = [f"# All picks by episode: {result.show_title}", ""]
    for ep, ps in result.per_episode.items():
        lines.append(f"## Episode {ep}")
        lines.append("")
        for i, p in enumerate(ps, 1):
            flags = f"  \n  Flags: {', '.join(p.flags)}" if p.flags else ""
            lines += [f"**{i}. {seg_str(p)}** ({p.kind}, {p.length:.0f}s, score {p.total:.2f})  ",
                      f"  Hook: {p.hook}  ",
                      f"  Cliffhanger: {p.cliffhanger}  ",
                      f"  Why: {p.why}{flags}", ""]
    (out / "all_picks.md").write_text("\n".join(lines), encoding="utf-8")

    (out / "run_meta.json").write_text(json.dumps(result.meta, indent=2), encoding="utf-8")
    if result.rejections:
        with open(out / "llm_rejections.jsonl", "w", encoding="utf-8") as f:
            for r in result.rejections:
                f.write(json.dumps(r) + "\n")
    return out
