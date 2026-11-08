"""Quality checks for a finished run.

1. Hard rules every pick must pass (length, boundaries, no mid-line cuts).
2. Agreement with a small set of reference promos I picked by hand from the
   scripts before tuning anything. They are one person's taste, not ground
   truth, but they catch the tool drifting towards talky or confusing clips.
3. If a second run is given (e.g. --no-sting-hints), how many top picks moved.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .parser import load_show, to_seconds


def _segs(promo: dict) -> list[tuple[int, float, float]]:
    return [(s["episode"], to_seconds(s["start"]), to_seconds(s["end"]))
            for s in promo["segments"]]


def _inter(a, b) -> float:
    total = 0.0
    for ea, sa, xa in a:
        for eb, sb, xb in b:
            if ea == eb:
                total += max(0.0, min(xa, xb) - max(sa, sb))
    return total


def _len(segs) -> float:
    return sum(x - s for _, s, x in segs)


def iou(a, b) -> float:
    i = _inter(a, b)
    return i / max(1e-6, _len(a) + _len(b) - i)


def load_reference(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            segs = []
            for part in r["segments"].split(";"):
                ep, rng = part.split(":", 1)
                s, e = rng.split("-")
                segs.append((int(ep), to_seconds(s), to_seconds(e)))
            hep, ht = r["hook_at"].split(":", 1)
            rows.append({"id": r["id"], "segs": segs, "hook": (int(hep), to_seconds(ht)),
                         "note": r["note"]})
    return rows


def hard_rules(data: dict, show_dir: str | Path) -> list[dict]:
    eps = {e.number: e for e in load_show(show_dir)}
    problems = []
    for ep, promos in data["per_episode"].items():
        for i, p in enumerate(promos, 1):
            segs = _segs(p)
            total = _len(segs)
            if not 19.5 <= total <= 60.5:
                problems.append({"episode": ep, "pick": i, "issue": f"length {total:.1f}s"})
            for e, s, x in segs:
                lines = [b for b in eps[e].beats if b.is_line]
                if x > eps[e].duration + 0.01 or s < 0:
                    problems.append({"episode": ep, "pick": i, "issue": "outside episode"})
                if any(l.start < s - 0.05 < l.end - 0.1 for l in lines):
                    problems.append({"episode": ep, "pick": i, "issue": f"starts mid-line at {s}"})
                if any(l.start < x - 0.05 < l.end - 0.1 for l in lines):
                    problems.append({"episode": ep, "pick": i, "issue": f"ends mid-line at {x}"})
    return problems


def reference_agreement(data: dict, refs: list[dict]) -> tuple[list[dict], dict]:
    rows = []
    for r in refs:
        eps = {e for e, _, _ in r["segs"]}
        picks = []
        for ep, promos in data["per_episode"].items():
            for rank, p in enumerate(promos, 1):
                segs = _segs(p)
                if {e for e, _, _ in segs} & eps:
                    picks.append((rank, int(ep), segs, p))
        best = max(picks, key=lambda x: iou(x[2], r["segs"]), default=None)
        top1 = [x for x in picks if x[0] == 1 and x[1] == r["segs"][0][0]]
        hep, ht = r["hook"]
        covers_hook = any(any(e == hep and s - 0.5 <= ht <= x for e, s, x in segs)
                          for _, _, segs, _ in picks)
        rows.append({
            "id": r["id"],
            "note": r["note"],
            "best_iou": round(iou(best[2], r["segs"]), 2) if best else 0.0,
            "best_rank": best[0] if best else None,
            "top1_iou": round(iou(top1[0][2], r["segs"]), 2) if top1 else 0.0,
            "covers_hook_moment": covers_hook,
            "best_pick": " + ".join(f"ep{s['episode']} {s['start']}-{s['end']}"
                                    for s in best[3]["segments"]) if best else "",
        })
    n = max(1, len(rows))
    summary = {
        "references": len(rows),
        "hit@1 (IoU>=0.4)": sum(r["top1_iou"] >= 0.4 for r in rows),
        "hit@3 (IoU>=0.4)": sum(r["best_iou"] >= 0.4 for r in rows),
        "any overlap@3": sum(r["best_iou"] > 0 for r in rows),
        "hook moment inside a pick": sum(r["covers_hook_moment"] for r in rows),
        "mean best IoU": round(sum(r["best_iou"] for r in rows) / n, 2),
    }
    return rows, summary


def stability(data: dict, other: dict) -> dict:
    same_top, moved = 0, []
    for ep, promos in data["per_episode"].items():
        o = other["per_episode"].get(ep, [])
        if not promos or not o:
            continue
        if iou(_segs(promos[0]), _segs(o[0])) >= 0.5:
            same_top += 1
        else:
            moved.append(int(ep))
    return {"episodes": len(data["per_episode"]), "same_top_pick": same_top, "moved": moved}


def write_quality_report(run_dir: str | Path, show_dir: str | Path,
                         reference: str | Path | None = None,
                         compare_dir: str | Path | None = None) -> Path:
    run_dir = Path(run_dir)
    data = json.loads((run_dir / "promos.json").read_text(encoding="utf-8"))
    out = {"meta": data["meta"]}

    problems = hard_rules(data, show_dir)
    n_picks = sum(len(v) for v in data["per_episode"].values())
    flags = {}
    for promos in data["per_episode"].values():
        for p in promos:
            for f in p["flags"]:
                key = f.split(":")[0]
                flags[key] = flags.get(key, 0) + 1
    out["hard_rules"] = {"picks": n_picks, "violations": problems}
    out["soft_flags"] = flags

    lines = [f"# Quality report: {data['meta']['show']}", "",
             f"Run: {data['meta'].get('llm') or 'rules only'}; {n_picks} picks over "
             f"{data['meta']['episodes']} episodes.", "",
             "## Hard rules", "",
             f"{n_picks - len({(p['episode'], p['pick']) for p in problems})}/{n_picks} picks pass "
             "(20-60s, inside the episode, no line cut mid-sentence)."]
    for p in problems:
        lines.append(f"- ep {p['episode']} pick {p['pick']}: {p['issue']}")
    lines += ["", "## Soft flags raised by the tool", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(flags.items(), key=lambda x: -x[1])] or ["- none"]

    if reference and Path(reference).exists():
        rows, summary = reference_agreement(data, load_reference(reference))
        out["reference"] = {"summary": summary, "rows": rows}
        lines += ["", "## Agreement with hand-picked reference promos", ""]
        lines += [f"- {k}: {v}" for k, v in summary.items()]
        lines += ["", "| Ref | Best IoU | Rank | Hook moment covered | Tool's closest pick | What I picked |",
                  "|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['id']} | {r['best_iou']} | {r['best_rank']} | "
                         f"{'yes' if r['covers_hook_moment'] else 'no'} | {r['best_pick']} | "
                         f"{r['note']} |")

    if compare_dir:
        other = json.loads((Path(compare_dir) / "promos.json").read_text(encoding="utf-8"))
        st = stability(data, other)
        out["stability"] = st
        lines += ["", f"## Stability vs {Path(compare_dir).name}", "",
                  f"Top pick unchanged (IoU>=0.5) in {st['same_top_pick']}/{st['episodes']} episodes.",
                  f"Moved: {st['moved']}"]

    (run_dir / "quality.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    path = run_dir / "quality.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
