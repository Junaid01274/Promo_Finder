"""Command line entry point.

    python -m promofinder run  "path/to/show folder" --out outputs/rules
    python -m promofinder run  "path/to/show folder" --provider gemini --out outputs/gemini
    python -m promofinder check outputs/rules --show "path/to/show folder"
"""

from __future__ import annotations

import argparse
import sys

from . import report
from .llm import LLM, PRESETS
from .pipeline import run


def _episodes(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="promofinder", description="Find promo clips in microdrama scripts.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="find and rank promos for one show")
    r.add_argument("show", help="folder of episode .txt screenplays")
    r.add_argument("--out", default="outputs/latest")
    r.add_argument("--provider", choices=["none", *PRESETS.keys()], default="none",
                   help="LLM for the second pass; 'none' = rules only, free and offline")
    r.add_argument("--model", default="", help="override the provider's default model")
    r.add_argument("--base-url", default="", help="any OpenAI-compatible endpoint")
    r.add_argument("--api-key-env", default="", help="env var that holds the key")
    r.add_argument("--per-episode", type=int, default=3)
    r.add_argument("--top", type=int, default=20, help="length of the post-first list")
    r.add_argument("--episodes", default=None, help="e.g. 1-10 or 3,7,12")
    r.add_argument("--workers", type=int, default=6, help="parallel model calls")
    r.add_argument("--no-sting-hints", action="store_true",
                   help="ignore SOUND FX notes that literally say 'cliffhanger' (robustness test)")

    c = sub.add_parser("check", help="quality report for a finished run")
    c.add_argument("run_dir")
    c.add_argument("--show", required=True)
    c.add_argument("--reference", default="eval/reference_promos.csv")
    c.add_argument("--compare", default=None, help="another run dir, to measure how many top picks moved")

    args = ap.parse_args(argv)

    if args.cmd == "run":
        llm = None
        if args.provider != "none":
            llm = LLM(args.provider, model=args.model, base_url=args.base_url,
                      api_key_env=args.api_key_env)
        res = run(args.show, llm=llm, per_ep=args.per_episode, top_n=args.top,
                  use_sting_hints=not args.no_sting_hints, episodes=_episodes(args.episodes),
                  workers=args.workers)
        out = report.write(res, args.out)
        m = res.meta
        print(f"{m['episodes']} episodes, {m['candidates_scored']:,} clips scored, "
              f"{sum(len(v) for v in res.per_episode.values())} picks -> {out}")
        if llm:
            print(f"LLM {llm.label}: {llm.calls} calls, {llm.cache_hits} cached, "
                  f"~{m['approx_tokens_in']:,} tokens in / ~{m['approx_tokens_out']:,} out, "
                  f"{m['llm_rejections']} picks rejected by validation")
        return 0

    if args.cmd == "check":
        from .checks import write_quality_report
        path = write_quality_report(args.run_dir, args.show, args.reference, args.compare)
        print(f"quality report -> {path}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
