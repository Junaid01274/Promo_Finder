"""Small web front end: upload a show's scripts, get ranked promos.

    streamlit run app.py
"""

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from promofinder import report
from promofinder.describe import segment_text
from promofinder.llm import LLM, PRESETS
from promofinder.pipeline import run

SAMPLE = Path(__file__).parent / "tests" / "fixtures" / "The Contract Bride"

st.set_page_config(page_title="Promo Finder", layout="wide")
st.title("Promo Finder")
st.caption("Reads microdrama screenplays and suggests 20-60s promo clips: "
           "hook in the first 3 seconds, cut on a cliffhanger, works for a stranger.")

with st.sidebar:
    st.header("Input")
    title = st.text_input("Show title", "The Contract Bride",
                          help="Used to work out the show's premise ('married', 'duke'...).")
    files = st.file_uploader("Episode screenplays (.txt)", type="txt", accept_multiple_files=True)
    st.caption("No files? The bundled two-episode sample show is used.")

    st.header("Model")
    provider = st.selectbox("Second pass", ["none (rules only, free)", *PRESETS.keys()])
    model = st.text_input("Model override", "")
    key = st.text_input("API key", type="password",
                        help="Only used for this run, never stored.") if not provider.startswith("none") else ""
    per_ep = st.slider("Picks per episode", 1, 5, 3)
    go = st.button("Find promos", type="primary")

if go:
    with tempfile.TemporaryDirectory() as tmp:
        show_dir = Path(tmp) / (title.strip() or "Untitled show")
        show_dir.mkdir()
        if files:
            for f in files:
                (show_dir / f.name).write_bytes(f.getvalue())
        else:
            for p in SAMPLE.glob("*.txt"):
                (show_dir / p.name).write_bytes(p.read_bytes())

        llm = None
        if not provider.startswith("none"):
            env = PRESETS[provider][2]
            if env and key:
                os.environ[env] = key
            llm = LLM(provider, model=model, cache_dir=Path(tmp) / "cache")

        log = st.empty()
        with st.spinner("Scoring every possible clip..."):
            res = run(show_dir, llm=llm, per_ep=per_ep, progress=lambda m: log.text(m))
        log.empty()
        out = report.write(res, Path(tmp) / "out")
        promos_json = (out / "promos.json").read_text(encoding="utf-8")
        promos_csv = (out / "promos.csv").read_text(encoding="utf-8")

    m = res.meta
    c1, c2, c3 = st.columns(3)
    c1.metric("Episodes", m["episodes"])
    c2.metric("Clips scored", f"{m['candidates_scored']:,}")
    c3.metric("Seconds", m["total_seconds"])

    st.subheader("Post these first")
    rows = []
    for i, (p, s, why) in enumerate(res.ranking, 1):
        rows.append({"#": i, "Segments": report.seg_str(p), "Length": f"{p.length:.0f}s",
                     "Hook": p.hook, "Cuts on": p.cliffhanger, "Why": why,
                     "Flags": ", ".join(p.flags)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("Every pick, by episode")
    for ep, promos in res.per_episode.items():
        with st.expander(f"Episode {ep}  ({len(promos)} picks)"):
            for i, p in enumerate(promos, 1):
                st.markdown(f"**{i}. {report.seg_str(p)}** · {p.kind} · {p.length:.0f}s · "
                            f"score {p.total:.2f}")
                st.markdown(f"Hook: {p.hook}  \nCuts on: {p.cliffhanger}  \nWhy: {p.why}")
                if p.flags:
                    st.warning(", ".join(p.flags))
                st.code(segment_text(p), language=None)

    st.download_button("Download JSON", promos_json, "promos.json", "application/json")
    st.download_button("Download CSV", promos_csv, "promos.csv", "text/csv")
    if res.rejections:
        with st.expander(f"{len(res.rejections)} model picks rejected by validation"):
            st.code(json.dumps(res.rejections, indent=2), language="json")
