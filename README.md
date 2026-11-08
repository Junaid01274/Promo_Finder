# Promo Finder

Reads microdrama screenplays and suggests promo clips: 20 to 60 seconds, a hook in
the first 3 seconds, a cut on a cliffhanger, and something a stranger can follow.
For each pick it gives the episode, the start and end of every segment, the hook,
the cliffhanger and a one-line reason, then ranks them in posting order.

Built for Kuku's AI Generalist assignment, Problem 1.

## Run it

Python 3.10+.

```bash
pip install -r requirements.txt
```

Rules only (free, offline, under a minute for 50 episodes):

```bash
python -m promofinder run "path/to/show folder" --out outputs/rules
```

With a model for the second pass (any one of these):

```bash
GEMINI_API_KEY=...    python -m promofinder run "path/to/show folder" --provider gemini --out outputs/gemini
GROQ_API_KEY=...      python -m promofinder run "path/to/show folder" --provider groq
ANTHROPIC_API_KEY=... python -m promofinder run "path/to/show folder" --provider anthropic --model claude-opus-5-5
python -m promofinder run "path/to/show folder" --provider ollama --model llama3.1
```

Model names go stale. When I ran this (Nov 2026) on a new free Gemini key,
`gemini-2.5-flash` had been retired, `gemini-3.8-flash` was overloaded, `gemini-3.5-flash`
worked well, and the Pro models needed billing. The `gemini` default is the
`gemini-flash-latest` alias; pin one with `--model` for a reproducible run.

On Windows PowerShell, set the key first with `$env:GEMINI_API_KEY="..."`. Or copy
`.env.example` to `.env` and put the key there; the tool reads it, and git ignores it.

Any other OpenAI-compatible endpoint (an in-house gateway, OpenRouter, vLLM...):

```bash
python -m promofinder run "path/to/show folder" --provider openai --base-url https://gateway.example/v1 --api-key-env MY_KEY --model some-model
```

Quality report for a finished run (hard-rule checks, agreement with the reference
picks, and optionally how many top picks moved compared with another run):

```bash
python -m promofinder check outputs/rules --show "path/to/show folder" --compare outputs/rules_no_sting_hints
```

Web app (upload scripts, paste a key if you want the model pass):

```bash
streamlit run app.py
```

Tests (no input pack or API key needed; they use a two-episode sample show written for this repo):

```bash
python tests/test_promofinder.py
```

## Input format

One folder per show, one `.txt` per episode, in the layout of the input pack:
`SCENE` blocks, then beats headed `**CAMERA:**`, `**ACTION:**`, `**SOUND FX:**`
or `**SPEAKER (delivery note):**`, each followed by `Start: (mm:ss.cc) | End: (mm:ss.cc)`
and the text. The folder name is treated as the show title.

## How it works

1. **Parse** every script into timed beats (3,727 for the 50-episode pack).
2. **Score each beat** with plain rules: loaded words (hang, poison, married, duke...),
   the delivery note ("Cold, Menacing" vs "Soft, Warm"), punctuation, and physical
   verbs in action lines. One-word reactions like "No..." are capped, because they
   can't hook a stranger on their own.
3. **Generate every legal clip**: every start and end where no line is cut, 20 to 60s
   long (~48,700 for the pack). Two multi-part shapes are tried as well:
   *stitches* (end of episode N + start of N+1, only when the same characters are in
   both) and *cold opens* (a 2-5s flash of the scene's peak, then the run-up to it).
4. **Score each clip**: hook 32%, cliffhanger 30%, intensity 18%, stands-alone 20%,
   a small nudge for clips that say what the title promises, and a small tax per cut.
5. **Keep the top 3** non-overlapping clips per episode, then rank across the show.
   The posting order also weighs episode position (new viewers start at episode 1)
   and holds back intimate scenes.
6. **Optional model pass.** The model sees numbered beats and the rule shortlist and
   answers with beat numbers, never timestamps. Code turns those into times, snaps
   them so no line is cut, rejects anything outside 20-60s, and logs every
   rejection to `llm_rejections.jsonl`. One last call reorders the top 20. All
   responses are cached in `.cache/llm`, so a re-run is free and repeatable.

## Outputs

| File | What it is |
|---|---|
| `ranking.md` | Top 20 in posting order with reasons, and the transcript of each clip |
| `all_picks.md` | Every pick, episode by episode, readable |
| `promos.csv` | Same, as a sheet |
| `promos.json` | Everything, including per-signal scores and flags |
| `quality.md` / `quality.json` | Output of `check` |
| `run_meta.json` | Timings, counts, model used, tokens |

## Layout

```
promofinder/
  parser.py      script -> timed beats
  signals.py     word lists and per-beat heat
  candidates.py  clip generation, scoring, de-duplication
  describe.py    hook / cliffhanger / reason text
  llm.py         provider adapters + disk cache
  judge.py       model pass with validation
  pipeline.py    end-to-end run and posting order
  checks.py      quality report
  report.py      writes the output files
  cli.py
app.py           Streamlit front end
eval/reference_promos.csv   20 hand-picked reference promos
tests/
```

## Confidentiality

The input pack is Kuku's content. `.gitignore` keeps the scripts, the `outputs/`
folder, the solution doc and `eval/reference_promos.csv` out of git, because they all
quote the show. They're shared through the submission's Drive link instead. Without the
reference file, `check` still runs the hard-rule checks and simply skips that section.
