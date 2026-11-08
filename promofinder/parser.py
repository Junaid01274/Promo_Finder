"""Turn one screenplay .txt file into a list of timed beats.

The scripts in the pack all follow the same layout:

    SCENE 01
    Scene heading: INT. HOTEL BALLROOM - NIGHT
    Scene start: 00:00.00
    ...
    **CAMERA:**                         <- or ACTION, SOUND FX, or a speaker
    Start: (00:00.00) | End: (00:04.80)
    text of the beat

A speaker header looks like **MARA (Shocked, Breathless):** and the
words in brackets are the delivery note, which turns out to be one of the most
useful signals we have.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

TIME_RE = re.compile(r"(\d+):(\d{2})\.(\d{2})")
BEAT_HEADER_RE = re.compile(r"^\*\*(.+?):\*\*\s*$")
BEAT_TIME_RE = re.compile(r"^Start:\s*\((.+?)\)\s*\|\s*End:\s*\((.+?)\)")
SCENE_RE = re.compile(r"^SCENE\s+(\d+)")
EPISODE_NUM_RE = re.compile(r"episode-(\d+)", re.I)

NON_SPEAKER = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sfx"}


def to_seconds(stamp: str) -> float:
    m = TIME_RE.search(stamp)
    if not m:
        raise ValueError(f"bad timestamp: {stamp!r}")
    mins, secs, hund = (int(g) for g in m.groups())
    return mins * 60 + secs + hund / 100


def fmt(seconds: float) -> str:
    """Seconds back to the script's own mm:ss.cc format."""
    seconds = max(0.0, seconds)
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"


@dataclass
class Beat:
    episode: int
    scene: int
    index: int          # position inside the episode, 0-based
    kind: str           # camera | action | sfx | line
    start: float
    end: float
    text: str
    speaker: str = ""
    tone: str = ""      # delivery note for dialogue, lowercased

    @property
    def is_line(self) -> bool:
        return self.kind == "line"

    def short(self) -> str:
        if self.is_line:
            tone = f" ({self.tone})" if self.tone else ""
            return f"{self.speaker}{tone}: {self.text}"
        return f"[{self.kind.upper()}] {self.text}"


@dataclass
class Scene:
    number: int
    heading: str
    start: float
    end: float


@dataclass
class Episode:
    number: int
    source: str
    logline: str
    continuity: str
    scenes: list[Scene] = field(default_factory=list)
    beats: list[Beat] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max((b.end for b in self.beats), default=0.0)

    def lines(self) -> list[Beat]:
        return [b for b in self.beats if b.is_line]


def _split_speaker(header: str) -> tuple[str, str]:
    """'MARA (O.S.) (Firm, Cold)' -> ('MARA', 'o.s., firm, cold')."""
    notes = re.findall(r"\(([^()]*)\)", header)
    name = re.sub(r"\([^()]*\)", "", header).strip()
    return name, ", ".join(n.strip() for n in notes).lower()


def parse_episode(path: str | Path) -> Episode:
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()

    number = 0
    m = EPISODE_NUM_RE.search(path.name)
    if m:
        number = int(m.group(1))
    logline = continuity = ""

    ep = Episode(number=number, source=path.name, logline="", continuity="")
    scene_no = 0
    heading = ""
    scene_start = 0.0

    i = 0
    while i < len(raw):
        line = raw[i].strip()

        if line.startswith("EPISODE NUMBER:"):
            digits = re.findall(r"\d+", line)
            if digits and not number:
                ep.number = number = int(digits[0])
        elif line.startswith("EPISODE LOGLINE:"):
            ep.logline = line.split(":", 1)[1].strip()
        elif line.startswith("CONTINUITY NOTES:"):
            ep.continuity = line.split(":", 1)[1].strip()
        elif SCENE_RE.match(line):
            scene_no = int(SCENE_RE.match(line).group(1))
        elif line.startswith("Scene heading:"):
            heading = line.split(":", 1)[1].strip()
        elif line.startswith("Scene start:"):
            scene_start = to_seconds(line)
        elif line.startswith("Scene end:"):
            ep.scenes.append(Scene(scene_no, heading, scene_start, to_seconds(line)))
        else:
            hm = BEAT_HEADER_RE.match(line)
            if hm and i + 1 < len(raw):
                tm = BEAT_TIME_RE.match(raw[i + 1].strip())
                if tm:
                    header = hm.group(1).strip()
                    # text runs until the next blank line
                    j = i + 2
                    body = []
                    while j < len(raw) and raw[j].strip():
                        body.append(raw[j].strip())
                        j += 1
                    text = " ".join(body).strip().strip('"').strip()

                    if header.upper() in NON_SPEAKER:
                        kind, speaker, tone = NON_SPEAKER[header.upper()], "", ""
                    else:
                        kind = "line"
                        speaker, tone = _split_speaker(header)

                    ep.beats.append(Beat(
                        episode=ep.number, scene=scene_no, index=len(ep.beats),
                        kind=kind, start=to_seconds(tm.group(1)), end=to_seconds(tm.group(2)),
                        text=text, speaker=speaker, tone=tone,
                    ))
                    i = j
                    continue
        i += 1

    # Beats are written in roughly time order but camera and dialogue overlap.
    # Sort by start so window logic can rely on it, then re-index.
    ep.beats.sort(key=lambda b: (b.start, b.end))
    for k, b in enumerate(ep.beats):
        b.index = k
    return ep


def load_show(folder: str | Path) -> list[Episode]:
    folder = Path(folder)
    files = sorted(p for p in folder.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no .txt screenplays in {folder}")
    eps = [parse_episode(p) for p in files]
    eps.sort(key=lambda e: e.number)
    return eps
