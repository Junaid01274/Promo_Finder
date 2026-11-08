"""Cheap, explainable signals computed per beat.

Nothing here is clever. It is a set of word lists tuned by reading the 50
episodes in the pack, plus a few punctuation rules. The point is to get a
shortlist that is good enough for a human or an LLM to choose from, at zero
cost, in under a second per show.
"""

from __future__ import annotations

import re

from .parser import Beat

# Delivery notes, i.e. the words inside "(...)" after a speaker name.
HOT_TONES = {
    "screaming", "shouting", "furious", "enraged", "outraged", "livid", "roaring",
    "hysterical", "shrieking", "thunderous", "booming", "venomous", "vicious",
    "menacing", "threatening", "sinister", "chilling", "deadly", "lethal", "cruel",
    "taunting", "mocking", "gloating", "spiteful", "scornful", "contemptuous",
    "derisive", "sneering", "smug", "malicious", "vindictive", "ruthless", "callous",
    "accusing", "accusatory", "condemning", "desperate", "pleading", "begging",
    "sobbing", "weeping", "devastated", "heartbroken", "horrified", "terrified",
    "panicked", "frantic", "shocked", "stunned", "gasping", "choked", "choking",
    "agonized", "anguished", "betrayed", "defiant", "fierce", "incredulous",
    "triumphant", "dangerous", "cutting", "biting", "scathing", "piercing",
    "explosive", "seething", "unhinged", "maniacal", "sadistic", "predatory",
    "crying", "broken", "shattered", "alarmed", "urgent", "demanding", "commanding",
    "harsh", "cold", "icy", "steely", "dark", "ominous", "warning", "vengeful",
    "fury", "rage", "panic", "terror", "horror", "disbelieving", "raw",
}
CALM_TONES = {
    "gentle", "soft", "softly", "warm", "tender", "quiet", "quietly", "calm",
    "reassuring", "loving", "fond", "playful", "teasing", "amused", "relieved",
    "reflective", "wistful", "curious", "hopeful", "peaceful", "grateful",
    "comforting", "soothing", "affectionate", "lighthearted", "nostalgic",
    "sincere", "earnest", "humble", "polite", "intimate", "chuckling", "laughing",
}

# Words inside dialogue that tend to stop a scroll.
HOT_WORDS = re.compile(
    r"\b(hang(s|ed)?|gallows|noose|execut\w*|witch\w*|poison\w*|arsenic|kill\w*|"
    r"die[sd]?|dead|death|corpse|murder\w*|burn\w*|blood\w*|bleed\w*|arrest\w*|"
    r"impost[eo]r|traitor|treason|stole|steal\w*|thief|lie|lying|liar|betray\w*|"
    r"disown\w*|beggar|worthless|nobody|penniless|rat|filth|disgrace\w*|crawl|"
    r"marry\w*|married|marriage|wedding|bride|husband|wife|divorce|pregnant|"
    r"duke|duchess|title|heir|alive|ghost|ring|proof|confess\w*|guilty|"
    r"marry me|accept my hand|propos\w*|"
    r"slap\w*|strike|hurt|kneel|bow|chain\w*|cell|dungeon|prison\w*)\b",
    re.I,
)
# Physical, visual moments in ACTION / CAMERA / SOUND FX text.
HOT_ACTION = re.compile(
    r"\b(slap\w*|strik(e|es|ing)|struck|punch\w*|shove[sd]?|shoving|drag\w*|grab\w*|"
    r"fling\w*|flung|hurl\w*|throw\w*|threw|smash\w*|shatter\w*|explo\w*|"
    r"detonat\w*|crash\w*|collaps\w*|tackl\w*|lung\w*|chok\w*|throat|whip\w*|"
    r"crop|blood\w*|bleed\w*|gash\w*|lacerat\w*|thorn\w*|fire|flame\w*|blaze|"
    r"inferno|burn\w*|scald\w*|searing|branding|iron|chain\w*|shackle\w*|"
    r"manacle\w*|handcuff\w*|bursts?|slam\w*|yank\w*|snatch\w*|rip\w*|tear\w*|"
    r"seiz\w*|kneel\w*|one knee|gasp\w*|scream\w*|sob\w*|convuls\w*|faint\w*|"
    r"recoil\w*|freez\w*|frozen|shock\w*|horror|terror|weapon|sword|club|pipe|"
    r"violent\w*|brutal\w*|forcibl\w*|struggl\w*|stun\w*|"
    r"rifle|halberd|gavel|naked|kiss\w*|ring|propos\w*)\b",
    re.I,
)
# Someone or something arriving: the classic visual cliffhanger.
ARRIVAL = re.compile(
    r"\b(emerg\w*|silhouett\w*|bursts? (in|open|through)|storm(s|ed)? in|"
    r"doors? (burst|swing|fly|flies|bang)\w*|steps? through|appears?|smoke|rubble|"
    r"breach\w*|footsteps|approach\w*|latch|someone is coming)\b",
    re.I,
)
CALM_ACTION = re.compile(
    r"\b(gentl\w*|soft\w*|warm\w*|tender\w*|smil\w*|laugh\w*|embrac\w*|nestl\w*|"
    r"peace\w*|relief|reliev\w*|calm\w*|quiet\w*|fade[sd]? (out|to black)|"
    r"fading out|hearth|crackl\w*|sweep\w*|broom|tea|sleep\w*|rest\w*)\b",
    re.I,
)
# Writers of these scripts often tag the ending sting. Useful, but I keep it in a
# separate bucket so it can be switched off to see how much the tool leans on it.
STING_HINT = re.compile(r"\b(cliffhanger|sting|cut(s|ting)? (sharply )?to black|"
                        r"freezes?|abrupt)\b", re.I)

RESOLUTION_WORDS = re.compile(
    r"\b(thank you|thank god|you saved|i'm sorry|i love you|we'll see|"
    r"together|safe now|it is enough|i accept|let him rest|goodbye)\b",
    re.I,
)
CONTINUATION_START = re.compile(r"^(\.\.\.|and\b|but\b|so\b|then\b|because\b|or\b)", re.I)
# Kept narrow on purpose: an earlier version matched "breathless" and "passionately"
# and flagged an ordinary argument scene as sexual content.
INTIMACY = re.compile(r"\b(naked|nude|shirtless|undress\w*|sheets pulled|illicit|"
                      r"forbidden entanglement|passionate embrace)\b", re.I)


def tone_heat(tone: str) -> float:
    if not tone:
        return 0.0
    words = set(re.findall(r"[a-z\-]+", tone))
    hot = len(words & HOT_TONES)
    calm = len(words & CALM_TONES)
    return min(1.0, 0.45 * hot) - 0.35 * calm


def is_reaction(b: Beat) -> bool:
    """'No...', 'What?', 'Daniel...' - emotional, but they carry no information,
    so they can't hook a stranger on their own."""
    words = re.findall(r"[A-Za-z']+", b.text)
    return b.is_line and len(words) <= 2 and not HOT_WORDS.search(b.text)


def beat_heat(b: Beat, use_sting_hints: bool = True) -> float:
    """0..1: how much this beat would make a scrolling viewer look up.

    Weights are deliberately small so that only beats with several things going
    for them (loaded words AND a hot delivery AND punch) get near 1.
    """
    text = b.text
    if b.is_line:
        content = len(HOT_WORDS.findall(text))
        score = 0.1 + 0.2 * min(3, content)
        score += 0.5 * max(-0.4, tone_heat(b.tone))
        if "!" in text:
            score += 0.08
        if "?!" in text or "!?" in text:
            score += 0.05
        words = len(text.split())
        if words <= 8 and content:   # short AND loaded lands hardest
            score += 0.08
        if words > 18:               # speeches don't
            score -= 0.15
        if RESOLUTION_WORDS.search(text):
            score -= 0.25
        if is_reaction(b):
            score = min(score, 0.3)
    else:
        score = 0.05
        score += 0.2 * min(3, len(HOT_ACTION.findall(text)) + (1 if ARRIVAL.search(text) else 0))
        score -= 0.12 * min(2, len(CALM_ACTION.findall(text)))
        if b.kind == "sfx" and use_sting_hints and STING_HINT.search(text):
            score += 0.15
        if b.kind == "camera":
            score *= 0.6        # camera notes repeat what ACTION already says
    return max(0.0, min(1.0, score))


def is_open_question(b: Beat) -> bool:
    """'Why?' or 'Where did you get this ring? Speak to me!' - a question in the
    last couple of sentences of the line, not one already answered inside it."""
    if not b.is_line or "?" not in b.text:
        return False
    tail = re.split(r"(?<=[.!?])\s+", b.text.strip())[-2:]
    return any("?" in s for s in tail)


def is_cut_off(b: Beat) -> bool:
    t = b.text.rstrip()
    return b.is_line and (t.endswith("...") or t.endswith("—") or t.endswith("-"))


def is_threat(b: Beat) -> bool:
    if not b.is_line:
        return False
    t = b.text.lower()
    return bool(re.search(
        r"\b(or (else|your|you)|will (hang|die|burn|pay)|you will|i will (destroy|make)|"
        r"before (sunrise|sunset|the magistrate)|choose your|last chance|"
        r"if you|unless|take her|remove her|arrest|seize|guards!?|do not move|"
        r"one (flame|word|blow)|say goodbye)\b", t))
