"""
feedback_handler.py
Looks at a new user message and decides: is this a CORRECTION of what the
agent just said, or is it just a normal new question?

This uses simple pattern matching (checking for phrases like "no,",
"actually", "always...", "should be...") rather than an LLM call, because
this classification is fast, cheap, and doesn't need deep understanding 
just recognizing a handful of common correcting someone phrasings.
"""

from __future__ import annotations

import re


CORRECTION_PATTERNS = [
    re.compile(r"^no[,.]?\s", re.I),
    re.compile(r"^actually[,.]?\s", re.I),
    re.compile(r"^that'?s (wrong|incorrect|not right)", re.I),
    re.compile(r"^please always\s", re.I),
    re.compile(r"^always\s", re.I),
    re.compile(r"^never\s", re.I),
    re.compile(r"^from now on[,]?\s", re.I),
    re.compile(r"^it should be\s", re.I),
    re.compile(r"\bnot\s+[\w/.\-]+[.!]?$", re.I),  
]


def is_correction(user_message: str) -> bool:
    text = user_message.strip()
    return any(pattern.search(text) for pattern in CORRECTION_PATTERNS)



