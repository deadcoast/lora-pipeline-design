"""One-word character and art-style tags shared by the desktop, pipeline, and library."""

from __future__ import annotations

import re

TAG_MAX_LENGTH = 32
_TAG = re.compile(r"^[a-z0-9][a-z0-9-]*$")
TAG_RULE = "One word: lowercase letters, numbers, hyphens."


def normalize_tag(text: str) -> str:
    """Lowercase and trim a tag so spelling variants collapse to one word."""
    return text.strip().lower()


def tag_error(text: str) -> str | None:
    """Return a plain-language problem with a normalized tag, or None."""
    tag = normalize_tag(text)
    if not tag:
        return "Enter one word."
    if len(tag) > TAG_MAX_LENGTH:
        return f"Use {TAG_MAX_LENGTH} characters or fewer."
    if _TAG.fullmatch(tag) is None:
        return TAG_RULE
    return None
