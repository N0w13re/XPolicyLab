"""Bind official sorting instructions to basket destinations.

`classify_objects_by_language` differs from `classify_objects` only in that the
reward requires a specific category-to-basket assignment instead of accepting
any consistent permutation. That assignment exists solely in the instruction
text, so P1 has to read it to place objects, not just to pick them.
"""

from __future__ import annotations

import re


BASKET_SIDES = ("left", "middle", "right")

_ASSIGNMENT_PATTERN = re.compile(
    r"([A-Za-z][A-Za-z0-9_\- ]*?)\s+objects?\s+into\s+the\s+(left|middle|right)\s+basket",
    flags=re.IGNORECASE,
)


# The template wraps each category in connectives ("Put X objects...", "and Y
# objects..."), which the capture group would otherwise absorb.
_LEADING_STOPWORDS = ("put", "place", "and", "then", "the", "all")


def normalize_category(name: str) -> str:
    """Fold spacing and case so locator labels match instruction wording."""
    collapsed = re.sub(r"[\s\-]+", "_", name.strip().lower()).strip("_")
    parts = collapsed.split("_")
    while len(parts) > 1 and parts[0] in _LEADING_STOPWORDS:
        parts.pop(0)
    return "_".join(parts)


def parse_basket_assignment(instruction: str) -> dict[str, str]:
    """Map each named object category to the basket side it belongs in."""
    assignment: dict[str, str] = {}
    for raw_category, side in _ASSIGNMENT_PATTERN.findall(instruction or ""):
        category = normalize_category(raw_category)
        if category:
            assignment[category] = side.lower()
    return assignment


def destination_side(instruction: str, category_label: str | None) -> str | None:
    """Return the basket side for a located object, or None when unbound."""
    if not category_label:
        return None
    assignment = parse_basket_assignment(instruction)
    if not assignment:
        return None

    category = normalize_category(str(category_label))
    if category in assignment:
        return assignment[category]

    # Locator labels and instruction categories sometimes differ in
    # pluralization or add a qualifier ("toy car" vs "car").
    for known, side in assignment.items():
        if category in known or known in category:
            return side
    return None
