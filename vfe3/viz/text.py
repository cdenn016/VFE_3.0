r"""Native-script text preparation for Matplotlib figure labels.

Decoded strings stay in logical Unicode order until the final display boundary. Japanese needs a
font with CJK glyph coverage; Arabic additionally needs contextual shaping and bidirectional layout
because Matplotlib's standard text renderer does not perform those operations.
"""

from typing import Callable, Optional, Sequence


MULTILINGUAL_SANS_SERIF = (
    "Noto Sans",
    "Yu Gothic",
    "Meiryo",
    "Noto Sans CJK JP",
    "IPAexGothic",
    "Noto Sans Arabic",
    "Noto Naskh Arabic",
    "DejaVu Sans",
)

_NON_ENGLISH_LINGUISTIC_DATASETS = frozenset(("wiki-ja", "wiki-ar"))
_ARABIC_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)


def supports_english_linguistic_taxonomies(dataset: str) -> bool:
    """Return whether English BPE-case and function/content diagnostics are valid."""
    return dataset not in _NON_ENGLISH_LINGUISTIC_DATASETS


def installed_multilingual_sans_serif() -> list[str]:
    """Return the configured fallback families that Matplotlib can resolve on this machine.

    A family counts as installed only when it has an upright regular-weight face. Matplotlib
    otherwise substitutes the nearest weight, so a machine with only ``NotoSans-Bold.ttf`` renders
    every normal-weight label bold. When the leading Latin family is unavailable, DejaVu Sans takes
    its place at the front, so Latin text keeps a Latin face that matches the mathtext glyphs
    instead of falling through to a CJK family.
    """
    from matplotlib import font_manager

    regular = set()
    for entry in font_manager.fontManager.ttflist:
        weight = font_manager.weight_dict.get(entry.weight, entry.weight)
        if entry.style == "normal" and isinstance(weight, int) and 350 <= weight <= 500:
            regular.add(entry.name)
    available = [name for name in MULTILINGUAL_SANS_SERIF if name in regular]
    if MULTILINGUAL_SANS_SERIF[0] not in available and "DejaVu Sans" in available:
        available.remove("DejaVu Sans")
        available.insert(0, "DejaVu Sans")
    return available or ["DejaVu Sans"]


def _contains_arabic(text: str) -> bool:
    """Return whether ``text`` contains a code point from an Arabic Unicode block."""
    return any(lo <= ord(char) <= hi for char in text for lo, hi in _ARABIC_RANGES)


def display_text(text: str) -> str:
    """Prepare logical Unicode ``text`` for Matplotlib, shaping Arabic only when present."""
    if not _contains_arabic(text):
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except ImportError as exc:
        raise RuntimeError(
            "Arabic figure text requires the 'viz' dependencies arabic-reshaper and python-bidi"
        ) from exc
    return get_display(arabic_reshaper.reshape(text))


def token_label(
    tid:       int,
    max_chars: int = 12,
    decode:    Optional[Callable[[Sequence[int]], str]] = None,
) -> str:
    """Return a short, printable, display-ready label for token id ``tid``."""
    if decode is None:
        return str(int(tid))
    try:
        logical = decode([int(tid)])
    except Exception:
        return str(int(tid))
    if logical == "" or logical.isspace():
        return {" ": "·", "\n": "\\n", "\t": "\\t"}.get(logical, "␣")
    logical = logical.replace("\n", "\\n").replace("\t", "\\t")
    if "�" in logical or not logical.isprintable():
        return str(int(tid))
    shortened = logical if len(logical) <= max_chars else logical[:max_chars - 1] + "…"
    return display_text(shortened)
