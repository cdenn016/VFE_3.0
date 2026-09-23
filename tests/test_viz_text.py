import pytest

from vfe3.viz.text import (
    MULTILINGUAL_SANS_SERIF,
    display_text,
    supports_english_linguistic_taxonomies,
    token_label,
)


def test_japanese_display_text_is_unchanged():
    assert display_text("日本語の表現") == "日本語の表現"


def test_arabic_display_text_is_shaped_for_matplotlib():
    # Arabic shaping needs the optional 'viz' extras (arabic-reshaper + python-bidi); without them
    # display_text raises by design, so there is nothing to assert here rather than a defect to fix.
    pytest.importorskip("arabic_reshaper")
    logical = "العربية"
    visual = display_text(logical)

    assert visual != logical
    assert any("\ufb50" <= ch <= "\ufeff" for ch in visual)


def test_token_label_rejects_invalid_decoded_fragment():
    assert token_label(17, decode=lambda ids: "�") == "17"


def test_token_label_keeps_visible_whitespace_markers():
    assert token_label(3, decode=lambda ids: "\n") == "\\n"
    assert token_label(4, decode=lambda ids: " ") == "·"


def test_font_fallbacks_and_dataset_policy_are_explicit():
    assert "Yu Gothic" in MULTILINGUAL_SANS_SERIF
    assert "Noto Sans Arabic" in MULTILINGUAL_SANS_SERIF
    assert supports_english_linguistic_taxonomies("wikitext-103")
    assert supports_english_linguistic_taxonomies("wiki-en")
    assert not supports_english_linguistic_taxonomies("wiki-ja")
    assert not supports_english_linguistic_taxonomies("wiki-ar")
