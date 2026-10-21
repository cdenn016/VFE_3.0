r"""Python 3.10 compatibility of ``vfe3.contracts`` (audit 2026-10-20 item 74).

``pyproject.toml`` declares ``requires-python = ">=3.10"``; PEP 655 ``NotRequired`` and PEP 678 ``add_note`` are
3.11 features, so the module imports the first from ``typing_extensions`` on 3.10 and routes notes through
:func:`vfe3.contracts.add_exception_note`.
"""
import ast
import pathlib

import pytest

from vfe3.contracts import _with_failure_detail, add_exception_note


_ROOT = pathlib.Path(__file__).resolve().parents[1]


class _NoAddNote(Exception):
    r"""An exception without PEP 678 ``add_note``, as on Python 3.10."""

    def __getattribute__(self, name: str):  # type: ignore[no-untyped-def]
        if name == "add_note":
            raise AttributeError(name)
        return super().__getattribute__(name)


def test_add_exception_note_uses_add_note_or_dunder_notes() -> None:
    exc = ValueError("x")
    add_exception_note(exc, "first")
    add_exception_note(exc, "second")
    assert exc.__notes__ == ["first", "second"]
    legacy = _NoAddNote("y")
    add_exception_note(legacy, "only")
    assert legacy.__notes__ == ["only"]


def test_checked_verdict_attaches_the_detail_note() -> None:
    def verdict(host: list) -> None:
        raise RuntimeError("bad")

    with pytest.raises(RuntimeError) as info:
        _with_failure_detail(verdict, lambda: "detail", 3)([True])
    assert info.value.__notes__ == ["registered at layer 3: detail"]


@pytest.mark.parametrize("relative", ["vfe3"])
def test_no_unguarded_python311_features_in_the_package(relative: str) -> None:
    offenders = []
    for path in sorted((_ROOT / relative).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        guarded = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for inner in ast.walk(node):
                    guarded.add(id(inner))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "typing" and id(node) not in guarded:
                if any(alias.name in {"NotRequired", "Required", "Self", "LiteralString", "Never"}
                       for alias in node.names):
                    offenders.append(f"{path.name}:{node.lineno} typing 3.11 import")
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_note"):
                offenders.append(f"{path.name}:{node.lineno} unguarded add_note")
    assert offenders == []
