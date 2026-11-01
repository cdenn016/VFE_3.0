r"""Documentation checks of the October 31, 2026 audit (remediation package ``docs-readme``).

``README.md`` names module paths, dotted ``vfe3`` references, upper-case symbols, identifiers (configuration fields
and values, metric columns, sweep names) and links. The checks below resolve each of them against this checkout, so
a rename or a removal in the code fails here instead of leaving the reference document stale
(gap-readme-fidelity-4 and -5: a deleted name registry and a retired sweep were still described), and no control
byte stands in for a LaTeX escape (attention-scoring-6). ``docs/`` is gitignored and holds local records (the user's
decision of November 1), so the README links nothing there (gap-readme-fidelity-7). The remaining tests pin the
click-to-run manifest of ``paper_artifacts.py``
(gap-readme-fidelity-7), the README's statement of the Triton-interpreter requirement (gap-readme-fidelity-8) and
the removal of the AGENTS.md launcher-completeness rule (config-launcher-7). No model is built.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import json
import os
import re
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Mapping, Optional
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

# Names the README mentions on purpose although no code defines them, with the reason.
_INTENTIONALLY_ABSENT = {
    "use_prior_bank": "the README states that no such switch exists",
}

_FENCE = re.compile(r"(?ms)^```.*?^```[^\n]*$")
# Inline code spans; a span written as $`...`$ is GitLab-style math, not code.
_SPAN = re.compile(r"(?<!\$)`([^`\n]+)`(?!\$)")
_LINK = re.compile(r"\]\(([^)\s]+)\)")
_IMG = re.compile(r'<img[^>]*\bsrc="([^"]+)"')
_SNAKE = r"_?[a-z][a-z0-9]*(?:_[a-z0-9]+)+"
_UPPER = r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+"
_IDENT_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _readme_text() -> str:
    return README.read_text(encoding="utf-8")


def _readme_prose() -> str:
    r"""README text without fenced code blocks (their backticks are not inline spans)."""
    return _FENCE.sub("", _readme_text())


def _spans() -> List[str]:
    return [span.strip() for span in _SPAN.findall(_readme_prose())]


def _python_sources() -> Iterator[Path]:
    r"""The checkout's Python sources: root scripts, the ``vfe3`` package and the tests, except this file, whose
    own strings must not vouch for a README name."""
    this = Path(__file__).resolve()
    yield from sorted(ROOT.glob("*.py"))
    for folder in ("vfe3", "tests"):
        yield from sorted(path for path in (ROOT / folder).rglob("*.py")
                          if "__pycache__" not in path.parts and path.resolve() != this)


def _docstring_nodes(tree: ast.AST) -> FrozenSet[int]:
    r"""``id`` of every docstring constant, so that prose inside docstrings is not counted as code."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                found.add(id(first.value))
    return frozenset(found)


@functools.lru_cache(maxsize=None)
def _parsed(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@functools.lru_cache(maxsize=None)
def _code_vocabulary() -> FrozenSet[str]:
    r"""Every identifier the sources use as code: names, attributes, definitions, arguments, keywords, imports and
    the identifier-like words of string literals other than docstrings (configuration values, metric columns and
    sweep names are strings). Comments and docstrings do not count."""
    words = set()
    for path in _python_sources():
        tree = _parsed(path)
        docstrings = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                words.add(node.id)
            elif isinstance(node, ast.Attribute):
                words.add(node.attr)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                words.add(node.name)
            elif isinstance(node, ast.arg):
                words.add(node.arg)
            elif isinstance(node, ast.keyword) and node.arg is not None:
                words.add(node.arg)
            elif isinstance(node, ast.alias):
                words.update(node.name.split("."))
                if node.asname:
                    words.add(node.asname)
            elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in docstrings):
                words.update(_IDENT_WORD.findall(node.value))
    return frozenset(words)


def _module_level_names(tree: ast.Module) -> FrozenSet[str]:
    r"""Names a module binds at its top level (assignments, definitions and imports)."""
    names = set()
    for node in tree.body:
        targets: List[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        for target in targets:
            for leaf in ast.walk(target):
                if isinstance(leaf, ast.Name):
                    names.add(leaf.id)
    return frozenset(names)


@functools.lru_cache(maxsize=None)
def _module_level_symbols() -> FrozenSet[str]:
    return frozenset().union(*(_module_level_names(_parsed(path)) for path in _python_sources()))


def _identifier_mentions(pattern: str) -> List[str]:
    r"""Identifiers matching ``pattern`` that a README span names: a whole span (quotes allowed), the left side of a
    ``name=value`` span, and a quoted identifier on its right side."""
    whole = re.compile(rf'^"?({pattern})"?$')
    left = re.compile(rf"^({pattern})\s*=")
    right = re.compile(rf'=\s*"({pattern})"')
    found = []
    for span in _spans():
        for regex in (whole, left):
            match = regex.match(span)
            if match:
                found.append(match.group(1))
        found.extend(right.findall(span))
    return sorted(set(found))


def _github_slug(heading: str) -> str:
    r"""The anchor GitHub gives a heading: lower case, punctuation other than ``-`` and ``_`` dropped, spaces to
    hyphens."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def _section(heading: str) -> str:
    r"""The README text under ``heading`` up to the next heading of the same or a higher level."""
    lines = _readme_text().splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^(#+)\s+(.*)$", line)
        if match and match.group(2).strip() == heading:
            level = len(match.group(1))
            body = []
            for later in lines[index + 1:]:
                other = re.match(r"^(#+)\s", later)
                if other and len(other.group(1)) <= level:
                    break
                body.append(later)
            return "\n".join(body)
    raise AssertionError(f"README has no heading {heading!r}")


# ---------------------------------------------------------------------------------------------------------------
# README references resolve against the checkout
# ---------------------------------------------------------------------------------------------------------------

def test_readme_links_resolve_and_avoid_the_local_docs_directory() -> None:
    r"""Every relative link and image of the README exists in the checkout, and none points into ``docs/``, which is
    gitignored and holds local records, so such a link is dead on every other machine (gap-readme-fidelity-7)."""
    prose = _readme_prose()
    targets = _LINK.findall(prose) + _IMG.findall(prose)
    assert targets, "the README parser found no links"
    problems = []
    for target in targets:
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path = target.split("#", 1)[0]
        if path.startswith("docs/"):
            problems.append(f"{target}: docs/ is local and untracked")
        elif not (ROOT / path).exists():
            problems.append(f"{target}: missing")
    assert not problems, problems


def test_readme_has_no_control_characters() -> None:
    r"""No C0 control byte besides tab, newline and carriage return: a ``\b`` written as an escape left a backspace
    in place of ``\beta`` in the routing-reference row (attention-scoring-6)."""
    text = _readme_text()
    found = sorted({f"U+{ord(ch):04X}" for ch in text if ord(ch) < 0x20 and ch not in "\t\n\r"})
    assert found == []


def test_readme_internal_anchors_name_headings() -> None:
    prose = _readme_prose()
    slugs = {_github_slug(match.group(2)) for match in re.finditer(r"(?m)^(#+)\s+(.*)$", prose)}
    anchors = [target[1:] for target in _LINK.findall(prose) if target.startswith("#")]
    assert anchors, "the README parser found no internal links"
    assert sorted(set(anchors) - slugs) == []


def test_readme_repository_paths_exist() -> None:
    r"""A span naming a file of the package, the tests or the assets exists; a bare ``name.py`` names some file of
    the checkout (link texts give module basenames). Run directories, caches, ``docs/`` records and placeholders
    (``<...>``) are not checked."""
    basenames = {path.name for path in _python_sources()}
    problems = []
    for span in _spans():
        if "<" in span or " " in span:
            continue
        path = re.sub(r"(::\w+|:\d+(-\d+)?)$", "", span)
        if path.startswith(("vfe3/", "tests/", "assets/")):
            if not (ROOT / path).exists():
                problems.append(span)
        elif re.fullmatch(r"[A-Za-z0-9_]+\.py", path) and path not in basenames:
            problems.append(span)
    assert not problems, problems


def test_readme_dotted_vfe3_references_resolve() -> None:
    r"""``vfe3.a.b.name`` names a module of the package and, after it, a name that module binds at top level."""
    references = [span for span in _spans() if re.match(r"^vfe3(\.\w+)+", span)]
    assert references, "the README parser found no dotted vfe3 references"
    problems = []
    for span in references:
        parts = re.match(r"^vfe3(\.\w+)+", span).group(0).split(".")
        module_path: Optional[Path] = None
        depth = 0
        for end in range(len(parts), 0, -1):
            stem = ROOT.joinpath(*parts[:end])
            for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
                if candidate.is_file():
                    module_path, depth = candidate, end
                    break
            if module_path is not None:
                break
        if module_path is None:
            problems.append(f"{span}: no module")
            continue
        rest = parts[depth:]
        if rest and rest[0] not in _module_level_names(_parsed(module_path)):
            problems.append(f"{span}: {module_path.relative_to(ROOT)} binds no {rest[0]!r}")
    assert not problems, problems


def test_readme_upper_case_symbols_are_defined() -> None:
    r"""An upper-case symbol the README names (``SWEEP_ORDER``, ``CUDA_TESTS``, ...) is bound at module level
    somewhere in the sources; the deleted ``DIAGNOSTIC_METRIC_OUTPUTS`` registry was still described
    (gap-readme-fidelity-4)."""
    names = _identifier_mentions(_UPPER)
    assert names, "the README parser found no upper-case symbols"
    assert sorted(set(names) - _module_level_symbols()) == []


def test_readme_identifiers_occur_in_code() -> None:
    r"""A lower-case identifier the README names (a field, a value, a metric column, a sweep, a function) occurs in
    the code, not only in comments or docstrings. A held-out ``val_`` column may be built from its training column.
    The retired ``e_q_mu_sigma_lr_grid`` sweep survived only in a test comment (gap-readme-fidelity-5)."""
    vocabulary = _code_vocabulary()
    names = _identifier_mentions(_SNAKE)
    assert len(names) > 50, "the README parser found too few identifiers"
    missing = []
    for name in names:
        if name in vocabulary or name in _INTENTIONALLY_ABSENT:
            continue
        if name.startswith("val_") and name[len("val_"):] in vocabulary:
            continue
        missing.append(name)
    assert missing == []


def test_focused_validation_names_the_cpu_runner_and_the_triton_requirement() -> None:
    r"""The documented focused command skips the CPU checks of the fused Triton kernels under an interpreter without
    triton; the section must name the runner that runs them and the requirement (gap-readme-fidelity-8)."""
    section = _section("Focused validation")
    assert "run_cpu_tests.py" in section
    assert re.search(r"interpreter with triton", section)


def test_agents_md_no_longer_requires_every_field_in_the_launcher() -> None:
    r"""The user retired the launcher-completeness rule on November 1 (config-launcher-7)."""
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "must also be added to the `config` dictionary" not in text
    assert "Record each day's changes" in text


# ---------------------------------------------------------------------------------------------------------------
# paper_artifacts.py: the manifest path is a CONFIG key, not a command-line argument
# ---------------------------------------------------------------------------------------------------------------

def _load_paper_artifacts() -> ModuleType:
    r"""Import ``paper_artifacts.py`` under a private name; its thread variables are restored afterwards."""
    spec = importlib.util.spec_from_file_location("paper_artifacts_under_test", ROOT / "paper_artifacts.py")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(os.environ):
        spec.loader.exec_module(module)
    return module


def test_paper_artifacts_reads_its_manifest_from_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""``CONFIG["manifest"]`` names the JSON manifest, and its keys reach the assembler without any command-line
    argument (gap-readme-fidelity-7: the script was the only root entrypoint that read ``sys.argv``)."""
    import vfe3.paper

    module = _load_paper_artifacts()
    manifest = {"arms": {"Evaluated": {"group": "Retraining", "runs": []}}, "selection": "final"}
    path = tmp_path / "m2_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setitem(module.CONFIG, "manifest", str(path))
    monkeypatch.setitem(module.CONFIG, "out_dir", str(tmp_path / "artifacts"))
    seen: Dict[str, Dict[str, Any]] = {}

    def fake_assemble(                  # the real assembler reads run directories
        config:  Mapping[str, Any],

        *,
        out_dir: Any,
        log:     Callable[[str], None],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        seen["config"] = dict(config)
        return {"tables": {}, "figures": {}, "out_dir": str(out_dir), "problems": []}

    monkeypatch.setattr(vfe3.paper, "assemble", fake_assemble)
    argv = module._config_argv([str(ROOT / "paper_artifacts.py")])
    assert argv[1:] == [str(path)]
    assert module.main(argv) == 0
    assert seen["config"]["selection"] == "final"
    assert seen["config"]["arms"] == manifest["arms"]

    monkeypatch.setitem(module.CONFIG, "manifest", None)
    assert module._config_argv(["paper_artifacts.py"]) == [module.__file__]


def test_paper_artifacts_refuses_a_command_line_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Run as a script with an argument, the earlier way of passing a manifest, the driver stops with a message
    naming ``CONFIG["manifest"]`` before it assembles anything, so the argument cannot be silently ignored."""
    monkeypatch.setattr(sys, "argv", ["paper_artifacts.py", "m2_manifest.json"])
    with mock.patch.dict(os.environ), pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(ROOT / "paper_artifacts.py"), run_name="__main__")
    assert "CONFIG['manifest']" in str(stopped.value.code)
