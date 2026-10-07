r"""Human-readable copy of a launcher's ``config`` dictionary (``config.md`` beside ``config.json``).

``config.json`` stores the full resolved :class:`~vfe3.config.VFE3Config`, alphabetized and without
comments, which is what every reader reloads. ``config.md`` is for people: it reproduces the
launcher's ``config = dict(...)`` block verbatim (order, section banners, alignment, comments),
together with the top-level ``config[...] = ...`` statements that follow it, and then lists every
value the source text does not show as a literal: the run seed, keys set by an expression or by a
later statement, and keys whose resolved value differs from the literal (the launcher file changed
after launch, or :class:`~vfe3.config.VFE3Config` coerced the value). Best-effort: a launcher without
such a block yields no file, and nothing here is read back.
"""

import ast
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple

_MISSING = object()


def config_block(source: str, name: str = "config") -> Optional[Tuple[str, List[Tuple[str, Optional[ast.expr]]]]]:
    r"""The verbatim ``name = dict(...)`` block and the ``name[...]`` statements after it.

    Returns ``(text, keys)``, ``keys`` holding each keyword of the ``dict`` call in source order with
    its value node (``None`` for keys added only by a later statement), or None without such a block.
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    start = end = None
    keys: List[Tuple[str, Optional[ast.expr]]] = []
    for node in tree.body:
        if start is None:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name
                    and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "dict"):
                start, end = node.lineno, node.end_lineno
                keys = [(kw.arg, kw.value) for kw in node.value.keywords if kw.arg is not None]
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else [])
        for target in targets:
            if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                    and target.value.id == name):
                end = node.end_lineno
                key = target.slice.value if isinstance(target.slice, ast.Constant) else None
                if isinstance(key, str) and key not in dict(keys):
                    keys.append((key, None))
    if start is None:
        return None
    return "\n".join(lines[start - 1:end]), keys


def _literal(node: Optional[ast.expr]) -> Any:
    if node is None:
        return _MISSING
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return _MISSING


def render_config_markdown(
    source:   str,
    resolved: Mapping[str, Any],

    *,
    launcher: str = "train_vfe3.py",
    header:   Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    r"""``config.md`` text: run header, the verbatim block, then values the literals do not show.

    ``resolved`` maps config keys to the values the run used (``seed`` included); ``header`` adds
    launcher-level settings (dataset, data seed) as a leading list.
    """
    found = config_block(source)
    if found is None:
        return None
    text, keys = found
    out = [f"# Run configuration ({launcher})", ""]
    for label, value in (header or {}).items():
        out.append(f"- {label}: `{value!r}`")
    if header:
        out.append("")
    out += ["```python", text, "```", ""]
    rows = []
    if "seed" in resolved and "seed" not in dict(keys):
        rows.append(("seed", resolved["seed"], "from SEEDS"))
    for key, node in keys:
        if key not in resolved:
            continue
        value, literal = resolved[key], _literal(node)
        if node is None:
            rows.append((key, value, "set after the block"))
        elif literal is _MISSING:
            rows.append((key, value, f"from `{ast.unparse(node)}`"))
        elif literal != value:
            rows.append((key, value, f"source literal `{literal!r}`"))
    if rows:
        out += ["## Values not shown as literals above", "", "| key | value | origin |", "|---|---|---|"]
        out += [f"| `{k}` | `{v!r}` | {why} |" for k, v, why in rows]
        out.append("")
    return "\n".join(out)


def write_config_markdown(
    run_dir:  'str | Path',
    source_path: 'str | Path',
    resolved: Mapping[str, Any],

    *,
    header:   Optional[Mapping[str, Any]] = None,
) -> Optional[Path]:
    r"""Write ``<run_dir>/config.md`` from the launcher at ``source_path``; None when it has no block."""
    source_path = Path(source_path)
    text = render_config_markdown(source_path.read_text(encoding="utf-8"), resolved,
                                  launcher=source_path.name, header=header)
    if text is None:
        return None
    path = Path(run_dir) / "config.md"
    path.write_text(text, encoding="utf-8")
    return path
