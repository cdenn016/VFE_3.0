"""Create a deterministic candidate run for rigorous-theory-search/v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from datetime import date as calendar_date
from pathlib import Path


TEMPLATE_DIRECTORY = Path(__file__).resolve().parents[1] / "assets" / "templates"
SLUG_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
JSON_ARTIFACTS = (
    "problem-contract.json",
    "approach-registry.json",
    "claim-ledger.json",
    "dependency-dag.json",
    "adversarial-report.json",
    "release.json",
)
MARKDOWN_ARTIFACTS = (
    "counterexample-register.md",
    "construction-or-strongest-theorem.md",
    "final-report.md",
)


def _validate_slug(slug: str) -> None:
    if not isinstance(slug, str) or not SLUG_PATTERN.fullmatch(slug):
        raise ValueError("slug must contain lowercase letters, digits, and single hyphens")


def _validate_date(value: str) -> None:
    try:
        parsed = calendar_date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError("date must be a real YYYY-MM-DD date") from error
    if parsed.isoformat() != value:
        raise ValueError("date must be a real YYYY-MM-DD date")


def _run_directory(output_root: Path, slug: str, value: str) -> Path:
    root = output_root.expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise ValueError("output_root must be a directory")
    run_dir = (root / f"{value}-{slug}").resolve()
    if run_dir.parent != root:
        raise ValueError("run directory escapes output_root")
    return run_dir


def _target_binding() -> tuple[str, str]:
    template = (TEMPLATE_DIRECTORY / "problem-contract.json").read_text(encoding="utf-8")
    target = json.loads(template)["target"]
    canonical = json.dumps(
        target,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    return f"contract-sha256-{digest}", digest


def _render(name: str, contract_id: str, target_digest: str) -> str:
    template = (TEMPLATE_DIRECTORY / name).read_text(encoding="utf-8")
    rendered = template.replace("{{CONTRACT_ID}}", contract_id).replace(
        "{{TARGET_DIGEST}}", target_digest,
    )
    if name.endswith(".json"):
        json.loads(rendered)
    return rendered


def _render_artifacts(contract_id: str, target_digest: str) -> dict[str, str]:
    return {
        name: _render(name, contract_id, target_digest)
        for name in (*JSON_ARTIFACTS, *MARKDOWN_ARTIFACTS)
    }


def _write_artifacts(run_dir: Path, artifacts: dict[str, str]) -> None:
    for name, contents in artifacts.items():
        (run_dir / name).write_text(contents, encoding="utf-8")


def _safe_cleanup(root: Path, staging: Path, prefix: str) -> None:
    if staging.parent == root and staging.name.startswith(prefix) and staging.is_dir() and not staging.is_symlink():
        shutil.rmtree(staging)


def scaffold_run(output_root: Path, slug: str, date: str) -> Path:
    """Create a non-overwriting candidate run beneath an explicitly supplied root."""
    _validate_slug(slug)
    _validate_date(date)
    run_dir = _run_directory(Path(output_root), slug, date)
    contract_id, target_digest = _target_binding()
    artifacts = _render_artifacts(contract_id, target_digest)
    if run_dir.exists():
        raise FileExistsError(f"refuses to overwrite existing run: {run_dir}")
    root = run_dir.parent
    root.mkdir(parents=True, exist_ok=True)
    prefix = f".{run_dir.name}.staging-"
    staging = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
    try:
        _write_artifacts(staging, artifacts)
        if run_dir.exists():
            raise FileExistsError(f"refuses to overwrite existing run: {run_dir}")
        staging.rename(run_dir)
    except BaseException:
        _safe_cleanup(root, staging, prefix)
        raise
    return run_dir


def main(argv: list[str] | None = None) -> int:
    """Create one candidate run; successful scaffolding establishes no theorem."""
    parser = argparse.ArgumentParser(description="Create a candidate run; this does not establish mathematical truth.")
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--date", required=True)
    arguments = parser.parse_args(argv)
    try:
        run_dir = scaffold_run(arguments.output_root, arguments.slug, arguments.date)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
