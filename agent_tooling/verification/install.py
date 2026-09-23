"""Install the verification skill, neutral agents, policies, and Stop hooks."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import stat
import sys
from pathlib import Path
from typing import Any

from agent_tooling.verification.skill.scripts.verification_gate import (
    GATE_COMMAND_TOKEN,
    GATE_SHELL_TOKEN,
    render_skill_markdown,
)


AGENT_ROLES = (
    "verifier-orchestrator",
    "verifier-code",
    "verifier-math",
    "verifier-evidence",
    "verifier-skeptic",
    "verifier-adjudicator",
)
BLOCK_MARKERS = {
    "global-policy.md": "VERIFICATION GLOBAL POLICY",
    "project-policy.md": "VERIFICATION PROJECT POLICY",
    "deep-audit-integration.md": "VERIFICATION DEEP AUDIT INTEGRATION",
}
SKILL_MANIFEST = (
    "SKILL.md",
    "scripts/verification_gate.py",
    "schemas/claim-ledger.schema.json",
    "evals/evals.json",
    "references/contract.md",
    "references/criteria-code.md",
    "references/criteria-math.md",
    "references/criteria-evidence.md",
    "references/criteria-experiment.md",
    "references/criteria-general.md",
)
ADJUDICATOR_OUTPUT = {
    "required": [
        "claim_id",
        "ledger_state",
        "result",
        "rationale",
        "view_ids",
        "evidence_ids",
        "result_location",
        "open_obligations",
        "validator_errors",
    ],
    "ledger_state": ["EVIDENCE_VERIFIED", "REFUTED", "INCONCLUSIVE"],
}
GATE_STOP_STATUS_MESSAGE = "Verification ledger gate"
_SKILL_TOKEN_COUNTS = {GATE_COMMAND_TOKEN: 2, GATE_SHELL_TOKEN: 2}
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _required_string(spec: dict[str, object], name: str) -> str:
    value = spec.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"agent spec requires nonempty {name}")
    return value


def _optional_string_list(spec: dict[str, object], name: str) -> list[str] | None:
    value = spec.get(name)
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"agent spec {name} must be a list of strings")
    return value


def _literal(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def effective_instructions(spec: dict[str, object]) -> str:
    """Return the single neutral instruction body rendered for both clients."""

    instructions = _required_string(spec, "instructions")
    output = spec.get("output")
    if output is None:
        return instructions
    if not isinstance(output, dict):
        raise ValueError("agent spec output must be an object")
    return f"{instructions}\n\nRequired output schema:\n{json.dumps(output, indent=2, ensure_ascii=False)}"


def render_claude_agent(spec: dict[str, object]) -> str:
    """Render one neutral verifier spec as Claude agent Markdown."""

    name = _required_string(spec, "name")
    description = _required_string(spec, "description")
    instructions = effective_instructions(spec)
    tools = _optional_string_list(spec, "tools")
    frontmatter = f"name: {_literal(name)}\ndescription: {_literal(description)}\n"
    if tools is not None:
        frontmatter += f"tools: {_literal(tools)}\n"
    return f"---\n{frontmatter}---\n\n{instructions}\n"


def render_codex_agent(spec: dict[str, object]) -> str:
    """Render one neutral verifier spec as literal-safe Codex TOML."""

    name = _required_string(spec, "name")
    description = _required_string(spec, "description")
    instructions = effective_instructions(spec)
    tools = _optional_string_list(spec, "tools")
    rendered = f"name = {_literal(name)}\ndescription = {_literal(description)}\ninstructions = {_literal(instructions)}\n"
    if tools is not None:
        rendered += f"tools = {_literal(tools)}\n"
    return rendered


def upsert_marked_block(text: str, marker: str, block: str) -> str:
    """Replace or append exactly one HTML-comment-delimited policy block."""

    begin = f"<!-- BEGIN {marker} -->"
    end = f"<!-- END {marker} -->"
    if begin in text or end in text:
        if text.count(begin) != 1 or text.count(end) != 1 or text.index(begin) > text.index(end):
            raise ValueError(f"invalid marked block for {marker}")
        before, remainder = text.split(begin, 1)
        _, after = remainder.split(end, 1)
        return f"{before}{begin}\n{block.rstrip()}\n{end}{after}"
    separator = "" if not text or text.endswith("\n") else "\n"
    return f"{text}{separator}{begin}\n{block.rstrip()}\n{end}\n"


def _remove_marked_block(text: str, marker: str) -> str:
    begin = f"<!-- BEGIN {marker} -->"
    end = f"<!-- END {marker} -->"
    if begin not in text and end not in text:
        return text
    if text.count(begin) != 1 or text.count(end) != 1 or text.index(begin) > text.index(end):
        raise ValueError(f"invalid marked block for {marker}")
    start = text.index(begin)
    finish = text.index(end, start) + len(end)
    if text[finish:finish + 2] == "\r\n":
        finish += 2
    elif text[finish:finish + 1] == "\n":
        finish += 1
    return text[:start] + text[finish:]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required in {path}")
    return value


def _read_block(path: Path, marker: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    begin = f"<!-- BEGIN {marker} -->"
    end = f"<!-- END {marker} -->"
    if text.count(begin) != 1 or text.count(end) != 1 or text.index(begin) > text.index(end):
        raise ValueError(f"source block {path} must contain one marked {marker} block")
    return text.split(begin, 1)[1].split(end, 1)[0].strip()


def _read_specs(source_root: Path) -> dict[str, dict[str, object]]:
    specs: dict[str, dict[str, object]] = {}
    for role in AGENT_ROLES:
        path = source_root / "agents" / f"{role}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"agent spec must be an object: {path}")
        if _required_string(value, "name") != role:
            raise ValueError(f"agent spec name must match filename: {path}")
        _required_string(value, "description")
        _required_string(value, "instructions")
        _optional_string_list(value, "tools")
        if role == "verifier-adjudicator" and value.get("output") != ADJUDICATOR_OUTPUT:
            raise ValueError("verifier-adjudicator requires the exact structured output contract")
        specs[role] = value
    return specs


def _preflight_skill(source: Path) -> None:
    for relative_path in SKILL_MANIFEST:
        required = source / relative_path
        if not required.is_file():
            raise FileNotFoundError(required)


def _read_preflight_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required in {path}")
    return value


def _preflight_skill_content(source: Path, shell: str) -> None:
    schema = _read_preflight_json(source / "schemas" / "claim-ledger.schema.json")
    if schema.get("type") != "object" or not isinstance(schema.get("required"), list) or not isinstance(schema.get("properties"), dict):
        raise ValueError("claim-ledger schema must define an object root, required fields, and properties")
    evals = _read_preflight_json(source / "evals" / "evals.json")
    if evals.get("skill_name") != "verification" or not isinstance(evals.get("evals"), list):
        raise ValueError("skill evals must define verification skill_name and an evals list")
    template = (source / "SKILL.md").read_text(encoding="utf-8")
    for token, expected_count in _SKILL_TOKEN_COUNTS.items():
        if template.count(token) != expected_count:
            raise ValueError(f"skill template must contain {token} exactly {expected_count} times")
    rendered = render_skill_markdown(source, shell=shell)
    if GATE_COMMAND_TOKEN in rendered or GATE_SHELL_TOKEN in rendered:
        raise ValueError("rendered skill retains a verification render token")
    shell_label = "powershell" if shell == "powershell" else "bash"
    if rendered.count(f"```{shell_label}") != _SKILL_TOKEN_COUNTS[GATE_SHELL_TOKEN]:
        raise ValueError("rendered skill does not contain the expected shell fences")


def _verification_skill_root(home: Path, destination: Path) -> Path:
    expected = home / "skills" / "verification"
    if destination.absolute() != expected.absolute():
        raise ValueError("verification skill destination must be the explicit home skill root")
    home_resolved = home.resolve()
    destination_resolved = destination.resolve()
    try:
        destination_resolved.relative_to(home_resolved)
    except ValueError as exc:
        raise ValueError("verification skill destination resolves outside its explicit home") from exc
    return destination_resolved


def _is_symlink_or_reparse(path: Path) -> bool:
    try:
        status = path.lstat()
    except FileNotFoundError:
        return False
    return path.is_symlink() or bool(getattr(status, "st_file_attributes", 0) & _REPARSE_POINT)


def _preflight_managed_destination(root: Path, destination: Path) -> None:
    root_absolute = root.absolute()
    destination_absolute = destination.absolute()
    try:
        relative = destination_absolute.relative_to(root_absolute)
    except ValueError as exc:
        raise ValueError(f"managed destination is outside its explicit root: {destination}") from exc
    current = root_absolute
    if _is_symlink_or_reparse(current):
        raise ValueError(f"managed destination contains a symlink or reparse point: {current}")
    for part in relative.parts:
        current /= part
        if _is_symlink_or_reparse(current):
            raise ValueError(f"managed destination contains a symlink or reparse point: {current}")
    root_resolved = root_absolute.resolve()
    destination_resolved = destination_absolute.resolve()
    try:
        destination_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"managed destination resolves outside its explicit root: {destination}") from exc


def _preflight_skill_destination(source: Path, destination: Path, home: Path) -> None:
    _preflight_managed_destination(home, destination)
    for source_path in sorted(source.rglob("*")):
        relative = source_path.relative_to(source)
        if "__pycache__" in relative.parts or source_path.suffix == ".pyc":
            continue
        _preflight_managed_destination(home, destination / relative)
    if destination.is_dir():
        for existing_path in destination.rglob("*"):
            _preflight_managed_destination(home, existing_path)


def _clean_skill_caches(home: Path, destination: Path) -> None:
    root = _verification_skill_root(home, destination)
    cache_directories = sorted(
        (path for path in root.rglob("__pycache__") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for cache_directory in cache_directories:
        resolved = cache_directory.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("cache directory resolves outside the verification skill root") from exc
        shutil.rmtree(resolved)
    for bytecode in root.rglob("*.pyc"):
        resolved = bytecode.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("bytecode file resolves outside the verification skill root") from exc
        if resolved.is_file():
            resolved.unlink()


def _copy_skill(source: Path, destination: Path, home: Path, shell: str) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    _verification_skill_root(home, destination)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    skill_markdown = render_skill_markdown(destination, shell=shell)
    (destination / "SKILL.md").write_text(skill_markdown, encoding="utf-8")
    _clean_skill_caches(home, destination)


def _posix_hook_path(path: Path | str) -> str:
    text = str(path)
    if text.startswith("/"):
        return text
    text = str(Path(path).resolve()).replace("\\", "/")
    if len(text) >= 3 and text[1:3] == ":/":
        return f"/{text[0].lower()}{text[2:]}"
    return text


def _gate_command(gate_path: Path, shell: str) -> str:
    if shell == "posix":
        return f"{shlex.quote(_posix_hook_path(sys.executable))} {shlex.quote(_posix_hook_path(gate_path))} hook"
    return f'& "{Path(sys.executable).resolve()}" "{gate_path.resolve()}" hook'


def _gate_stop_handler(command: str, shell: str) -> dict[str, object]:
    return {
        "hooks": [
            {
                "type": "command",
                "statusMessage": GATE_STOP_STATUS_MESSAGE,
                "shell": shell,
                "command": command,
            }
        ]
    }


def _is_python_executable(value: str) -> bool:
    executable = value.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return re.fullmatch(r"python(?:\d+(?:\.\d+)*)?(?:\.exe)?", executable) is not None


def _split_hook_command(command: str, shell: str) -> list[str] | None:
    try:
        if shell == "posix":
            return shlex.split(command, posix=True)
        match = re.fullmatch(r'\s*(?:&\s*)?"([^"]+)"\s+"([^"]+)"\s+(\S+)\s*', command)
        return list(match.groups()) if match is not None else None
    except ValueError:
        return None


def _is_verification_handler(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    shell = value.get("shell")
    if (
        value.get("type") != "command"
        or value.get("statusMessage") != GATE_STOP_STATUS_MESSAGE
        or shell not in {"powershell", "posix"}
    ):
        return False
    command = value.get("command")
    if not isinstance(command, str):
        return False
    parts = _split_hook_command(command, shell)
    if parts is None or len(parts) != 3 or not _is_python_executable(parts[0]) or parts[2] != "hook":
        return False
    gate_path = parts[1].replace("\\", "/")
    return gate_path.endswith("skills/verification/scripts/verification_gate.py")


def _remove_verification_handlers(stop: list[Any]) -> list[Any]:
    retained: list[Any] = []
    for entry in stop:
        if not isinstance(entry, dict):
            retained.append(entry)
            continue
        handlers = entry.get("hooks")
        if isinstance(handlers, list):
            remaining = [handler for handler in handlers if not _is_verification_handler(handler)]
            if remaining:
                updated = dict(entry)
                updated["hooks"] = remaining
                retained.append(updated)
        elif not _is_verification_handler(entry):
            retained.append(entry)
    return retained


def _install_stop_handler(settings: dict[str, Any], gate_path: Path, shell: str) -> dict[str, Any]:
    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError("settings hooks must be an object")
    stop = hooks.setdefault("Stop", [])
    if not isinstance(stop, list):
        raise ValueError("settings hooks.Stop must be a list")
    command = _gate_command(gate_path, shell)
    stop[:] = _remove_verification_handlers(stop)
    stop.append(_gate_stop_handler(command, shell))
    return settings


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _read_required_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read required text file {path}: {exc}") from exc
    if not text.strip():
        raise ValueError(f"required text file is empty: {path}")
    return text


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def install(args: argparse.Namespace) -> None:
    """Install from explicit paths after validating every source and JSON destination."""

    source_root = Path(args.source_root)
    claude_home = Path(args.claude_home)
    codex_home = Path(args.codex_home)
    project_root = Path(args.project_root) if args.project_root is not None else None
    shell = str(args.shell)
    if shell not in {"powershell", "posix"}:
        raise ValueError("shell must be powershell or posix")
    skill_source = source_root / "skill"
    _preflight_skill(skill_source)
    _preflight_skill_content(skill_source, shell)
    specs = _read_specs(source_root)
    blocks = {filename: _read_block(source_root / "blocks" / filename, marker) for filename, marker in BLOCK_MARKERS.items()}

    claude_settings_path = claude_home / "settings.json"
    codex_hooks_path = codex_home / "hooks.json"
    claude_deep_audit_path = claude_home / "skills" / "deep-audit" / "SKILL.md"
    codex_deep_audit_path = codex_home / "skills" / "deep-audit" / "SKILL.md"
    claude_skill = claude_home / "skills" / "verification"
    codex_skill = codex_home / "skills" / "verification"
    _preflight_skill_destination(skill_source, claude_skill, claude_home)
    _preflight_skill_destination(skill_source, codex_skill, codex_home)
    _preflight_managed_destination(claude_home, claude_settings_path)
    _preflight_managed_destination(codex_home, codex_hooks_path)
    _preflight_managed_destination(claude_home, claude_home / "CLAUDE.md")
    _preflight_managed_destination(codex_home, codex_home / "AGENTS.md")
    _preflight_managed_destination(claude_home, claude_deep_audit_path)
    _preflight_managed_destination(codex_home, codex_deep_audit_path)
    for role in AGENT_ROLES:
        _preflight_managed_destination(claude_home, claude_home / "agents" / f"{role}.md")
        _preflight_managed_destination(codex_home, codex_home / "agents" / f"{role}.toml")
    if project_root is not None:
        _preflight_managed_destination(project_root, project_root / "CLAUDE.md")
        _preflight_managed_destination(project_root, project_root / "AGENTS.md")
    claude_deep_audit = _read_required_text(claude_deep_audit_path)
    codex_deep_audit = _read_required_text(codex_deep_audit_path)
    claude_settings = _read_json(claude_settings_path)
    codex_hooks = _read_json(codex_hooks_path)
    claude_global = _read_text(claude_home / "CLAUDE.md")
    codex_global = _read_text(codex_home / "AGENTS.md")
    project_claude = _read_text(project_root / "CLAUDE.md") if project_root is not None else ""
    project_codex = _read_text(project_root / "AGENTS.md") if project_root is not None else ""

    claude_settings = _install_stop_handler(claude_settings, claude_skill / "scripts" / "verification_gate.py", shell)
    codex_hooks = _install_stop_handler(codex_hooks, codex_skill / "scripts" / "verification_gate.py", shell)
    global_marker = BLOCK_MARKERS["global-policy.md"]
    deep_audit_marker = BLOCK_MARKERS["deep-audit-integration.md"]
    claude_global = _remove_marked_block(claude_global, deep_audit_marker)
    codex_global = _remove_marked_block(codex_global, deep_audit_marker)
    claude_global = upsert_marked_block(claude_global, global_marker, blocks["global-policy.md"])
    codex_global = upsert_marked_block(codex_global, global_marker, blocks["global-policy.md"])
    claude_deep_audit = upsert_marked_block(claude_deep_audit, deep_audit_marker, blocks["deep-audit-integration.md"])
    codex_deep_audit = upsert_marked_block(codex_deep_audit, deep_audit_marker, blocks["deep-audit-integration.md"])
    if project_root is not None:
        marker = BLOCK_MARKERS["project-policy.md"]
        project_claude = upsert_marked_block(project_claude, marker, blocks["project-policy.md"])
        project_codex = upsert_marked_block(project_codex, marker, blocks["project-policy.md"])

    _copy_skill(skill_source, claude_skill, claude_home, shell)
    _copy_skill(skill_source, codex_skill, codex_home, shell)
    for role, spec in specs.items():
        _write_text(claude_home / "agents" / f"{role}.md", render_claude_agent(spec))
        _write_text(codex_home / "agents" / f"{role}.toml", render_codex_agent(spec))
    _write_text(claude_home / "CLAUDE.md", claude_global)
    _write_text(codex_home / "AGENTS.md", codex_global)
    _write_text(claude_deep_audit_path, claude_deep_audit)
    _write_text(codex_deep_audit_path, codex_deep_audit)
    if project_root is not None:
        _write_text(project_root / "CLAUDE.md", project_claude)
        _write_text(project_root / "AGENTS.md", project_codex)
    _write_text(claude_settings_path, json.dumps(claude_settings, indent=2) + "\n")
    _write_text(codex_hooks_path, json.dumps(codex_hooks, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--claude-home", required=True, type=Path)
    parser.add_argument("--codex-home", required=True, type=Path)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--shell", choices=("powershell", "posix"), required=True)
    install(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
