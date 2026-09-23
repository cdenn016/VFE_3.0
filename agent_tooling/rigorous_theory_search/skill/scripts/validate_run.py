"""Structural validator for rigorous-theory-search/v1 run packages.

This validator checks declared structure and evidence metadata only; it cannot
establish the mathematical truth of a claimed theorem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "rigorous-theory-search/v1"
JSON_ARTIFACTS = (
    "problem-contract.json", "approach-registry.json", "claim-ledger.json",
    "dependency-dag.json", "adversarial-report.json", "release.json",
)
MARKDOWN_ARTIFACTS = (
    "counterexample-register.md", "construction-or-strongest-theorem.md",
    "final-report.md",
)
STATES = {"CANDIDATE", "LLM_SUPPORTED", "EVIDENCE_VERIFIED", "REFUTED", "INCONCLUSIVE"}
CLAIM_KINDS = {"MATHEMATICAL", "PHYSICAL_INTERPRETATION", "MODELING_POSTULATE", "OPERATIONAL_IDENTIFICATION", "NUMERICAL_OBSERVATION"}
EVIDENCE_KINDS = {"DERIVATION", "FORMAL_PROOF", "APPLICABLE_THEOREM", "PRIMARY_SOURCE", "COUNTEREXAMPLE", "NONEXISTENCE_PROOF", "NUMERICAL", "SYMBOLIC_CHECK", "FIGURE", "AGENT_ASSESSMENT"}
TERMINAL_STATUSES = {"COMPLETE_AFFIRMATIVE", "COMPLETE_NEGATIVE", "INCONCLUSIVE"}
MATH_EVIDENCE = {"DERIVATION", "FORMAL_PROOF", "APPLICABLE_THEOREM"}
NEGATIVE_CERTIFICATES = {"COUNTEREXAMPLE", "NONEXISTENCE_PROOF"}
TARGET_STRING_FIELDS = (
    "statement", "kind", "quantifiers", "regularity", "equivalence",
    "falsification_criterion", "literature_policy",
)
TARGET_LIST_FIELDS = (
    "domains", "codomains", "measures", "boundary_conditions", "symmetries",
    "premises", "modeling_postulates", "search_priors", "permitted_theorems",
)
NONEMPTY_TARGET_LIST_FIELDS = {"domains", "codomains", "premises"}
FINAL_REPORT_HEADINGS = (
    "# Rigorous theory search report",
    "## Frozen contract",
    "## Terminal status",
    "## Certificate",
    "## Strongest verified result",
    "## Dependency closure",
    "## Independent reconstruction",
    "## Oracle erasure",
    "## Unresolved obligations",
    "## Scope and limitations",
)
METADATA_RE = re.compile(r"<!-- rigorous-theory-search-metadata (\{.*\}) -->\Z")
SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
SCAFFOLD_MARKERS = (
    "candidate mechanism to be specified",
    "candidate representation to be specified",
    "candidate invariant or obstruction to be investigated",
    "state the precise proof or counterexample obligation",
    "no bridge result has been established",
    "specify a falsifying instance or failed obligation",
    "all mathematical obligations remain open",
    "candidate only; novelty is unassessed",
    "no counterexample search has been recorded",
    "no construction or theorem has been established",
    "this scaffold is a structural checkpoint only",
    "candidate-run report",
)
ELIGIBLE_EVIDENCE = {
    "MATHEMATICAL": {
        "EVIDENCE_VERIFIED": MATH_EVIDENCE,
        "REFUTED": NEGATIVE_CERTIFICATES,
    },
    "PHYSICAL_INTERPRETATION": {
        "EVIDENCE_VERIFIED": MATH_EVIDENCE | {"PRIMARY_SOURCE", "NUMERICAL"},
        "REFUTED": NEGATIVE_CERTIFICATES | {"PRIMARY_SOURCE", "NUMERICAL"},
    },
    "MODELING_POSTULATE": {
        "EVIDENCE_VERIFIED": MATH_EVIDENCE | {"PRIMARY_SOURCE"},
        "REFUTED": NEGATIVE_CERTIFICATES | {"PRIMARY_SOURCE"},
    },
    "OPERATIONAL_IDENTIFICATION": {
        "EVIDENCE_VERIFIED": MATH_EVIDENCE | {"PRIMARY_SOURCE", "NUMERICAL"},
        "REFUTED": NEGATIVE_CERTIFICATES | {"PRIMARY_SOURCE", "NUMERICAL"},
    },
    "NUMERICAL_OBSERVATION": {
        "EVIDENCE_VERIFIED": {"NUMERICAL", "SYMBOLIC_CHECK"},
        "REFUTED": {"NUMERICAL", "SYMBOLIC_CHECK", "COUNTEREXAMPLE"},
    },
}


def _load(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {token}")
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid JSON {path.name}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{path.name}: expected JSON object")
        return {}
    return value


def _required_artifacts(run_dir: Path, errors: list[str]) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for name in JSON_ARTIFACTS:
        path = run_dir / name
        if not path.is_file():
            errors.append(f"missing required artifact: {name}")
        else:
            documents[name] = _load(path, errors)
    for name in MARKDOWN_ARTIFACTS:
        path = run_dir / name
        try:
            present = path.is_file() and bool(path.read_text(encoding="utf-8").strip())
        except OSError:
            present = False
        if not present:
            errors.append(f"missing or empty required artifact: {name}")
    return documents


def _target_digest(target: Any, errors: list[str]) -> str:
    if not isinstance(target, dict):
        return ""
    try:
        canonical = json.dumps(
            target,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        errors.append(f"problem-contract.json: target is not canonical JSON: {exc}")
        return ""
    return hashlib.sha256(canonical).hexdigest()


def _markdown_metadata(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        first_line = path.read_text(encoding="utf-8").lstrip("\ufeff").splitlines()[0]
    except (OSError, IndexError):
        return {}
    match = METADATA_RE.fullmatch(first_line)
    if match is None:
        errors.append(f"{path.name}: missing metadata header")
        return {}
    try:
        metadata = json.loads(match.group(1))
    except json.JSONDecodeError:
        errors.append(f"{path.name}: invalid metadata header")
        return {}
    if not isinstance(metadata, dict):
        errors.append(f"{path.name}: invalid metadata header")
        return {}
    return metadata


def _schema_and_contract(
    run_dir: Path,
    documents: dict[str, dict[str, Any]],
    errors: list[str],
) -> str:
    problem = documents.get("problem-contract.json", {})
    digest = _target_digest(problem.get("target"), errors)
    expected_contract = f"contract-sha256-{digest}" if digest else ""
    contract = problem.get("contract_id")
    if not isinstance(contract, str) or not contract:
        errors.append("problem-contract.json: missing contract_id")
    elif expected_contract and contract != expected_contract:
        errors.append("problem-contract.json: canonical target digest does not match contract_id")
    _json_bindings(documents, contract, digest, errors)
    _markdown_bindings(run_dir, contract, digest, errors)
    return digest


def _json_bindings(
    documents: dict[str, dict[str, Any]],
    contract: Any,
    digest: str,
    errors: list[str],
) -> None:
    for name, document in documents.items():
        if document.get("schema_version") != SCHEMA:
            errors.append(f"{name}: unsupported schema_version")
        if contract and document.get("contract_id") != contract:
            errors.append(f"{name}: mismatched contract_id")
        if digest and document.get("target_digest") != digest:
            errors.append(f"{name}: mismatched target_digest")


def _markdown_bindings(
    run_dir: Path,
    contract: Any,
    digest: str,
    errors: list[str],
) -> None:
    for name in MARKDOWN_ARTIFACTS:
        path = run_dir / name
        if not path.is_file():
            continue
        metadata = _markdown_metadata(path, errors)
        if metadata and metadata.get("schema_version") != SCHEMA:
            errors.append(f"{name}: unsupported schema_version")
        if metadata and contract and metadata.get("contract_id") != contract:
            errors.append(f"{name}: mismatched contract_id")
        if metadata and digest and metadata.get("target_digest") != digest:
            errors.append(f"{name}: mismatched target_digest")


def _target_schema_fields(target: Any, errors: list[str]) -> None:
    if not isinstance(target, dict):
        errors.append("problem-contract.json: missing target")
        return
    target_fields = (
        *TARGET_STRING_FIELDS, *TARGET_LIST_FIELDS, "quantifier_class",
        "negative_certificate_kind",
    )
    for field in target_fields:
        if field not in target:
            errors.append(f"target: missing {field}")
    for field in TARGET_STRING_FIELDS:
        value = target.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"target: {field} must be a nonempty string")
    _target_list_fields(target, errors)
    if target.get("kind") not in CLAIM_KINDS:
        errors.append("target: invalid kind")
    if target.get("quantifier_class") not in {"UNIVERSAL", "EXISTENTIAL", "MIXED"}:
        errors.append("target: invalid quantifier_class")
    _negative_certificate_schema(target, errors)


def _target_list_fields(target: dict[str, Any], errors: list[str]) -> None:
    for field in TARGET_LIST_FIELDS:
        value = target.get(field)
        valid_items = isinstance(value, list) and all(
            isinstance(item, str) and bool(item.strip()) for item in value
        )
        if not valid_items or field in NONEMPTY_TARGET_LIST_FIELDS and not value:
            requirement = "nonempty list" if field in NONEMPTY_TARGET_LIST_FIELDS else "list"
            errors.append(f"target: {field} must be a {requirement} of nonempty strings")


def _negative_certificate_schema(target: dict[str, Any], errors: list[str]) -> None:
    certificate = target.get("negative_certificate_kind")
    if certificate not in NEGATIVE_CERTIFICATES:
        errors.append(
            "target: negative_certificate_kind must be COUNTEREXAMPLE or NONEXISTENCE_PROOF"
        )
        return
    quantifier = target.get("quantifier_class")
    if quantifier == "UNIVERSAL" and certificate != "COUNTEREXAMPLE":
        errors.append("target: UNIVERSAL requires negative_certificate_kind COUNTEREXAMPLE")
    if quantifier == "EXISTENTIAL" and certificate != "NONEXISTENCE_PROOF":
        errors.append("target: EXISTENTIAL requires negative_certificate_kind NONEXISTENCE_PROOF")


def _family_id(
    family: dict[str, Any], index: int, family_ids: set[str], errors: list[str]
) -> None:
    if "family_id" not in family:
        return
    family_id = family["family_id"]
    if not isinstance(family_id, str) or not family_id.strip():
        errors.append(f"mechanism family {index}: family_id must be a nonempty string")
    elif family_id in family_ids:
        errors.append(f"duplicate mechanism family ID: {family_id}")
    else:
        family_ids.add(family_id)


def _family_core_mechanism(family: dict[str, Any], index: int, errors: list[str]) -> None:
    if "core_mechanism" not in family:
        return
    value = family["core_mechanism"]
    if not isinstance(value, str) or not value.strip():
        errors.append(f"mechanism family {index}: core_mechanism must be a nonempty string")


def _family_obligation_ids(family: dict[str, Any], index: int, errors: list[str]) -> None:
    if "target_obligation_ids" not in family:
        return
    value = family["target_obligation_ids"]
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        errors.append(
            f"mechanism family {index}: target_obligation_ids must be a list of nonempty strings"
        )


def _schema_fields(documents: dict[str, dict[str, Any]], errors: list[str]) -> None:
    _target_schema_fields(documents.get("problem-contract.json", {}).get("target"), errors)
    approaches = documents.get("approach-registry.json", {}).get("mechanism_families")
    required = (
        "family_id", "core_mechanism", "target_obligation_ids", "representation",
        "invariant_or_obstruction", "obligations", "bridge", "failure_test",
        "verified_results", "open_gaps", "novelty_fingerprint", "disposition",
    )
    if not isinstance(approaches, list):
        errors.append("approach-registry.json: mechanism_families must be a list")
        return
    family_ids: set[str] = set()
    for index, family in enumerate(approaches):
        if not isinstance(family, dict):
            errors.append(f"mechanism family {index}: expected object")
            continue
        for field in required:
            if field not in family:
                errors.append(f"mechanism family {index}: missing {field}")
        _family_id(family, index, family_ids, errors)
        _family_core_mechanism(family, index, errors)
        _family_obligation_ids(family, index, errors)


def _ids(records: Any, label: str, errors: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(records, list):
        errors.append(f"{label}: expected list")
        return result
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            errors.append(f"{label}: record missing ID")
            continue
        record_id = record["id"]
        if record_id in result:
            errors.append(f"duplicate {label[:-1]} ID: {record_id}")
            continue
        result[record_id] = record
    return result


def _safe_path(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and "\x00" not in value and "\\" not in value and ":" not in value and not PurePosixPath(value).is_absolute() and ".." not in PurePosixPath(value).parts


def _list_field(record: dict[str, Any], field: str) -> list[Any]:
    value = record.get(field, [])
    return value if isinstance(value, list) else []


def _path_has_link(run_dir: Path, relative: PurePosixPath) -> bool:
    current = run_dir
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return True
        is_junction = getattr(current, "is_junction", None)
        if callable(is_junction) and is_junction():
            return True
    return False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_artifact_file(
    run_dir: Path,
    record: dict[str, Any],
    label: str,
    errors: list[str],
) -> None:
    reference = _artifact_reference(record, label, errors)
    if reference is None:
        return
    relative, declared = reference
    candidate = _contained_regular_file(run_dir, relative, label, errors)
    if candidate is None:
        return
    try:
        actual = _file_sha256(candidate)
    except OSError as exc:
        errors.append(f"{label}: artifact cannot be read: {exc}")
        return
    if actual != declared.lower():
        errors.append(f"{label}: artifact_sha256 mismatch")


def _artifact_reference(
    record: dict[str, Any],
    label: str,
    errors: list[str],
) -> tuple[PurePosixPath, str] | None:
    value = record.get("artifact_path")
    declared = record.get("artifact_sha256")
    path_valid = _safe_path(value)
    digest_valid = isinstance(declared, str) and SHA256_RE.fullmatch(declared) is not None
    if not path_valid:
        errors.append(f"{label}: unsafe artifact_path")
    if not digest_valid:
        errors.append(f"{label}: invalid artifact_sha256")
    if not path_valid or not digest_valid:
        return None
    return PurePosixPath(value), declared


def _contained_regular_file(
    run_dir: Path,
    relative: PurePosixPath,
    label: str,
    errors: list[str],
) -> Path | None:
    candidate = run_dir.joinpath(*relative.parts)
    if _path_has_link(run_dir, relative):
        errors.append(f"{label}: artifact path contains a symlink")
        return None
    if not candidate.exists():
        errors.append(f"{label}: artifact is missing")
        return None
    try:
        root = run_dir.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        contained = resolved.is_relative_to(root)
        regular = stat.S_ISREG(candidate.stat(follow_symlinks=False).st_mode)
    except OSError as exc:
        errors.append(f"{label}: artifact cannot be inspected: {exc}")
        return None
    if not contained:
        errors.append(f"{label}: artifact escapes run directory")
        return None
    if not regular:
        errors.append(f"{label}: artifact is not a regular file")
        return None
    return candidate


def _validate_evidence(
    run_dir: Path,
    evidence: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    for evidence_id, record in evidence.items():
        if record.get("kind") not in EVIDENCE_KINDS:
            errors.append(f"evidence {evidence_id}: invalid kind")
        if not isinstance(record.get("supports"), bool):
            errors.append(f"evidence {evidence_id}: supports must be boolean")
        scope = record.get("scope")
        if not isinstance(scope, str) or not scope.strip():
            errors.append(f"evidence {evidence_id}: scope must be a nonempty string")
        side_conditions = record.get("side_conditions")
        if not isinstance(side_conditions, list) or any(
            not isinstance(item, str) or not item.strip() for item in side_conditions
        ):
            errors.append(f"evidence {evidence_id}: side_conditions must be a list of strings")
        _validate_artifact_file(run_dir, record, f"evidence {evidence_id}", errors)


def _claim_shape(
    claim_id: str,
    record: dict[str, Any],
    target_digest: str,
    errors: list[str],
) -> None:
    if record.get("kind") not in CLAIM_KINDS:
        errors.append(f"claim {claim_id}: invalid kind")
    if record.get("state") not in STATES:
        errors.append(f"claim {claim_id}: invalid state")
    for field in ("statement", "quantifiers", "falsifier"):
        if not _nonempty_string(record.get(field)):
            errors.append(f"claim {claim_id}: {field} must be a nonempty string")
    if target_digest and record.get("target_digest") != target_digest:
        errors.append(f"claim {claim_id}: mismatched target_digest")


def _claim_references(
    claim_id: str,
    record: dict[str, Any],
    assumptions: dict[str, dict[str, Any]],
    evidence: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    reference_fields = (
        ("assumption_ids", assumptions, "assumption"),
        ("evidence_ids", evidence, "evidence"),
        ("bridge_premise_ids", assumptions, "bridge"),
    )
    for field, known, noun in reference_fields:
        values = record.get(field, [])
        if not isinstance(values, list):
            errors.append(f"claim {claim_id}: {field} must be a list")
            continue
        for item in values:
            if item not in known:
                errors.append(f"claim {claim_id}: unknown {noun} {item}")


def _references(
    claims: dict[str, dict[str, Any]],
    assumptions: dict[str, dict[str, Any]],
    evidence: dict[str, dict[str, Any]],
    target_digest: str,
    errors: list[str],
) -> None:
    for claim_id, record in claims.items():
        _claim_shape(claim_id, record, target_digest, errors)
        _claim_references(claim_id, record, assumptions, evidence, errors)


def _graph(claims: dict[str, dict[str, Any]], edges: Any, errors: list[str]) -> dict[str, list[str]]:
    graph = {claim_id: [] for claim_id in claims}
    if not isinstance(edges, list):
        errors.append("dependency-dag.json: edges must be a list")
        return graph
    for edge in edges:
        if not isinstance(edge, dict):
            errors.append("dependency-dag.json: invalid edge")
            continue
        source, destination = edge.get("from"), edge.get("to")
        if source == "SEARCH_PRIOR_AFFIRMATIVE" or destination == "SEARCH_PRIOR_AFFIRMATIVE":
            errors.append("SEARCH_PRIOR_AFFIRMATIVE is forbidden in dependency endpoints")
        if source not in claims or destination not in claims:
            errors.append(f"unknown dependency: {source} -> {destination}")
        else:
            graph[source].append(destination)
    return graph


def _cycles(graph: dict[str, list[str]], errors: list[str]) -> None:
    active: list[str] = []
    seen: set[str] = set()
    emitted: set[tuple[str, ...]] = set()
    def visit(node: str) -> None:
        if node in active:
            cycle = active[active.index(node):]
            rotated = min(tuple(cycle[index:] + cycle[:index]) for index in range(len(cycle)))
            if rotated not in emitted:
                emitted.add(rotated)
                errors.append("dependency cycle: " + " -> ".join((*rotated, rotated[0])))
            return
        if node in seen:
            return
        active.append(node)
        for neighbor in sorted(graph[node]):
            visit(neighbor)
        active.pop()
        seen.add(node)
    for node in sorted(graph):
        visit(node)


def _contains_search_prior(value: Any) -> bool:
    if isinstance(value, str):
        return "SEARCH_PRIOR_AFFIRMATIVE" in value
    if isinstance(value, dict):
        return any(_contains_search_prior(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_search_prior(item) for item in value)
    return False


def _search_prior_leaks(contract: dict[str, Any], ledger: dict[str, Any], errors: list[str]) -> None:
    target = contract.get("target", {})
    if not isinstance(target, dict):
        if _contains_search_prior(target):
            errors.append("SEARCH_PRIOR_AFFIRMATIVE is forbidden outside target.search_priors")
        return
    allowed = target.get("search_priors", [])
    if _contains_search_prior({key: value for key, value in target.items() if key != "search_priors"}):
        errors.append("SEARCH_PRIOR_AFFIRMATIVE is forbidden outside target.search_priors")
    if _contains_search_prior(ledger.get("assumptions", [])) or _contains_search_prior(ledger.get("evidence", [])):
        errors.append("SEARCH_PRIOR_AFFIRMATIVE is forbidden in assumptions or evidence")
    if not isinstance(allowed, list):
        errors.append("problem-contract target.search_priors must be a list")


def _all_search_prior_leaks(run_dir: Path, documents: dict[str, dict[str, Any]], errors: list[str]) -> None:
    for name, document in documents.items():
        candidate: Any = document
        if name == "problem-contract.json" and isinstance(document.get("target"), dict):
            target = dict(document["target"])
            target.pop("search_priors", None)
            candidate = dict(document, target=target)
        if _contains_search_prior(candidate):
            errors.append(f"SEARCH_PRIOR_AFFIRMATIVE is forbidden in {name}")
    for name in MARKDOWN_ARTIFACTS:
        path = run_dir / name
        if path.is_file() and "SEARCH_PRIOR_AFFIRMATIVE" in path.read_text(encoding="utf-8"):
            errors.append(f"SEARCH_PRIOR_AFFIRMATIVE is forbidden in {name}")


def _ancestors(target: str, graph: dict[str, list[str]]) -> set[str]:
    result: set[str] = set()
    stack = list(graph.get(target, []))
    while stack:
        node = stack.pop()
        if node not in result:
            result.add(node)
            stack.extend(graph.get(node, []))
    return result


def _evidence_kinds(record: dict[str, Any], evidence: dict[str, dict[str, Any]]) -> set[str]:
    return {str(evidence.get(item, {}).get("kind")) for item in _list_field(record, "evidence_ids")}


def _claim_eligibility(claims: dict[str, dict[str, Any]], evidence: dict[str, dict[str, Any]], errors: list[str]) -> None:
    for claim_id, record in claims.items():
        state = record.get("state")
        kind = record.get("kind")
        if state not in {"EVIDENCE_VERIFIED", "REFUTED"} or kind not in ELIGIBLE_EVIDENCE:
            continue
        expected_support = state == "EVIDENCE_VERIFIED"
        eligible = ELIGIBLE_EVIDENCE[kind][state]
        direct = [evidence.get(item, {}) for item in _list_field(record, "evidence_ids")]
        if not any(
            item.get("kind") in eligible and item.get("supports") is expected_support
            for item in direct
        ):
            errors.append(f"claim {claim_id} lacks eligible {state} evidence")
        kinds = {str(item.get("kind")) for item in direct if item.get("supports") is expected_support}
        if kind == "MATHEMATICAL" and state == "EVIDENCE_VERIFIED" and not kinds & MATH_EVIDENCE:
            errors.append(f"mathematical claim {claim_id} lacks direct derivation, formal proof, or applicable theorem")
        if kind == "MATHEMATICAL" and state == "REFUTED" and not kinds & NEGATIVE_CERTIFICATES:
            errors.append(f"refuted mathematical claim {claim_id} lacks direct counterexample or nonexistence proof")


def _target_claim_binding(
    documents: dict[str, dict[str, Any]],
    claims: dict[str, dict[str, Any]],
    target_digest: str,
    errors: list[str],
) -> None:
    target_id = documents.get("release.json", {}).get("target_claim")
    target = documents.get("problem-contract.json", {}).get("target")
    if target_id not in claims:
        errors.append("release.json: target_claim must name a claim")
        return
    if not isinstance(target, dict):
        return
    record = claims[target_id]
    expected = {
        "statement": target.get("statement"),
        "kind": target.get("kind"),
        "quantifiers": target.get("quantifiers"),
        "target_digest": target_digest,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            errors.append(f"target claim {target_id}: mismatched {field}")


def _terminal_target_check(status: str, target: str, release: dict[str, Any], claims: dict[str, dict[str, Any]], errors: list[str]) -> bool:
    required_state = {"COMPLETE_AFFIRMATIVE": "EVIDENCE_VERIFIED", "COMPLETE_NEGATIVE": "REFUTED", "INCONCLUSIVE": "INCONCLUSIVE"}[status]
    if claims[target].get("state") != required_state:
        errors.append(f"{status} target claim {target} must be {required_state}")
    if status == "INCONCLUSIVE":
        return True
    certificate = release.get("certificate_claim")
    if certificate not in claims:
        errors.append("release.json: certificate_claim must resolve to a claim")
        return False
    if certificate != target:
        errors.append("release.json: certificate relationship requires certificate_claim == target_claim")
        return False
    if claims[certificate].get("state") != required_state:
        errors.append(f"{status} certificate claim {certificate} must be {required_state}")
    return True


def _affirmative_checks(target: str, claims: dict[str, dict[str, Any]], evidence: dict[str, dict[str, Any]], graph: dict[str, list[str]], errors: list[str]) -> None:
    for claim_id in {target, *_ancestors(target, graph)}:
        record = claims[claim_id]
        if record.get("state") != "EVIDENCE_VERIFIED":
            prefix = "target" if claim_id == target else "ancestor"
            errors.append(f"complete affirmative {prefix} {claim_id} is not EVIDENCE_VERIFIED")


def _negative_checks(documents: dict[str, dict[str, Any]], release: dict[str, Any], claims: dict[str, dict[str, Any]], evidence: dict[str, dict[str, Any]], errors: list[str]) -> None:
    target = documents.get("problem-contract.json", {}).get("target")
    required = target.get("negative_certificate_kind") if isinstance(target, dict) else None
    if required not in NEGATIVE_CERTIFICATES:
        return
    record = claims.get(release.get("target_claim"), {})
    if not any(
        evidence.get(item, {}).get("kind") == required
        and evidence.get(item, {}).get("supports") is False
        for item in _list_field(record, "evidence_ids")
    ):
        errors.append(f"complete negative release lacks {required} negative certificate")


def _inconclusive_checks(release: dict[str, Any], errors: list[str]) -> None:
    if not isinstance(release.get("strongest_result"), str) or not release["strongest_result"].strip():
        errors.append("inconclusive release requires strongest_result")
    if not isinstance(release.get("unresolved_obligations"), list) or not release["unresolved_obligations"]:
        errors.append("inconclusive release requires unresolved_obligations")


def _physical_checks(target: str, record: dict[str, Any], assumptions: dict[str, dict[str, Any]], evidence: dict[str, dict[str, Any]], errors: list[str]) -> None:
    if record.get("kind") != "PHYSICAL_INTERPRETATION" or record.get("state") not in {"EVIDENCE_VERIFIED", "REFUTED"}:
        return
    state = str(record["state"])
    expected_support = state == "EVIDENCE_VERIFIED"
    eligible = ELIGIBLE_EVIDENCE["PHYSICAL_INTERPRETATION"][state]
    bridge_ok = any(assumptions.get(item, {}).get("kind") in {"MODELING_POSTULATE", "OPERATIONAL_IDENTIFICATION"} for item in _list_field(record, "bridge_premise_ids"))
    direct_ok = any(
        evidence.get(item, {}).get("kind") in eligible
        and evidence.get(item, {}).get("supports") is expected_support
        for item in _list_field(record, "evidence_ids")
    )
    if not bridge_ok or not direct_ok:
        errors.append(f"physical interpretation {target} requires a declared bridge and direct eligible evidence")


def _all_physical_checks(claims: dict[str, dict[str, Any]], assumptions: dict[str, dict[str, Any]], evidence: dict[str, dict[str, Any]], errors: list[str]) -> None:
    for claim_id, record in claims.items():
        if record.get("state") in {"EVIDENCE_VERIFIED", "REFUTED"}:
            _physical_checks(claim_id, record, assumptions, evidence, errors)


def _release_target(release: dict[str, Any], claims: dict[str, dict[str, Any]], errors: list[str]) -> tuple[str, str] | None:
    status, target = release.get("terminal_status"), release.get("target_claim")
    if status not in TERMINAL_STATUSES:
        errors.append("release.json: invalid terminal_status")
        return None
    if target not in claims:
        errors.append("release.json: target_claim must name a claim")
        return None
    return status, target


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _claim_id_list(
    value: Any,
    label: str,
    claims: dict[str, dict[str, Any]],
    errors: list[str],
) -> set[str]:
    if not isinstance(value, list) or not value or any(not _nonempty_string(item) for item in value):
        errors.append(f"{label}: claim_ids must be a nonempty list of claim IDs")
        return set()
    result = set(value)
    for claim_id in sorted(result - claims.keys()):
        errors.append(f"{label}: unknown claim {claim_id}")
    return result & claims.keys()


def _attack_checks(
    run_dir: Path,
    attacks: Any,
    claims: dict[str, dict[str, Any]],
    complete_release: bool,
    errors: list[str],
) -> set[str]:
    if not isinstance(attacks, list) or not attacks:
        errors.append("adversarial-report.json: attacks must be a substantive nonempty list")
        return set()
    covered: set[str] = set()
    seen: set[str] = set()
    for index, attack in enumerate(attacks):
        if not isinstance(attack, dict):
            errors.append(f"attack {index}: expected object")
            continue
        covered.update(_attack_record_checks(
            run_dir, attack, index, seen, claims, complete_release, errors,
        ))
    return covered


def _attack_record_checks(
    run_dir: Path,
    attack: dict[str, Any],
    index: int,
    seen: set[str],
    claims: dict[str, dict[str, Any]],
    complete_release: bool,
    errors: list[str],
) -> set[str]:
    attack_id = attack.get("id")
    label = f"attack {attack_id if _nonempty_string(attack_id) else index}"
    if not _nonempty_string(attack_id) or attack_id in seen:
        errors.append(f"{label}: id must be a unique nonempty string")
    else:
        seen.add(attack_id)
    for field in ("attack", "response"):
        if not _nonempty_string(attack.get(field)):
            errors.append(f"{label}: {field} must be substantive")
    for field in ("artifact_path", "artifact_sha256"):
        if not _nonempty_string(attack.get(field)):
            errors.append(f"{label}: {field} must be nonempty")
    disposition = attack.get("disposition")
    if disposition not in {"REJECTED", "SUSTAINED", "PARTIALLY_SUSTAINED"}:
        errors.append(f"{label}: invalid disposition")
    elif complete_release and disposition != "REJECTED":
        errors.append(f"{label}: {disposition} attack blocks a complete release")
    covered = _claim_id_list(attack.get("claim_ids"), label, claims, errors)
    _validate_artifact_file(run_dir, attack, label, errors)
    return covered


def _adversarial_record_checks(
    run_dir: Path,
    adversarial: dict[str, Any],
    name: str,
    claims: dict[str, dict[str, Any]],
    errors: list[str],
) -> set[str]:
    record = adversarial.get(name)
    label = f"adversarial-report.json: {name}"
    if not isinstance(record, dict):
        errors.append(f"{label} must be artifact-backed and substantive")
        return set()
    required = ("method", "conclusion", "artifact_path", "artifact_sha256")
    if any(not _nonempty_string(record.get(field)) for field in required):
        errors.append(f"{label} must be artifact-backed and substantive")
    if record.get("result") != "PASS":
        errors.append(f"{label} result must be PASS")
    covered = _claim_id_list(record.get("claim_ids"), label, claims, errors)
    _validate_artifact_file(run_dir, record, label, errors)
    return covered


def _coverage_error(label: str, covered: set[str], closure: set[str], errors: list[str]) -> None:
    missing = sorted(closure - covered)
    if missing:
        errors.append(f"{label} omit{'s' if not label.endswith('attacks') else ''} dependency-closure claims: {', '.join(missing)}")


def _adversarial_checks(
    run_dir: Path,
    adversarial: dict[str, Any],
    target: str,
    status: str,
    claims: dict[str, dict[str, Any]],
    graph: dict[str, list[str]],
    errors: list[str],
) -> None:
    closure = {target, *_ancestors(target, graph)}
    attacks = _attack_checks(
        run_dir,
        adversarial.get("attacks"),
        claims,
        status in {"COMPLETE_AFFIRMATIVE", "COMPLETE_NEGATIVE"},
        errors,
    )
    reconstruction = _adversarial_record_checks(
        run_dir, adversarial, "independent_reconstruction", claims, errors,
    )
    erasure = _adversarial_record_checks(
        run_dir, adversarial, "oracle_erasure", claims, errors,
    )
    _coverage_error("attacks", attacks, closure, errors)
    _coverage_error("independent_reconstruction", reconstruction, closure, errors)
    _coverage_error("oracle_erasure", erasure, closure, errors)


def _contains_scaffold_marker(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.casefold()
        return any(marker in lowered for marker in SCAFFOLD_MARKERS)
    if isinstance(value, list):
        return any(_contains_scaffold_marker(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_scaffold_marker(item) for item in value.values())
    return False


def _release_portfolio_checks(documents: dict[str, dict[str, Any]], errors: list[str]) -> None:
    families = documents.get("approach-registry.json", {}).get("mechanism_families")
    if not isinstance(families, list) or not families:
        errors.append("release requires a nonempty mechanism portfolio")
        return
    if _contains_scaffold_marker(families):
        errors.append("mechanism portfolio contains scaffold sentinel content")


def _markdown_headings(text: str) -> tuple[str, ...]:
    headings: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        marker = stripped[:3]
        if marker in {"```", "~~~"}:
            fence = None if fence == marker else marker if fence is None else fence
            continue
        if fence is None and re.fullmatch(r"#{1,6} [^#\s].*", stripped):
            headings.append(stripped)
    return tuple(headings)


def _release_markdown_checks(run_dir: Path, errors: list[str]) -> None:
    for name in MARKDOWN_ARTIFACTS:
        path = run_dir / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if _contains_scaffold_marker(text):
            errors.append(f"{name}: scaffold content is not releasable")
        if name == "final-report.md" and _markdown_headings(text) != FINAL_REPORT_HEADINGS:
            errors.append("final-report.md: headings must exactly match the release contract")


def _terminal_dependency_checks(
    status: str,
    target: str,
    claims: dict[str, dict[str, Any]],
    graph: dict[str, list[str]],
    errors: list[str],
) -> None:
    for claim_id in sorted(_ancestors(target, graph)):
        state = claims[claim_id].get("state")
        if status in {"COMPLETE_AFFIRMATIVE", "COMPLETE_NEGATIVE"} and state != "EVIDENCE_VERIFIED":
            errors.append(
                f"{status} dependency ancestor {state}: {claim_id} must be EVIDENCE_VERIFIED"
            )
        elif state in {"CANDIDATE", "LLM_SUPPORTED"}:
            errors.append(f"{status} dependency ancestor {state}: {claim_id}")


def _terminal_status_checks(status: str, documents: dict[str, dict[str, Any]], release: dict[str, Any], target: str, claims: dict[str, dict[str, Any]], evidence: dict[str, dict[str, Any]], graph: dict[str, list[str]], certificate_valid: bool, errors: list[str]) -> None:
    _terminal_dependency_checks(status, target, claims, graph, errors)
    if status == "COMPLETE_AFFIRMATIVE":
        _affirmative_checks(target, claims, evidence, graph, errors)
    elif status == "COMPLETE_NEGATIVE":
        if certificate_valid:
            _negative_checks(documents, release, claims, evidence, errors)
    else:
        _inconclusive_checks(release, errors)


def _release_checks(
    run_dir: Path,
    documents: dict[str, dict[str, Any]],
    claims: dict[str, dict[str, Any]],
    evidence: dict[str, dict[str, Any]],
    graph: dict[str, list[str]],
    mode: str,
    errors: list[str],
) -> None:
    release = documents.get("release.json", {})
    if mode == "checkpoint":
        if release.get("terminal_status") is not None:
            errors.append("release.json: checkpoint mode requires terminal_status null")
        return
    resolved = _release_target(release, claims, errors)
    if resolved is None:
        return
    status, target = resolved
    certificate_valid = _terminal_target_check(status, target, release, claims, errors)
    _release_portfolio_checks(documents, errors)
    _release_markdown_checks(run_dir, errors)
    _adversarial_checks(
        run_dir,
        documents.get("adversarial-report.json", {}),
        target,
        status,
        claims,
        graph,
        errors,
    )
    _terminal_status_checks(status, documents, release, target, claims, evidence, graph, certificate_valid, errors)


def validate_run(run_dir: Path, *, mode: str = "checkpoint") -> list[str]:
    """Return sorted structural diagnostics; this cannot establish mathematical truth."""
    if mode not in {"checkpoint", "release"}:
        return [f"invalid mode: {mode}"]
    errors: list[str] = []
    documents = _required_artifacts(run_dir, errors)
    target_digest = _schema_and_contract(run_dir, documents, errors)
    _schema_fields(documents, errors)
    ledger = documents.get("claim-ledger.json", {})
    assumptions = _ids(ledger.get("assumptions", []), "assumptions", errors)
    evidence = _ids(ledger.get("evidence", []), "evidence", errors)
    claims = _ids(ledger.get("claims", []), "claims", errors)
    _validate_evidence(run_dir, evidence, errors)
    _references(claims, assumptions, evidence, target_digest, errors)
    _target_claim_binding(documents, claims, target_digest, errors)
    _claim_eligibility(claims, evidence, errors)
    graph = _graph(claims, documents.get("dependency-dag.json", {}).get("edges", []), errors)
    _cycles(graph, errors)
    _search_prior_leaks(documents.get("problem-contract.json", {}), ledger, errors)
    _all_search_prior_leaks(run_dir, documents, errors)
    _all_physical_checks(claims, assumptions, evidence, errors)
    _release_checks(run_dir, documents, claims, evidence, graph, mode, errors)
    return sorted(set(errors))


def main(argv: list[str] | None = None) -> int:
    """Run structural validation; successful output does not prove mathematical truth."""
    parser = argparse.ArgumentParser(description="Validate run structure; it cannot establish mathematical truth.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--mode", choices=("checkpoint", "release"), default="checkpoint")
    arguments = parser.parse_args(argv)
    diagnostics = validate_run(arguments.run_dir, mode=arguments.mode)
    for diagnostic in diagnostics:
        print(diagnostic)
    return 1 if diagnostics else 0


if __name__ == "__main__":
    raise SystemExit(main())
