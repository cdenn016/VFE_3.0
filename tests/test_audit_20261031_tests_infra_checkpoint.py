r"""Every callable that vfe3 hands to ``torch.utils.checkpoint`` recomputes under the fp32 matmul policy of its
forward (audit 2026-10-31 gap-checkpoint-recompute-semantics-6).

A non-reentrant checkpoint recomputes its callable in backward under the policy current at that time, and
``train_step`` calls ``backward()`` outside every :func:`vfe3.runtime.ieee_fp32_matmul` scope. A tile whose forward
ran inside a pinned scope therefore recomputes at the process policy ("high" on the launcher) unless the tile pins
itself, and four tiles did not, which no test could detect. The static test lists every checkpoint call in the
package and requires each callable to be classified:

* pinned: decorated by ``ieee_fp32_matmul()`` (the wrapper carries ``__wrapped__`` and the pin's context manager in
  its closure), or a body whose GEMMs all lie inside a ``with ieee_fp32_matmul()`` block;
* elementwise: no GEMM, so the policy cannot change the recompute;
* caller policy: never entered from a pinned scope, so forward and recompute run under the same policy (GEMMs the
  callable pins itself, such as ``_decode_av``, stay pinned in both).

A new checkpoint call fails until it is classified, and a classified entry whose call is gone fails until it is
removed. The dynamic test checks the caller-policy premise on the routes that reach those callables, with the
process policy at "high". The four tiles that lacked the pin, and the emulated real-versus-pass-through A/B of
their recompute, belong to the minimal-agent remediation (A014, A015); with that patch merged the two spread tiles
are classified as pinned, the two minimal-agent tiles no longer pass through checkpoint, and
:data:`_PENDING_MINIMAL_AGENT` is empty.
"""
import ast
import contextlib
import importlib
import inspect
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest
import torch
import torch.utils.checkpoint

import vfe3
from tests.profiles import live_values
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.runtime import ieee_fp32_matmul, restored_fp32_matmul_precision

_PACKAGE = Path(vfe3.__file__).resolve().parent
_PIN_GENERATOR = inspect.unwrap(ieee_fp32_matmul)          # the generator function under @contextmanager
_GEMM_NAMES = frozenset({"einsum", "matmul", "mm", "bmm", "baddbmm", "addmm", "addbmm", "addmv", "mv",
                         "tensordot", "linear", "chain_matmul", "multi_dot"})
# Package functions a body pinned by a with block may call outside that block (they run no GEMM).
_OUTSIDE_PIN_HELPERS = frozenset({"_compute_dtype"})

_FunctionNode = (ast.FunctionDef, ast.AsyncFunctionDef)
_Scope = Tuple[ast.AST, ...]

# Callable (module:qualname) -> classification; see the module docstring.
_CLASSIFIED: Dict[str, str] = {
    "vfe3.relational_scoring:_projected_moments_tile": "pinned",
    "vfe3.relational_scoring:_projected_root_factor_tile": "pinned",
    "vfe3.relational_scoring:_projected_kl_tile": "pinned",
    "vfe3.relational_scoring:_projected_kl_direct_tile": "pinned",
    "vfe3.relational_scoring:_diagonal_kl_direct_tile": "pinned",
    "vfe3.relational_scoring:_noisy_query_tile": "caller_policy",
    "vfe3.relational_scoring:_diagonal_noisy_query_tile": "elementwise",
    "vfe3.relational_scoring:_mean_distance_tile": "elementwise",
    # The type-level diagonal moments' token tiles (scoring package, relational_type_moments).
    "vfe3.relational_scoring:_diagonal_square_sums_tile": "pinned",
    # Pinned by the minimal-agent package (A014, A015), which also moved the two minimal-agent tiles off checkpoint.
    "vfe3.model.relational_operators:_centered_spread_tile": "pinned",
    "vfe3.model.relational_operators:_spread_squares_tile": "pinned",
    "vfe3.model.relational_operators:RelationalOperators._minimal_agent_body": "pinned",
    "vfe3.model.relational_operators:RelationalOperators._predictive_covariance_update": "pinned",
    "vfe3.model.prototype_agents:_exact_diagonal_tile": "pinned",
    "vfe3.model.prototype_agents:_exact_block_tile": "pinned",
    "vfe3.model.prototype_agents:_exact_em_tile": "pinned",
    "vfe3.model.cue_result_memory:_chunk_moments": "pinned",
    "vfe3.model.cue_result_memory:_diagonal_tile": "pinned",
    # The diagonal and full KL decoders share PriorBank._kl_decode_ce and its chunk closure (decode package).
    "vfe3.model.prior_bank:PriorBank._kl_decode_ce.<locals>._chunk_summaries": "caller_policy",
    "vfe3.model.prior_bank:PriorBank.decode_ce_linear_chunked.<locals>._chunk_summaries": "caller_policy",
    "vfe3.model.prior_bank:PriorBank.decode_ce_linear_chunked.<locals>._chunk_summaries_second_order":
        "caller_policy",
    "vfe3.model.prior_bank:PriorBank.decode_ce_expected_likelihood_chunked.<locals>._chunk_summaries":
        "caller_policy",
}
# Checkpointed without the pin although their forward runs pinned (lane findings 1 and 2). The minimal-agent
# remediation, now merged, pins the two spread tiles (classified above) and replaces the two minimal-agent
# checkpoints with an IEEE recompute Function, so nothing is pending.
_PENDING_MINIMAL_AGENT: frozenset = frozenset()


def _module_trees() -> Iterator[Tuple[str, ast.Module]]:
    for path in sorted(_PACKAGE.rglob("*.py")):
        parts = path.relative_to(_PACKAGE.parent).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        yield ".".join(parts), ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _maybe_checkpoint(call: ast.Call) -> bool:
    func = call.func
    return ((isinstance(func, ast.Name) and func.id == "checkpoint")
            or (isinstance(func, ast.Attribute) and func.attr == "checkpoint"))


def _is_torch_checkpoint(call: ast.Call, module: ModuleType) -> bool:
    func = call.func
    if isinstance(func, ast.Name):
        return getattr(module, func.id, None) is torch.utils.checkpoint.checkpoint
    if isinstance(func.value, ast.Name):
        return getattr(module, func.value.id, None) is torch.utils.checkpoint
    return ast.unparse(func.value) == "torch.utils.checkpoint"


def _checkpoint_sites() -> List[Tuple[str, ModuleType, ast.Module, _Scope, ast.Call]]:
    r"""(module name, module, tree, enclosing class and function nodes, call) of every checkpoint call."""
    sites = []
    for name, tree in _module_trees():
        calls: List[Tuple[_Scope, ast.Call]] = []

        def walk(node: ast.AST, scope: _Scope) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (*_FunctionNode, ast.ClassDef)):
                    walk(child, scope + (child,))
                    continue
                if isinstance(child, ast.Call) and _maybe_checkpoint(child):
                    calls.append((scope, child))
                walk(child, scope)

        walk(tree, ())
        if calls:
            module = importlib.import_module(name)
            sites += [(name, module, tree, scope, call) for scope, call in calls if _is_torch_checkpoint(call, module)]
    return sites


def _qualname(scope: _Scope) -> str:
    parts: List[str] = []
    for node in scope:
        parts.append(node.name)
        if isinstance(node, _FunctionNode):
            parts.append("<locals>")
    return ".".join(parts)


def _local_nodes(owner: ast.AST) -> Iterator[ast.AST]:
    r"""The nodes of ``owner``'s body, not descending into nested definitions (which are yielded themselves)."""
    stack = list(ast.iter_child_nodes(owner))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (*_FunctionNode, ast.ClassDef, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(node))


def _value_names(value: ast.expr) -> List[str]:
    if isinstance(value, ast.Name):
        return [value.id]
    if isinstance(value, ast.IfExp):
        return _value_names(value.body) + _value_names(value.orelse)
    raise AssertionError(f"checkpoint callable assigned from {ast.unparse(value)}")


def _assigned_names(owner: ast.AST, name: str) -> List[str]:
    r"""The names a local variable ``name`` of ``owner`` is assigned from (positional tuple unpacking included)."""
    sources: List[str] = []
    for node in _local_nodes(owner):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                sources += _value_names(node.value)
            elif isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                for element, value in zip(target.elts, node.value.elts):
                    if isinstance(element, ast.Name) and element.id == name:
                        sources += _value_names(value)
    return sources


def _candidates(module_name: str, tree: ast.Module, scope: _Scope,
                expr: ast.expr) -> List[Tuple[str, ast.AST]]:
    r"""(key, definition node) of every callable the checkpoint argument ``expr`` can name."""
    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name) and expr.value.id == "self":
        owner = next(node for node in reversed(scope) if isinstance(node, ast.ClassDef))
        method = next(node for node in owner.body if isinstance(node, _FunctionNode) and node.name == expr.attr)
        return [(f"{module_name}:{owner.name}.{expr.attr}", method)]
    assert isinstance(expr, ast.Name), f"unresolved checkpoint callable {ast.unparse(expr)}"
    for depth in range(len(scope), 0, -1):
        owner = scope[depth - 1]
        if not isinstance(owner, _FunctionNode):
            continue
        nested = [node for node in _local_nodes(owner) if isinstance(node, _FunctionNode) and node.name == expr.id]
        if nested:
            return [(f"{module_name}:{_qualname(scope[:depth])}.{expr.id}", nested[0])]
        sources = _assigned_names(owner, expr.id)
        if sources:
            return [found for source in sources
                    for found in _candidates(module_name, tree, scope[:depth], ast.Name(id=source))]
    definition = next(node for node in tree.body if isinstance(node, _FunctionNode) and node.name == expr.id)
    return [(f"{module_name}:{expr.id}", definition)]


def _carries_pin(obj: Any) -> bool:
    r"""Whether ``obj`` is wrapped by ``ieee_fp32_matmul()`` used as a decorator."""
    function = getattr(obj, "__func__", obj)
    if not hasattr(function, "__wrapped__"):
        return False
    for cell in getattr(function, "__closure__", None) or ():
        try:
            content = cell.cell_contents
        except ValueError:
            continue
        if isinstance(content, contextlib.ContextDecorator) and getattr(content, "func", None) is _PIN_GENERATOR:
            return True
    return False


def _is_pin(expr: ast.expr, module: ModuleType) -> bool:
    if not isinstance(expr, ast.Call):
        return False
    func = expr.func
    if isinstance(func, ast.Name):
        return getattr(module, func.id, None) is ieee_fp32_matmul
    return isinstance(func, ast.Attribute) and func.attr == "ieee_fp32_matmul"


def _has_pin_block(definition: ast.AST, module: ModuleType) -> bool:
    return any(isinstance(node, (ast.With, ast.AsyncWith)) and any(_is_pin(item.context_expr, module)
                                                                    for item in node.items)
               for node in _local_nodes(definition))


def _package_function(module: ModuleType, name: str) -> bool:
    obj = getattr(module, name, None)
    return inspect.isfunction(obj) and str(getattr(obj, "__module__", "")).startswith("vfe3")


def _unpinned_work(definition: ast.AST, module: ModuleType, *, honor_pins: bool) -> List[str]:
    r"""GEMMs, package-function calls and ``self`` method calls of ``definition`` outside any pinned with block."""
    found: List[str] = []

    def visit(node: ast.AST, pinned: bool) -> None:
        if isinstance(node, (ast.With, ast.AsyncWith)):
            inside = pinned or (honor_pins and any(_is_pin(item.context_expr, module) for item in node.items))
            for statement in node.body:
                visit(statement, inside)
            return
        if isinstance(node, (*_FunctionNode, ast.ClassDef, ast.Lambda)):
            return
        if not pinned:
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                found.append(f"@ (line {node.lineno})")
            elif isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                if name in _GEMM_NAMES:
                    found.append(f"{name} (line {node.lineno})")
                elif isinstance(func, ast.Name) and _package_function(module, func.id) \
                        and func.id not in _OUTSIDE_PIN_HELPERS:
                    found.append(f"{func.id}() (line {node.lineno})")
                elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
                    found.append(f"self.{func.attr}() (line {node.lineno})")
        for child in ast.iter_child_nodes(node):
            visit(child, pinned)

    for statement in definition.body:
        visit(statement, False)
    return found


def _runtime_object(module: ModuleType, key: str) -> Optional[Any]:
    qualname = key.split(":", 1)[1]
    if "<locals>" in qualname:
        return None
    obj: Any = module
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _ast_decorated_pin(definition: ast.AST, module: ModuleType) -> bool:
    return any(_is_pin(decorator, module) for decorator in getattr(definition, "decorator_list", ()))


def test_every_checkpointed_callable_is_classified_and_recomputes_under_its_forward_policy(
        record_property: Any) -> None:
    found: Dict[str, Tuple[ModuleType, ast.AST, str]] = {}
    for module_name, module, tree, scope, call in _checkpoint_sites():
        assert call.args, f"{module_name}:{call.lineno} passes no callable to checkpoint"
        for key, definition in _candidates(module_name, tree, scope, call.args[0]):
            found.setdefault(key, (module, definition, f"{module_name}:{call.lineno}"))
    assert len(found) >= len(_CLASSIFIED), sorted(found)
    record_property("pending_minimal_agent", ",".join(sorted(set(found) & _PENDING_MINIMAL_AGENT)))
    unclassified = sorted(set(found) - set(_CLASSIFIED) - _PENDING_MINIMAL_AGENT)
    assert unclassified == [], f"classify these checkpointed callables: {unclassified}"
    stale = sorted(set(_CLASSIFIED) - set(found))
    assert stale == [], f"no checkpoint call names these any more: {stale}"
    problems: Dict[str, Any] = {}
    for key, kind in _CLASSIFIED.items():
        module, definition, site = found[key]
        if kind == "pinned":
            obj = _runtime_object(module, key)
            decorated = _carries_pin(obj) if obj is not None else _ast_decorated_pin(definition, module)
            if not decorated:
                outside = _unpinned_work(definition, module, honor_pins=True)
                if not _has_pin_block(definition, module) or outside:
                    problems[key] = (site, "not pinned", outside)
        elif kind == "elementwise":
            work = _unpinned_work(definition, module, honor_pins=False)
            if work:
                problems[key] = (site, "not elementwise", work)
        else:
            assert kind == "caller_policy", (key, kind)
    assert problems == {}


def test_the_pin_detector_tells_a_decorated_callable_from_a_bare_one() -> None:
    def bare(x: torch.Tensor) -> torch.Tensor:
        return x @ x

    pinned = ieee_fp32_matmul()(bare)
    restored = restored_fp32_matmul_precision()(bare)   # another @contextmanager decorator: not the pin
    assert _carries_pin(pinned) and not _carries_pin(bare)
    assert not _carries_pin(restored) and not _carries_pin(torch.no_grad()(bare))
    from vfe3.model.prototype_agents import _exact_em_tile
    from vfe3.relational_scoring import _mean_distance_tile
    assert _carries_pin(_exact_em_tile) and not _carries_pin(_mean_distance_tile)


_PROBES: Dict[str, Tuple[str, str]] = {
    "noisy_query_tile": ("vfe3.relational_scoring", "_noisy_query_tile"),
    # called in the decode CE's own scope right before each chunk checkpoint (or the Triton keep decision)
    "decode_ce": ("vfe3.model.prior_bank", "_decode_ce_should_checkpoint"),
}
_CALLER_POLICY_ROUTES: Dict[str, Tuple[Dict[str, Any], Tuple[str, ...]]] = {
    # a dense comparison covariance (no RoPE commutant projection), so the noisy-query cost takes its tiles
    "full_chunked_noisy_query": (dict(family="gaussian_full", decode_mode="full_chunked", rope_mode="none",
                                      relational_routing="noisy_query"), ("noisy_query_tile", "decode_ce")),
    "diagonal_chunked": (dict(decode_mode="diagonal_chunked"), ("decode_ce",)),
    "linear": (dict(decode_mode="linear"), ("decode_ce",)),
    "expected_likelihood_chunked": (dict(decode_mode="expected_likelihood_chunked"), ("decode_ce",)),
}


@pytest.mark.usefixtures("restore_model_process_globals")
@pytest.mark.parametrize("case", sorted(_CALLER_POLICY_ROUTES))
def test_the_caller_policy_callables_are_never_entered_from_a_pinned_scope(
        case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    overrides, probes = _CALLER_POLICY_ROUTES[case]
    seen: Dict[str, List[str]] = {probe: [] for probe in probes}
    for probe in probes:
        module_name, attribute = _PROBES[probe]
        module = importlib.import_module(module_name)
        original = getattr(module, attribute)

        def recorder(*args: Any, _original: Any = original, _probe: str = probe, **kwargs: Any) -> Any:
            seen[_probe].append(torch.get_float32_matmul_precision())
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, attribute, recorder)
    cfg = VFE3Config(**live_values(fp32_matmul_precision="high", **overrides))
    torch.manual_seed(0)
    model = VFEModel(cfg)
    assert torch.get_float32_matmul_precision() == "high"
    model.train()
    gen = torch.Generator().manual_seed(2)
    ids = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len), generator=gen)
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len), generator=gen)
    loss = model(ids, targets)[1]
    loss.backward()
    assert {probe: sorted(set(values)) for probe, values in seen.items()} == {probe: ["high"] for probe in probes}
