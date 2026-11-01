r"""``strict_once_differentiable``: the fused Functions' refusal of a double backward, for every upstream leaf.

``torch.autograd.function.once_differentiable`` attaches its ``DelayedError`` node only when an incoming gradient
requires grad, and hangs it off detached copies of the backward's outputs. A loss linear in the Function's output
passes a constant cotangent, so no node is attached; and when one is, ``torch.autograd.grad(..., inputs=...)`` prunes
it as unreachable from the requested leaf. Either way a second derivative toward a leaf that reaches the Function
through a nonlinear map (``mu_q`` through ``lhs = [sq + mc_q^2, -2 mc_q]`` in the decoder) came back without the
Function's own second-order term and without an error, 50 to 92 percent wrong (audit 2026-10-31 triton-kernels-2).

This decorator attaches the node whenever the backward runs in grad mode (``create_graph=True``), and feeds it the
outputs minus an exact zero that depends on every saved input and incoming gradient that requires grad (the sum of an
empty slice of each), so a second differentiation toward any leaf upstream of those reaches the node and raises. The
first-order values are the plain backward's bitwise (:math:`x - 0 = x`, signed zeros included), and a plain backward
(grad mode off) is untouched. A Function that does not save one of its differentiable inputs links none of that
input's upstream leaves, so each decorated Function saves all of them.
"""
import functools
from typing import Any, Callable, List

import torch
from torch._C import _functions

_MESSAGE = b"trying to differentiate twice a function that was marked with @once_differentiable"


def strict_once_differentiable(backward: Callable[..., Any]) -> Callable[..., Any]:
    r"""Decorate a custom Function's ``backward`` so that any double backward through it raises (module docstring)."""

    @functools.wraps(backward)
    def wrapper(ctx: Any, *grads: Any) -> Any:
        with torch.no_grad():
            outputs = backward(ctx, *grads)
        if not torch.is_grad_enabled():
            return outputs
        single = not isinstance(outputs, tuple)
        outs = (outputs,) if single else outputs
        # Saved inputs (not this Function's own saved outputs) and incoming gradients with a graph of their own.
        linked: List[torch.Tensor] = [t for t in ctx.saved_tensors if t.requires_grad and t.grad_fn is not ctx]
        linked += [g for g in grads if isinstance(g, torch.Tensor) and g.requires_grad]
        if linked:
            zero = sum(t.reshape(-1)[:0].sum() for t in linked)                 # exactly 0, a graph to each one
            fake = [None if v is None else v.detach() - zero.to(v.dtype) for v in outs]
        else:
            fake = [None if v is None else v.detach().requires_grad_() for v in outs]
        result = _functions.DelayedError(_MESSAGE, len(outs))(*fake)
        if single:
            return result[0] if isinstance(result, tuple) else result
        return result if isinstance(result, tuple) else (result,)

    return wrapper
