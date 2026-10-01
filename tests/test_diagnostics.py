r"""One-forward diagnostic snapshots and replay-free evaluation consumers."""

from tests.profiles import test_config
from dataclasses import FrozenInstanceError
import math

import pytest
import torch

from vfe3 import metrics
from vfe3.model.model import VFEModel
from vfe3.train import _val_diagnostics
from vfe3.viz import extract


def _model(**overrides) -> VFEModel:
    base = dict(
        vocab_size=12,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=2,
        n_e_steps=2,
        pos_phi="none",
        pos_rotation="none",
    )
    base.update(overrides)
    torch.manual_seed(17)
    model = VFEModel(test_config(**base))
    model.eval()
    return model


def _tokens() -> torch.Tensor:
    return torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])


def _assert_scalar_dict_equal(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for key in actual:
        if actual[key] is None or expected[key] is None:
            assert actual[key] is expected[key], key
            continue
        if isinstance(actual[key], list):
            assert len(actual[key]) == len(expected[key]), key
            for actual_item, expected_item in zip(actual[key], expected[key]):
                if isinstance(actual_item, float) and math.isnan(actual_item):
                    assert math.isnan(expected_item), key
                else:
                    assert math.isclose(actual_item, expected_item,
                                        rel_tol=1e-5, abs_tol=1e-6), key
        elif isinstance(actual[key], float) and math.isnan(actual[key]):
            # A metric both paths evaluate as NaN (e.g. a 0/0 flatness residual on flat transport)
            # agrees on being undefined; isclose would call that unequal.
            assert math.isnan(expected[key]), key
        else:
            assert math.isclose(actual[key], expected[key], rel_tol=1e-5, abs_tol=1e-6), key


def test_diagnostics_registry_dispatch_substitutes_attention_entropy() -> None:
    r"""PB-07: ``diagnostics()`` routes the row-entropy ``attention_entropy`` metric through the metric
    registry (aliased to ``attn_entropy``), so a registered override reaches it there -- while the
    free-energy component named ``attention_entropy`` (returned by ``free_energy_terms``) is untouched,
    proving the alias prevents the row entropy from overwriting the component of the same name."""
    model = _model(embed_dim=2, n_heads=1, n_layers=1)
    tokens = _tokens()
    baseline = model.diagnostics(tokens)
    sentinel = -12345.0
    original = metrics.get_metric("attention_entropy")
    try:
        metrics.register_metric("attention_entropy", override=True)(lambda **kw: sentinel)
        d = model.diagnostics(tokens)
    finally:
        metrics.register_metric("attention_entropy", override=True)(original)
    assert metrics.get_metric("attention_entropy") is original           # restore succeeded
    assert d["attn_entropy"] == sentinel                                 # row entropy routed via registry
    assert d["attention_entropy"] != sentinel                           # free-energy component distinct
    assert d["attention_entropy"] == pytest.approx(baseline["attention_entropy"], rel=1e-6, abs=1e-9)


def test_effective_rank_registry_dispatch_uses_explicit_diagonal_flag() -> None:
    r"""PB-07: the registered ``effective_rank`` wrapper REQUIRES an explicit ``diagonal`` flag and
    threads it to ``_spectrum``. A diagonal (N, K) variance table with N == K is square in its last two
    axes, so shape auto-inference misclassifies it as a full covariance and eigvalsh a variance vector;
    the explicit flag keeps the diagonal interpretation."""
    torch.manual_seed(0)
    K = 3
    sigma = torch.rand(K, K) + 0.5                                       # N == K diagonal variance table
    got = metrics.get_metric("effective_rank")(sigma=sigma, diagonal=True)
    ref = float(metrics.effective_rank(sigma).mean())                    # variances ARE the spectrum
    eig = torch.linalg.eigvalsh(0.5 * (sigma + sigma.transpose(-1, -2)))
    wrong = float(metrics.effective_rank(eig).mean())                    # misclassified full-cov reading
    assert got == pytest.approx(ref, rel=1e-6, abs=1e-9)                 # explicit flag -> diagonal spectrum
    assert abs(ref - wrong) > 1e-3                                       # the two readings genuinely differ


def test_eval_diagnostics_builds_one_snapshot(monkeypatch) -> None:
    model = _model()
    tokens = _tokens()
    targets = torch.roll(tokens, shifts=-1, dims=1)
    calls = 0
    training_modes = []
    real_forward_beliefs = model.forward_beliefs

    def counted_forward_beliefs(*args, **kwargs):
        nonlocal calls
        calls += 1
        # forward_beliefs no longer takes a `training` kwarg; the module's own mode is what
        # decides eval semantics now, so record that instead.
        training_modes.append(model.training)
        return real_forward_beliefs(*args, **kwargs)

    monkeypatch.setattr(model, "forward_beliefs", counted_forward_beliefs)
    diagnostics = _val_diagnostics(model, [(tokens, targets)], torch.device("cpu"))

    assert calls == 1
    assert training_modes == [False]
    assert math.isfinite(diagnostics.metrics["val_inner_alignment_energy_total"])
    # The "val_free_energy_total" alias is retired (audit 2026-07-25 F7): it duplicated
    # val_inner_alignment_energy_total under a name that claimed to include -E_q[log p(o|x)],
    # which that value never carried. Nothing to compare it against now.
    assert "val_free_energy_total" not in diagnostics.metrics
    assert math.isfinite(diagnostics.metrics["estep_f_drop"])
    assert math.isfinite(diagnostics.metrics["pos_loss_ratio"])


def test_snapshot_and_independent_diagnostics_are_value_equal() -> None:
    model = _model()
    tokens = _tokens()
    snapshot = model.build_diagnostic_snapshot(tokens)

    with pytest.raises(FrozenInstanceError):
        snapshot.logits = torch.empty_like(snapshot.logits)

    torch.manual_seed(29)
    expected = model.diagnostics(tokens)
    torch.manual_seed(29)
    actual = model.diagnostics(tokens, snapshot=snapshot)
    _assert_scalar_dict_equal(actual, expected)

    assert torch.allclose(
        model.attention_maps(tokens, snapshot=snapshot),
        model.attention_maps(tokens),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.manual_seed(31)
    per_layer_actual = model.diagnostics_per_layer(tokens, snapshot=snapshot)
    torch.manual_seed(31)
    per_layer_expected = model.diagnostics_per_layer(tokens)
    _assert_scalar_dict_equal(per_layer_actual, per_layer_expected)
    assert torch.allclose(snapshot.logits, model(tokens), atol=1e-6, rtol=1e-6)


def test_attention_and_trace_reuse_snapshot_without_forward_replay(monkeypatch) -> None:
    import vfe3.model.block as block_module
    import vfe3.model.model as model_module
    import vfe3.model.stack as stack_module

    model = _model()
    tokens = _tokens()
    snapshot = model.build_diagnostic_snapshot(tokens)

    def replay_forbidden(*args, **kwargs):
        raise AssertionError("snapshot consumer replayed model inference")

    monkeypatch.setattr(model, "forward_beliefs", replay_forbidden)
    monkeypatch.setattr(model.prior_bank, "encode", replay_forbidden)
    monkeypatch.setattr(model_module, "vfe_stack", replay_forbidden)
    monkeypatch.setattr(model_module, "vfe_block", replay_forbidden)
    monkeypatch.setattr(stack_module, "vfe_stack", replay_forbidden)
    monkeypatch.setattr(block_module, "e_step", replay_forbidden)
    monkeypatch.setattr(extract, "_encode_one", replay_forbidden)
    monkeypatch.setattr(extract, "e_step_iteration", replay_forbidden)
    monkeypatch.setattr(extract, "vfe_block", replay_forbidden)

    assert model.diagnostics(tokens, snapshot=snapshot)
    assert model.attention_maps(tokens, snapshot=snapshot).shape == (2, 2, 4, 4)
    assert model.diagnostics_per_layer(tokens, snapshot=snapshot)
    assert extract.e_step_belief_trace(model, tokens, snapshot=snapshot)["mu"].shape[0] == 3
    diagonal_flags = []
    real_spd_geodesic_distance = metrics.spd_geodesic_distance

    def record_spd_covariance_family(
        sigma_a: torch.Tensor,
        sigma_b: torch.Tensor,

        *,
        diagonal: bool | None = None,
        eps:      float = 1e-12,
    ) -> torch.Tensor:
        diagonal_flags.append(diagonal)
        return real_spd_geodesic_distance(
            sigma_a,
            sigma_b,
            diagonal=diagonal,
            eps=eps,
        )

    monkeypatch.setattr(metrics, "spd_geodesic_distance", record_spd_covariance_family)
    assert extract.across_layer_belief_trace(model, tokens, snapshot=snapshot)["mu"].shape[0] == 2
    assert diagonal_flags == [True]
    assert extract.numerical_health(model, tokens, snapshot=snapshot)
    assert extract.converged_state(model, tokens, snapshot=snapshot)
