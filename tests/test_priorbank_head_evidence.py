"""Opt-in PriorBank-native head-evidence mixer state and ownership."""

from tests.profiles import test_config

import math

import pytest
import torch
import torch.nn.functional as F

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.model import prior_bank as prior_bank_mod
from vfe3.model.prior_bank import PriorBank, set_decode_av_precision
from vfe3.numerics import bounded_variance_from_log
from vfe3.train import _floor_lr_lambdas, build_optimizer


_MISSING = object()


def diagonal_head_reference(mu_q, var_q, mu_v, var_v, weights, dims):
    pieces = []
    start = 0
    for weight, dim in zip(weights, dims):
        stop = start + dim
        term = (
            (var_q[..., start:stop, None] +
             (mu_q[..., start:stop, None] - mu_v[start:stop]).square()) /
            var_v[start:stop]
            + var_v[start:stop].log()
            - var_q[..., start:stop, None].log()
            - 1.0
        ).sum(dim=-2) * 0.5
        pieces.append(weight * term)
        start = stop
    return torch.stack(pieces).sum(dim=0)


def _reference_bank(
    mode: str,
    dims: tuple[int, ...],
    *,
    enabled: bool,
    unigram: bool = False,
) -> PriorBank:
    torch.manual_seed(711)
    bank = PriorBank(
        5,
        sum(dims),
        9,
        decode_tau=1.7,
        diagonal_covariance=not mode.startswith("full"),
        family="gaussian_full" if mode.startswith("full") else "gaussian_diagonal",
        decode_mode=mode,
        decode_chunk_size=2,
        decode_ce_checkpoint="off",
        decode_unigram_prior=unigram,
        unigram_kappa=0.37,
        irrep_dims=list(dims),
        use_priorbank_head_evidence_mixer=enabled,
    ).double()
    with torch.no_grad():
        bank.mu_embed.copy_(torch.tensor([
            [-0.7, 0.2, 0.9, -0.1, 0.4],
            [0.3, -0.8, 0.1, 0.5, -0.2],
            [0.6, 0.4, -0.5, -0.3, 0.7],
            [-0.2, 0.9, 0.3, -0.6, 0.1],
            [0.8, -0.1, -0.4, 0.2, -0.9],
        ], dtype=torch.float64))
        bank.sigma_log_embed.copy_(torch.tensor([
            [0.9, 1.4, 0.7, 1.8, 1.1],
            [1.6, 0.8, 1.2, 0.6, 1.5],
            [1.1, 1.7, 0.9, 1.3, 0.8],
            [0.7, 1.0, 1.6, 1.1, 1.9],
            [1.8, 1.2, 0.8, 1.5, 0.7],
        ], dtype=torch.float64).log())
        if enabled:
            bank.head_evidence_logits.copy_(torch.tensor([0.8, -0.45], dtype=torch.float64))
    if unigram:
        bank.set_unigram_log_prior(torch.tensor([2, 11, 5, 3, 17]))
    return bank


def _diagonal_query() -> tuple[torch.Tensor, torch.Tensor]:
    mu_q = torch.tensor([
        [[-0.4, 0.7, -0.2, 0.5, 0.1], [0.6, -0.3, 0.8, -0.7, 0.2]],
        [[0.9, 0.1, -0.6, 0.4, -0.5], [-0.8, 0.5, 0.3, -0.2, 0.7]],
    ], dtype=torch.float64, requires_grad=True)
    var_q = torch.tensor([
        [[1.3, 0.6, 1.7, 0.8, 1.1], [0.7, 1.9, 0.9, 1.4, 0.5]],
        [[1.8, 1.0, 0.6, 1.2, 1.5], [0.9, 1.6, 1.3, 0.7, 1.9]],
    ], dtype=torch.float64, requires_grad=True)
    return mu_q, var_q


def _full_query() -> tuple[torch.Tensor, torch.Tensor]:
    mu_q, _ = _diagonal_query()
    factors = torch.tensor([
        [1.2, 0.0, 0.0, 0.0, 0.0],
        [0.2, 1.0, 0.0, 0.0, 0.0],
        [0.35, -0.15, 1.1, 0.0, 0.0],
        [-0.1, 0.25, 0.2, 0.9, 0.0],
        [0.3, 0.1, -0.2, 0.15, 1.05],
    ], dtype=torch.float64)
    base = factors @ factors.T + 0.2 * torch.eye(5, dtype=torch.float64)
    cov_q = torch.stack((base, base * 1.13, base * 0.87, base * 1.29)).reshape(2, 2, 5, 5)
    return mu_q, cov_q.clone().requires_grad_()


def _full_head_reference(
    mu_q: torch.Tensor,
    cov_q: torch.Tensor,
    mu_v: torch.Tensor,
    var_v: torch.Tensor,
    weights: torch.Tensor,
    dims: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    block_kls = []
    block_slices = []
    start = 0
    for dim in dims:
        stop = start + dim
        sl = slice(start, stop)
        block_slices.append(sl)
        marginal = cov_q[..., sl, sl]
        diff = mu_q[..., sl, None] - mu_v[sl]
        kl_h = 0.5 * (
            (torch.diagonal(marginal, dim1=-2, dim2=-1)[..., None] + diff.square())
            .div(var_v[sl])
            .sum(dim=-2)
            + var_v[sl].log().sum(dim=-2)
            - torch.linalg.slogdet(marginal).logabsdet[..., None]
            - dim
        )
        block_kls.append(kl_h)
        start = stop
    correction = 0.5 * (
        sum(torch.linalg.slogdet(cov_q[..., sl, sl]).logabsdet for sl in block_slices)
        - torch.linalg.slogdet(cov_q).logabsdet
    )
    weighted = sum(w_h * kl_h for w_h, kl_h in zip(weights, block_kls))
    return weighted + correction[..., None], weighted, correction


def _restore_registry_entry(registry: dict[str, object], name: str, previous: object) -> None:
    if previous is _MISSING:
        registry.pop(name, None)
    else:
        registry[name] = previous


def _enabled_cfg(**overrides: object) -> VFE3Config:
    values = {
        "vocab_size": 11,
        "embed_dim": 4,
        "n_heads": 2,
        "max_seq_len": 8,
        "n_layers": 1,
        "gauge_group": "block_glk",
        "use_priorbank_head_evidence_mixer": True,
        "family": "gaussian_diagonal",
        "divergence_family": "renyi",
        "renyi_order": 1.0,
        "decode_mode": "diagonal_chunked",
        "lambda_alpha_mode": "constant",
    }
    values.update(overrides)
    return test_config(**values)


def test_head_evidence_disabled_does_not_change_state_dict():
    assert test_config(use_priorbank_head_evidence_mixer=False).use_priorbank_head_evidence_mixer is False
    torch.manual_seed(7)
    base = PriorBank(11, 4, 8, irrep_dims=[2, 2], decode_mode="diagonal_chunked")
    torch.manual_seed(7)
    explicit = PriorBank(
        11,
        4,
        8,
        irrep_dims=[2, 2],
        decode_mode="diagonal_chunked",
        use_priorbank_head_evidence_mixer=False,
    )
    assert base.state_dict().keys() == explicit.state_dict().keys()
    for key in base.state_dict():
        torch.testing.assert_close(base.state_dict()[key], explicit.state_dict()[key], rtol=0, atol=0)
    assert not hasattr(base, "head_evidence_logits")


def test_zero_logits_produce_identity_head_and_coordinate_weights():
    pb = PriorBank(
        11,
        4,
        8,
        irrep_dims=[1, 3],
        decode_mode="diagonal_chunked",
        use_priorbank_head_evidence_mixer=True,
    )
    head, coord = pb.head_evidence_weights(dtype=torch.float64, device=torch.device("cpu"))
    torch.testing.assert_close(head, torch.ones(2, dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(coord, torch.ones(4, dtype=torch.float64), rtol=0, atol=0)


def test_head_evidence_parameter_is_enabled_only_and_starts_at_exact_zero():
    off = PriorBank(11, 4, 8, irrep_dims=[2, 2], decode_mode="diagonal_chunked")
    on = PriorBank(
        11,
        4,
        8,
        irrep_dims=[2, 2],
        decode_mode="diagonal_chunked",
        use_priorbank_head_evidence_mixer=True,
    )
    assert not hasattr(off, "head_evidence_logits")
    assert isinstance(on.head_evidence_logits, torch.nn.Parameter)
    torch.testing.assert_close(on.head_evidence_logits, torch.zeros(2), rtol=0, atol=0)


def test_disabled_head_evidence_preserves_caller_irrep_dims_list_identity():
    dims = [2, 2]
    pb = PriorBank(11, 4, 8, irrep_dims=dims, decode_mode="diagonal_chunked")
    assert pb.irrep_dims is dims
    assert pb.irrep_dims == [2, 2]


@pytest.mark.registry_mutation
def test_head_evidence_rejects_overridden_builtin_family_registration():
    from vfe3.families.base import _FAMILIES, register_family
    from vfe3.families.gaussian import DiagonalGaussian

    name = "gaussian_diagonal"
    previous = _FAMILIES.get(name, _MISSING)
    try:
        @register_family(name, override=True)
        class _ReplacementDiagonal(DiagonalGaussian):
            pass

        with pytest.raises(ValueError, match="built-in canonical registry identities"):
            _enabled_cfg()
    finally:
        _restore_registry_entry(_FAMILIES, name, previous)


@pytest.mark.registry_mutation
def test_head_evidence_rejects_overridden_builtin_renyi_functional():
    from vfe3.families.base import _FUNCTIONALS, register_functional

    name = "renyi"
    previous = _FUNCTIONALS.get(name, _MISSING)
    try:
        @register_functional(name, override=True)
        def _replacement_renyi(*args: object, **kwargs: object) -> torch.Tensor:
            return previous(*args, **kwargs)

        with pytest.raises(ValueError, match="built-in canonical registry identities"):
            _enabled_cfg()
    finally:
        _restore_registry_entry(_FUNCTIONALS, name, previous)


@pytest.mark.registry_mutation
def test_head_evidence_rejects_replaced_canonical_decoder_registration():
    from vfe3.model import prior_bank as prior_bank_mod

    name = "diagonal_chunked"
    previous = prior_bank_mod._DECODERS[name]
    try:
        prior_bank_mod.register_decode(
            name,
            supports_full=previous.supports_full,
            supports_chunked=previous.supports_chunked,
            fused_ce=previous.fused_ce,
            family_consistent=previous.family_consistent,
            covariance_kinds=previous.covariance_kinds,
            can_omit_base_mean=previous.can_omit_base_mean,
            can_omit_base_variance=previous.can_omit_base_variance,
            override=True,
        )(previous.callable)

        with pytest.raises(ValueError, match="built-in canonical registry identities"):
            _enabled_cfg()
    finally:
        prior_bank_mod._DECODERS[name] = previous


@pytest.mark.registry_mutation
def test_head_evidence_rejects_replaced_canonical_decoder_callable():
    from vfe3.model import prior_bank as prior_bank_mod

    name = "diagonal_chunked"
    previous = prior_bank_mod._DECODERS[name]
    try:
        @prior_bank_mod.register_decode(
            name,
            supports_full=previous.supports_full,
            supports_chunked=previous.supports_chunked,
            fused_ce=previous.fused_ce,
            family_consistent=previous.family_consistent,
            covariance_kinds=previous.covariance_kinds,
            can_omit_base_mean=previous.can_omit_base_mean,
            can_omit_base_variance=previous.can_omit_base_variance,
            override=True,
        )
        def _replacement_decoder(*args: object, **kwargs: object) -> torch.Tensor:
            return previous.callable(*args, **kwargs)

        with pytest.raises(ValueError, match="built-in canonical registry identities"):
            _enabled_cfg()
    finally:
        prior_bank_mod._DECODERS[name] = previous


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"decode_mode": "linear"}, "prior-bank decode_mode"),
        ({"gauge_group": "glk"}, "gauge_group=.*glk"),
        ({"renyi_order": 0.5}, "renyi_order=0.5"),
        ({"divergence_family": "squared_hellinger"}, "divergence_family=.*squared_hellinger"),
        ({"decode_mode": "family_chunked"}, "canonical KL decode"),
        ({"family": "gaussian_full", "decode_mode": "diagonal_chunked"}, "gaussian_full.*full.*full_chunked"),
    ],
)
def test_head_evidence_rejects_invalid_configurations(overrides: dict[str, object], match: str):
    with pytest.raises(ValueError, match=match):
        _enabled_cfg(**overrides)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"decode_mode": "linear"}, "prior-bank decode_mode"),
        ({"irrep_dims": [4]}, "at least two"),
        ({"irrep_dims": [True, 3]}, "plain positive integers"),
        ({"irrep_dims": [2.0, 2]}, "plain positive integers"),
        ({"irrep_dims": [1, 1]}, r"sum\(irrep_dims\)==K"),
        ({"renyi_order": 0.5}, "canonical KL functional"),
        ({"divergence_family": "squared_hellinger"}, "canonical KL functional"),
        ({"decode_mode": "family_chunked"}, "supported family/decode contract"),
        (
            {"family": "gaussian_full", "decode_mode": "diagonal_chunked"},
            "supported family/decode contract",
        ),
    ],
)
def test_direct_head_evidence_construction_rejects_invalid_contract(
    overrides: dict[str, object],
    match: str,
) -> None:
    """Direct PriorBank construction must enforce the same locally knowable mixer contract."""
    values = dict(
        vocab_size=11,
        K=4,
        n_gen=8,
        family="gaussian_diagonal",
        divergence_family="renyi",
        renyi_order=1.0,
        decode_mode="diagonal_chunked",
        irrep_dims=[2, 2],
        use_priorbank_head_evidence_mixer=True,
    )
    values.update(overrides)
    with pytest.raises(ValueError, match=match):
        PriorBank(**values)


def test_direct_head_evidence_accepts_precise_projected_rank_exception() -> None:
    """Only projected diagonal content may feed the built-in full scorer under this mixer."""
    bank = PriorBank(
        vocab_size=11,
        K=4,
        n_gen=8,
        family="gaussian_diagonal",
        divergence_family="renyi",
        renyi_order=1.0,
        decode_mode="full_chunked",
        encode_mode="canonical_content_projected",
        irrep_dims=[2, 2],
        use_priorbank_head_evidence_mixer=True,
    )
    assert tuple(bank.irrep_dims) == (2, 2)
    assert bank.head_evidence_logits.shape == (2,)


def _group_owning(optimizer, parameter):
    return next(
        group for group in optimizer.param_groups
        if any(candidate is parameter for candidate in group["params"])
    )


def test_optimizer_owns_head_evidence_logits_once_and_inherits_mean_lr():
    cfg = _enabled_cfg(m_p_mu_lr=0.0123)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    owned = [
        group for group in optimizer.param_groups
        if any(parameter is model.prior_bank.head_evidence_logits for parameter in group["params"])
    ]
    assert len(owned) == 1
    group = owned[0]
    assert group["lr"] == cfg.m_p_mu_lr
    assert group["weight_decay"] == 0.0
    assert group["role"] == "mu"


def test_both_mixers_receive_independent_explicit_learning_rates():
    cfg = _enabled_cfg(
        use_head_mixer=True,
        m_p_mu_lr=0.0123,
        m_head_evidence_lr=0.0011,
        m_head_mixer_lr=0.0022,
    )
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)

    evidence = _group_owning(optimizer, model.prior_bank.head_evidence_logits)
    legacy = _group_owning(optimizer, model.head_mixer.mixer_delta)

    assert evidence["lr"] == pytest.approx(0.0011)
    assert legacy["lr"] == pytest.approx(0.0022)
    assert evidence["weight_decay"] == 0.0
    assert legacy["weight_decay"] == pytest.approx(cfg.weight_decay)
    assert evidence["lr_aux_role"] == "head_evidence"
    assert legacy["lr_aux_role"] == "head_mixer"


def test_both_mixer_learning_rates_inherit_mean_lr_when_none():
    cfg = _enabled_cfg(
        use_head_mixer=True,
        m_p_mu_lr=0.0123,
        m_head_evidence_lr=None,
        m_head_mixer_lr=None,
    )
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)

    assert _group_owning(
        optimizer, model.prior_bank.head_evidence_logits)["lr"] == cfg.m_p_mu_lr
    assert _group_owning(
        optimizer, model.head_mixer.mixer_delta)["lr"] == cfg.m_p_mu_lr


def test_mixer_learning_rate_scheduler_preserves_zero_and_independent_floor():
    cfg = _enabled_cfg(
        use_head_mixer=True,
        m_head_evidence_lr=0.0,
        m_head_mixer_lr=0.0022,
        warmup_steps=2,
        max_steps=10,
        min_lr=1e-5,
        min_lr_frac=0.01,
    )
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _floor_lr_lambdas(base_lrs, cfg))

    for _ in range(cfg.max_steps + 5):
        optimizer.step()
        scheduler.step()

    evidence = _group_owning(optimizer, model.prior_bank.head_evidence_logits)
    legacy = _group_owning(optimizer, model.head_mixer.mixer_delta)
    assert evidence["lr"] == 0.0
    assert legacy["lr"] == pytest.approx(max(cfg.min_lr, cfg.min_lr_frac * 0.0022))


def test_head_mixer_and_head_evidence_are_legal_and_separately_owned():
    cfg = _enabled_cfg(use_head_mixer=True)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    evidence_groups = [
        group for group in optimizer.param_groups
        if any(parameter is model.prior_bank.head_evidence_logits for parameter in group["params"])
    ]
    mixer_parameters = tuple(model.head_mixer.parameters())
    mixer_groups = [
        group for group in optimizer.param_groups
        if any(parameter is candidate for parameter in group["params"] for candidate in mixer_parameters)
    ]
    assert model.head_mixer is not None
    assert len(evidence_groups) == 1
    assert len(mixer_groups) == 1
    assert evidence_groups[0] is not mixer_groups[0]


@pytest.mark.parametrize("mode", ["diagonal", "diagonal_chunked"])
def test_diagonal_reference_matches_nonuniform_unequal_head_blocks_and_gradients(mode: str):
    dims = (2, 3)
    bank = _reference_bank(mode, dims, enabled=True)
    mu_q, var_q = _diagonal_query()
    head, _ = bank.head_evidence_weights(dtype=mu_q.dtype, device=mu_q.device)
    mu_v = bank._decode_mu_table().T
    var_v = bounded_variance_from_log(bank._decode_sigma_log_table(), eps=bank.eps).T
    expected_divergence = diagonal_head_reference(mu_q, var_q, mu_v, var_v, head, dims)
    expected = -expected_divergence / bank._tau_eff()
    actual = bank.decode(mu_q, var_q)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    probe = torch.linspace(-0.9, 1.1, actual.numel(), dtype=actual.dtype).reshape_as(actual)
    leaves = (mu_q, var_q, bank.mu_embed, bank.sigma_log_embed, bank.head_evidence_logits)
    actual_grads = torch.autograd.grad((actual * probe).sum(), leaves, retain_graph=True)
    expected_grads = torch.autograd.grad((expected * probe).sum(), leaves)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-11, atol=2e-11)
    assert bank.head_evidence_logits.grad is None
    assert actual_grads[-1].abs().sum() > 0


@pytest.mark.parametrize("mode", ["full", "full_chunked"])
def test_full_reference_retains_unweighted_cross_block_correction_and_gradients(mode: str):
    dims = (2, 3)
    bank = _reference_bank(mode, dims, enabled=True)
    mu_q, cov_q = _full_query()
    head, _ = bank.head_evidence_weights(dtype=mu_q.dtype, device=mu_q.device)
    mu_v = bank._decode_mu_table().T
    var_v = bounded_variance_from_log(bank._decode_sigma_log_table(), eps=bank.eps).T
    expected_divergence, weighted_blocks, correction = _full_head_reference(
        mu_q, cov_q, mu_v, var_v, head, dims)
    assert correction.abs().min() > 0
    expected = -expected_divergence / bank._tau_eff()
    actual = bank.decode(mu_q, cov_q)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)

    probe = torch.linspace(1.1, -0.7, actual.numel(), dtype=actual.dtype).reshape_as(actual)
    leaves = (mu_q, cov_q, bank.mu_embed, bank.sigma_log_embed, bank.head_evidence_logits)
    actual_grads = torch.autograd.grad((actual * probe).sum(), leaves, retain_graph=True)
    expected_grads = torch.autograd.grad((expected * probe).sum(), leaves, retain_graph=True)
    weighted_only_logit_grad = torch.autograd.grad(
        ((-weighted_blocks / bank._tau_eff()) * probe).sum(), bank.head_evidence_logits)[0]
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=4e-10, atol=4e-10)
    torch.testing.assert_close(actual_grads[-1], weighted_only_logit_grad, rtol=2e-11, atol=2e-11)
    assert actual_grads[-1].abs().sum() > 0


@pytest.mark.parametrize("mode", ["diagonal", "diagonal_chunked", "full", "full_chunked"])
def test_initial_identity_is_bitwise_for_outputs_and_legacy_gradients(mode: str):
    dims = (2, 3)
    disabled = _reference_bank(mode, dims, enabled=False, unigram=True).float()
    enabled = _reference_bank(mode, dims, enabled=True, unigram=True).float()
    incompatible = enabled.load_state_dict(disabled.state_dict(), strict=False)
    assert incompatible.missing_keys == ["head_evidence_logits"]
    assert incompatible.unexpected_keys == []
    with torch.no_grad():
        enabled.head_evidence_logits.zero_()

    query_factory = _full_query if mode.startswith("full") else _diagonal_query
    mu_off_raw, sigma_off_raw = query_factory()
    mu_off = mu_off_raw.float().detach().requires_grad_()
    sigma_off = sigma_off_raw.float().detach().requires_grad_()
    mu_on = mu_off.detach().clone().requires_grad_()
    sigma_on = sigma_off.detach().clone().requires_grad_()
    logits_off = disabled.decode(mu_off, sigma_off)
    logits_on = enabled.decode(mu_on, sigma_on)
    assert torch.equal(logits_on, logits_off)

    probe = torch.linspace(-0.6, 0.8, logits_off.numel(), dtype=logits_off.dtype).reshape_as(logits_off)
    off_leaves = (mu_off, sigma_off, disabled.mu_embed, disabled.sigma_log_embed)
    on_leaves = (mu_on, sigma_on, enabled.mu_embed, enabled.sigma_log_embed)
    off_grads = torch.autograd.grad((logits_off * probe).sum(), off_leaves)
    on_grads = torch.autograd.grad(
        (logits_on * probe).sum(), on_leaves + (enabled.head_evidence_logits,))
    for off_grad, on_grad in zip(off_grads, on_grads[:-1]):
        assert torch.equal(on_grad, off_grad)
    assert on_grads[-1].abs().sum() > 0


@pytest.mark.parametrize("mode", ["diagonal_chunked", "full_chunked"])
def test_fused_head_evidence_ce_matches_dense_with_all_reduction_boundaries_and_gradients(
    mode: str,
):
    bank = _reference_bank(mode, (2, 3), enabled=True, unigram=True)
    bank.decode_ce_checkpoint = "on"
    with torch.no_grad():
        bank.decode_log_scale.fill_(0.31)
    mu_q, sigma_q = _full_query() if mode == "full_chunked" else _diagonal_query()
    targets = torch.tensor([[0, -100], [4, 2]])
    fused_fn = (
        bank.decode_ce_full_chunked
        if mode == "full_chunked"
        else bank.decode_ce_diagonal_chunked
    )
    fused = fused_fn(
        mu_q,
        sigma_q,
        targets,
        chunk_size=2,
        ignore_index=-100,
    )
    dense_logits = bank.decode(mu_q, sigma_q)
    dense = F.cross_entropy(
        dense_logits.reshape(-1, bank.vocab_size),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="mean",
    )
    torch.testing.assert_close(fused, dense, rtol=4e-11, atol=4e-11)

    leaves = (
        mu_q,
        sigma_q,
        bank.mu_embed,
        bank.sigma_log_embed,
        bank.decode_log_scale,
        bank.head_evidence_logits,
    )
    fused_grads = torch.autograd.grad(fused, leaves, retain_graph=True, allow_unused=True)
    dense_grads = torch.autograd.grad(dense, leaves)
    for fused_grad, dense_grad in zip(fused_grads, dense_grads):
        assert fused_grad is not None
        assert torch.isfinite(fused_grad).all()
        torch.testing.assert_close(fused_grad, dense_grad, rtol=3e-9, atol=3e-10)
    assert fused_grads[-2].abs().sum() > 0
    assert fused_grads[-1].abs().sum() > 0


def test_simultaneous_head_mixers_have_separate_gradients_and_diagnostics():
    torch.manual_seed(812)
    cfg = _enabled_cfg(use_head_mixer=True, decode_chunk_size=3)
    model = VFEModel(cfg)
    with torch.no_grad():
        model.prior_bank.head_evidence_logits.copy_(torch.tensor([0.7, -0.35]))
        model.head_mixer.mixer_delta.copy_(torch.tensor([[0.12, -0.04], [0.03, -0.08]]))
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    targets[0, 2] = -100

    _, loss, _ = model(tokens, targets)
    loss.backward()
    evidence_grad = model.prior_bank.head_evidence_logits.grad
    mixer_grads = [parameter.grad for parameter in model.head_mixer.mixer_deltas]
    assert evidence_grad is not None and torch.isfinite(evidence_grad).all()
    assert evidence_grad.abs().sum() > 0
    assert all(grad is not None and torch.isfinite(grad).all() for grad in mixer_grads)
    assert sum(grad.abs().sum() for grad in mixer_grads) > 0

    diagnostics = model.diagnostics(tokens)
    evidence_keys = {
        "head_evidence_weights",
        "head_evidence_entropy",
        "head_evidence_max_abs_drift",
    }
    assert evidence_keys <= diagnostics.keys()
    assert "head_mixer_drift" in diagnostics
    weights, _ = model.prior_bank.head_evidence_weights(
        dtype=model.prior_bank.head_evidence_logits.dtype,
        device=model.prior_bank.head_evidence_logits.device,
    )
    probabilities = weights / weights.numel()
    torch.testing.assert_close(
        torch.tensor(diagnostics["head_evidence_weights"]), weights.detach().cpu())
    assert diagnostics["head_evidence_entropy"] == pytest.approx(
        float(-(probabilities * probabilities.log()).sum().detach()))
    assert diagnostics["head_evidence_max_abs_drift"] == pytest.approx(
        float((weights - 1.0).abs().max().detach()))

    old_head_mixer_drift = diagnostics["head_mixer_drift"]
    with torch.no_grad():
        model.prior_bank.head_evidence_logits.add_(torch.tensor([-0.2, 0.5]))
    assert model.diagnostics(tokens)["head_mixer_drift"] == old_head_mixer_drift

    evidence_diagnostics = {
        key: model.diagnostics(tokens)[key] for key in evidence_keys
    }
    with torch.no_grad():
        model.head_mixer.mixer_delta.add_(0.17)
    after_mixer = model.diagnostics(tokens)
    assert {key: after_mixer[key] for key in evidence_keys} == evidence_diagnostics

    with torch.no_grad():
        model.prior_bank.head_evidence_logits.copy_(torch.tensor([1000.0, -1000.0]))
    collapsed = model.diagnostics(tokens)
    assert collapsed["head_evidence_entropy"] == pytest.approx(0.0)
    assert math.isfinite(collapsed["head_evidence_entropy"])


@pytest.mark.parametrize("mode", ["diagonal_chunked", "full_chunked"])
def test_fused_head_evidence_builds_weighted_query_lhs_once_across_chunks_and_checkpoint(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
):
    bank = _reference_bank(mode, (2, 3), enabled=True, unigram=True)
    bank.decode_ce_checkpoint = "on"
    mu_q, sigma_q = _full_query() if mode == "full_chunked" else _diagonal_query()
    targets = torch.tensor([[0, 3], [4, 2]])
    weighted_lhs_calls = 0
    real_decode_av_lhs = prior_bank_mod._decode_av_lhs

    def counting_decode_av_lhs(*args, **kwargs):
        nonlocal weighted_lhs_calls
        coord_delta = kwargs.get("coord_delta")
        if coord_delta is None and len(args) >= 3:
            coord_delta = args[2]
        if coord_delta is not None:
            weighted_lhs_calls += 1
        return real_decode_av_lhs(*args, **kwargs)

    monkeypatch.setattr(prior_bank_mod, "_decode_av_lhs", counting_decode_av_lhs)
    fused_fn = (
        bank.decode_ce_full_chunked
        if mode == "full_chunked"
        else bank.decode_ce_diagonal_chunked
    )
    loss = fused_fn(mu_q, sigma_q, targets, chunk_size=2)
    loss.backward()

    assert weighted_lhs_calls == 1


@pytest.mark.parametrize("mode", ["diagonal_chunked", "full_chunked"])
def test_fused_fp64_decode_av_island_divides_before_float32_cast_with_gradient_parity(
    mode: str,
):
    bank = _reference_bank(mode, (2, 3), enabled=True, unigram=True).float()
    bank.decode_ce_checkpoint = "on"
    with torch.no_grad():
        bank.decode_log_scale.fill_(0.31)
    mu_raw, sigma_raw = _full_query() if mode == "full_chunked" else _diagonal_query()
    mu_q = mu_raw.float().detach().requires_grad_()
    sigma_q = sigma_raw.float().detach().requires_grad_()
    targets = torch.tensor([[0, -100], [4, 2]])
    bank.decode_av_precision = "fp64"
    previous = set_decode_av_precision("fp64")
    try:
        fused_fn = (
            bank.decode_ce_full_chunked
            if mode == "full_chunked"
            else bank.decode_ce_diagonal_chunked
        )
        fused = fused_fn(
            mu_q,
            sigma_q,
            targets,
            chunk_size=2,
        )

        weights, _ = bank.head_evidence_weights(dtype=torch.float32, device=mu_q.device)
        mu_v = bank._decode_mu_table().T.double()
        var_v = bounded_variance_from_log(
            bank._decode_sigma_log_table(), eps=bank.eps).T.double()
        if mode == "full_chunked":
            divergence, _, _ = _full_head_reference(
                mu_q.double(), sigma_q.double(), mu_v, var_v, weights.double(), (2, 3))
        else:
            divergence = diagonal_head_reference(
                mu_q.double(), sigma_q.double(), mu_v, var_v, weights.double(), (2, 3))
        oracle_logits = -divergence / bank._tau_eff().double()
        oracle_logits = oracle_logits + bank._unigram_bias()
        dense_logits = bank.decode(mu_q, sigma_q)
        torch.testing.assert_close(dense_logits, oracle_logits, rtol=0.0, atol=2.5e-7)
        oracle = F.cross_entropy(
            oracle_logits.reshape(-1, bank.vocab_size),
            targets.reshape(-1),
            ignore_index=-100,
        )
        leaves = (
            mu_q,
            sigma_q,
            bank.mu_embed,
            bank.sigma_log_embed,
            bank.decode_log_scale,
            bank.head_evidence_logits,
        )
        fused_grads = torch.autograd.grad(fused, leaves, retain_graph=True)
        oracle_grads = torch.autograd.grad(oracle, leaves)
        torch.testing.assert_close(fused, oracle, rtol=0.0, atol=2e-8)
        for fused_grad, oracle_grad in zip(fused_grads, oracle_grads):
            torch.testing.assert_close(fused_grad, oracle_grad, rtol=2e-6, atol=2e-7)
    finally:
        set_decode_av_precision(previous)


def test_dense_full_fp64_head_evidence_casts_only_completed_logits_to_query_dtype():
    bank = _reference_bank("full", (2, 3), enabled=True, unigram=True).float()
    mu_raw, sigma_raw = _full_query()
    mu_q = mu_raw.float().detach().requires_grad_()
    sigma_q = sigma_raw.float().detach().requires_grad_()
    bank.decode_av_precision = "fp64"
    previous = set_decode_av_precision("fp64")
    try:
        logits = bank.decode(mu_q, sigma_q)
        assert logits.dtype == mu_q.dtype
        assert torch.isfinite(logits).all()
    finally:
        set_decode_av_precision(previous)
