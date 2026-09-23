"""Projected canonical-content materialization and realized-frame contracts."""

import os

import pytest
import torch
import torch.nn.functional as F

from vfe3.config import VFE3Config
from vfe3.geometry.lie_ops import CompactBlockElement
from vfe3.geometry.transport import CompactFactoredTransport, FactoredTransport
from vfe3.model.canonical_content import (
    CanonicalFrameContext,
    project_canonical_diagonal,
    pullback_diagonal_query,
)
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import PriorBank, get_decode_registration, get_encode

TASK6_DEVICE = torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))


def _projected_cfg(**overrides: object) -> VFE3Config:
    values = dict(
        vocab_size=11,
        embed_dim=4,
        n_heads=2,
        max_seq_len=5,
        batch_size=2,
        n_layers=1,
        n_e_steps=1,
        e_step_update="gradient",
        oracle_unroll_grad=True,
        family="gaussian_diagonal",
        transport_mode="flat",
        e_phi_lr=0.0,
        decode_mode="full_chunked",
        encode_mode="canonical_content_projected",
        lambda_alpha_mode="constant",
        max_steps=1,
    )
    values.update(overrides)
    return VFE3Config(
        _allow_legacy_dense_decode=values.get("decode_mode") == "full",
        **values,
    )


def _projected_bank(*, untie_decode_bank: bool = False) -> PriorBank:
    torch.manual_seed(71)
    return PriorBank(
        vocab_size=9,
        K=4,
        n_gen=8,
        family="gaussian_diagonal",
        encode_mode="canonical_content_projected",
        decode_mode="full",
        untie_decode_bank=untie_decode_bank,
    )


def _noncommuting_frames() -> tuple[torch.Tensor, torch.Tensor]:
    token_frame = torch.tensor(
        [[1.5, 0.4], [-0.2, 0.9]], dtype=torch.float64)
    positional_frame = torch.tensor(
        [[1.0, -0.3], [0.25, 1.2]], dtype=torch.float64)
    return token_frame, positional_frame


def test_pushforward_matches_hand_reference_for_batched_nonorthogonal_frames() -> None:
    """Changing the covariance pushforward from squared frame entries breaks this reference."""
    token_frame, positional_frame = _noncommuting_frames()
    frame = (token_frame @ positional_frame).expand(2, 3, 2, 2).clone()
    mu_c = torch.tensor(
        [
            [[0.2, -0.7], [1.1, 0.3], [-0.4, 0.8]],
            [[-0.6, 0.5], [0.9, -1.2], [0.1, 0.4]],
        ],
        dtype=torch.float64,
    )
    var_c = torch.tensor(
        [
            [[0.5, 1.3], [0.7, 0.4], [1.5, 0.2]],
            [[0.8, 0.6], [0.3, 1.7], [1.1, 0.9]],
        ],
        dtype=torch.float64,
    )

    got_mu, got_var = project_canonical_diagonal(mu_c, var_c, frame)

    expected_mu = torch.einsum("...ij,...j->...i", frame, mu_c)
    expected_var = torch.einsum("...ij,...j->...i", frame.square(), var_c)
    assert torch.equal(got_mu, expected_mu)
    assert torch.equal(got_var, expected_var)
    assert got_mu.dtype == mu_c.dtype and got_mu.device == mu_c.device
    assert got_var.dtype == var_c.dtype and got_var.device == var_c.device


def test_positional_order_is_token_frame_times_right_positional_frame() -> None:
    """Reversing the configured right-factor multiplication changes materialized content."""
    token_frame, positional_frame = _noncommuting_frames()
    mu_c = torch.tensor([0.6, -0.35], dtype=torch.float64)
    var_c = torch.tensor([0.45, 1.25], dtype=torch.float64)
    right_order = token_frame @ positional_frame
    reversed_order = positional_frame @ token_frame

    got_mu, got_var = project_canonical_diagonal(mu_c, var_c, right_order)
    wrong_mu = reversed_order @ mu_c
    wrong_var = reversed_order.square() @ var_c

    assert torch.equal(got_mu, right_order @ mu_c)
    assert torch.equal(got_var, right_order.square() @ var_c)
    assert not torch.allclose(got_mu, wrong_mu)
    assert not torch.allclose(got_var, wrong_var)


def test_pullback_matches_hand_reference_and_mean_inverse_consistency() -> None:
    """Using a transpose or forward frame in pullback breaks canonical query coordinates."""
    token_frame, positional_frame = _noncommuting_frames()
    frame = (token_frame @ positional_frame).expand(2, 3, 2, 2).clone()
    frame_inv = torch.linalg.inv(frame)
    mu_c = torch.tensor(
        [
            [[0.4, -0.1], [0.8, 0.7], [-0.3, 1.0]],
            [[-0.5, 0.6], [0.2, -0.9], [1.3, 0.1]],
        ],
        dtype=torch.float64,
    )
    mu_q = torch.einsum("...ij,...j->...i", frame, mu_c)
    var_q = torch.tensor(
        [
            [[0.9, 0.4], [1.2, 0.8], [0.5, 1.1]],
            [[0.6, 1.4], [0.7, 0.3], [1.5, 0.2]],
        ],
        dtype=torch.float64,
    )

    got_mu, got_cov = pullback_diagonal_query(mu_q, var_q, frame_inv)

    expected_mu = torch.einsum("...ij,...j->...i", frame_inv, mu_q)
    expected_cov = frame_inv @ torch.diag_embed(var_q) @ frame_inv.transpose(-1, -2)
    assert torch.allclose(got_mu, mu_c, atol=1e-14, rtol=1e-14)
    assert torch.equal(got_mu, expected_mu)
    assert torch.equal(got_cov, expected_cov)
    assert torch.allclose(
        frame @ frame_inv,
        torch.eye(2, dtype=torch.float64).expand_as(frame),
        atol=1e-14,
        rtol=1e-14,
    )


def test_pushforward_and_pullback_pass_float64_gradcheck() -> None:
    """Detaching or casting any projected moment or frame factor breaks differentiability."""
    generator = torch.Generator().manual_seed(51)
    mu_c = torch.randn(1, 2, 2, dtype=torch.float64, generator=generator, requires_grad=True)
    var_c = (torch.rand(1, 2, 2, dtype=torch.float64, generator=generator) + 0.7).requires_grad_()
    frame = (
        torch.eye(2, dtype=torch.float64).expand(1, 2, 2, 2).clone()
        + 0.12 * torch.randn(1, 2, 2, 2, dtype=torch.float64, generator=generator)
    ).requires_grad_()
    assert torch.autograd.gradcheck(
        project_canonical_diagonal,
        (mu_c, var_c, frame),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )

    mu_q = torch.randn(1, 2, 2, dtype=torch.float64, generator=generator, requires_grad=True)
    var_q = (torch.rand(1, 2, 2, dtype=torch.float64, generator=generator) + 0.8).requires_grad_()
    frame_inv = torch.linalg.inv(frame.detach()).requires_grad_()
    assert torch.autograd.gradcheck(
        pullback_diagonal_query,
        (mu_q, var_q, frame_inv),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )


def test_coordinate_transforms_preserve_float32_under_bf16_autocast() -> None:
    """Autocast may not change the public moment/frame coordinate dtype."""
    generator = torch.Generator().manual_seed(83)
    mu = torch.randn(2, 3, 4, generator=generator)
    var = torch.rand(2, 3, 4, generator=generator) + 0.4
    frame = (
        torch.eye(4).expand(2, 3, 4, 4).clone()
        + 0.08 * torch.randn(2, 3, 4, 4, generator=generator)
    )
    frame_inv = torch.linalg.inv(frame)
    expected_push = project_canonical_diagonal(mu, var, frame)
    expected_pull = pullback_diagonal_query(mu, var, frame_inv)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        got_push = project_canonical_diagonal(mu, var, frame)
        got_pull = pullback_diagonal_query(mu, var, frame_inv)

    for got, expected in zip((*got_push, *got_pull), (*expected_push, *expected_pull)):
        assert got.dtype == torch.float32
        assert torch.equal(got, expected)


@pytest.mark.parametrize(
    ("which", "match"),
    [
        ("moment", "identical shapes"),
        ("frame", "square"),
        ("leading", "leading shape"),
        ("dtype", "same dtype"),
    ],
)
def test_pushforward_rejects_incompatible_shapes_and_dtypes(which: str, match: str) -> None:
    """Silent broadcasting or promotion would hide a mismatched realized frame."""
    mu = torch.zeros(2, 3, dtype=torch.float64)
    var = torch.ones_like(mu)
    frame = torch.eye(3, dtype=torch.float64).expand(2, 3, 3).clone()
    if which == "moment":
        var = torch.ones(2, 4, dtype=torch.float64)
    elif which == "frame":
        frame = torch.zeros(2, 3, 4, dtype=torch.float64)
    elif which == "leading":
        frame = torch.eye(3, dtype=torch.float64).expand(1, 3, 3).clone()
    else:
        frame = frame.float()

    with pytest.raises(ValueError, match=match):
        project_canonical_diagonal(mu, var, frame)


def test_frame_context_rejects_noninverse_layout_mismatch() -> None:
    """A context with differently shaped factors cannot identify one realized vertex frame."""
    with pytest.raises(ValueError, match="identical shapes"):
        CanonicalFrameContext(
            forward=torch.eye(2, dtype=torch.float64).expand(2, 2, 2).clone(),
            inverse=torch.eye(2, dtype=torch.float64).expand(1, 2, 2).clone(),
        )


def test_projected_registry_returns_existing_canonical_table_coordinates() -> None:
    """A separate projected embedding table would duplicate parameters and alter initialization."""
    get_encode("canonical_content_projected")
    projected = _projected_bank()
    token_ids = torch.tensor([[0, 4, 2], [7, 1, 8]])

    got = projected.encode(token_ids)

    assert torch.equal(got.mu, projected.mu_embed[token_ids])
    assert torch.equal(got.sigma, torch.exp(projected.sigma_log_embed[token_ids]))
    assert torch.equal(got.phi, projected.phi_embed[token_ids])
    assert not any("canonical" in name for name in projected._parameters)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("family", "gaussian_full"),
        ("e_phi_lr", 0.01),
        ("decode_mode", "linear"),
        ("decode_mode", "diagonal_chunked"),
    ],
)
def test_projected_config_rejects_each_incompatible_cross_field_pair(
    field: str,
    value: object,
) -> None:
    """Relaxing any projected-mode gate would invalidate its shared-frame decode contract."""
    with pytest.raises(ValueError, match="canonical_content_projected"):
        _projected_cfg(**{field: value})


@pytest.mark.parametrize("decode_mode", ["full", "full_chunked"])
@pytest.mark.parametrize("pos_phi", ["learned", "frozen"])
@pytest.mark.parametrize("untie_decode_bank", [False, True])
def test_projected_config_accepts_supported_decode_position_and_tie_variants(
    decode_mode: str,
    pos_phi: str,
    untie_decode_bank: bool,
) -> None:
    """Narrowing a legal projected variant would block current-frame rematerialization."""
    cfg = _projected_cfg(
        decode_mode=decode_mode,
        pos_phi=pos_phi,
        untie_decode_bank=untie_decode_bank,
        m_phi_lr=0.02,
    )
    assert cfg.decode_mode == decode_mode
    assert cfg.pos_phi == pos_phi
    assert cfg.untie_decode_bank is untie_decode_bank


@pytest.mark.parametrize("decode_mode", ["full", "full_chunked"])
def test_projected_config_accepts_full_head_evidence_after_pullback(
    decode_mode: str,
) -> None:
    """The projected rank exception must reach the built-in full evidence decoder only."""
    cfg = _projected_cfg(
        decode_mode=decode_mode,
        use_priorbank_head_evidence_mixer=True,
    )
    assert cfg.encode_mode == "canonical_content_projected"
    assert cfg.family == "gaussian_diagonal"
    assert cfg.decode_mode == decode_mode
    assert cfg.use_priorbank_head_evidence_mixer is True


@pytest.mark.registry_mutation
@pytest.mark.parametrize("mixer_enabled", [False, True])
def test_projected_construction_rejects_overridden_gaussian_full_family(
    mixer_enabled: bool,
) -> None:
    """Projected analytic full decoding must not resolve a same-name custom full family."""
    from vfe3.families.base import _FAMILIES, register_family
    from vfe3.families.gaussian import FullGaussian

    name = "gaussian_full"
    previous = _FAMILIES[name]
    try:
        @register_family(name, override=True)
        class _ReplacementFullGaussian(FullGaussian):
            pass

        with pytest.raises(ValueError, match=r"canonical_content_projected.*gaussian_full"):
            _projected_cfg(use_priorbank_head_evidence_mixer=mixer_enabled)
        with pytest.raises(ValueError, match=r"canonical_content_projected.*gaussian_full"):
            PriorBank(
                vocab_size=9,
                K=4,
                n_gen=8,
                family="gaussian_diagonal",
                encode_mode="canonical_content_projected",
                decode_mode="full_chunked",
                irrep_dims=[2, 2],
                use_priorbank_head_evidence_mixer=mixer_enabled,
            )
    finally:
        _FAMILIES[name] = previous
    assert _FAMILIES[name] is previous


@pytest.mark.registry_mutation
@pytest.mark.parametrize("construction_seam", ["config", "prior_bank"])
def test_projected_construction_rejects_overridden_diagonal_family_without_mixer(
    construction_seam: str,
) -> None:
    """Projected materialization must not resolve a same-name custom table family."""
    from vfe3.families.base import _FAMILIES, register_family
    from vfe3.families.gaussian import DiagonalGaussian

    name = "gaussian_diagonal"
    previous = _FAMILIES[name]
    try:
        @register_family(name, override=True)
        class _ReplacementDiagonalGaussian(DiagonalGaussian):
            pass

        with pytest.raises(ValueError, match=r"canonical_content_projected.*gaussian_diagonal"):
            if construction_seam == "config":
                _projected_cfg(use_priorbank_head_evidence_mixer=False)
            else:
                _projected_bank()
    finally:
        _FAMILIES[name] = previous
    assert _FAMILIES[name] is previous


@pytest.mark.registry_mutation
@pytest.mark.parametrize("override_kind", ["registration", "callable_map"])
def test_projected_construction_rejects_overridden_encoder_identity(
    override_kind: str,
) -> None:
    """The projected name alone cannot authorize another encoder implementation."""
    from vfe3.model import prior_bank as prior_bank_mod

    name = "canonical_content_projected"
    previous_registration = prior_bank_mod._ENCODER_REGISTRATIONS[name]
    previous_callable = prior_bank_mod._ENCODERS[name]

    def replacement_encoder(pb, token_ids):
        return previous_callable(pb, token_ids)

    try:
        if override_kind == "registration":
            prior_bank_mod.register_encode(
                name,
                override=True,
                can_omit_base_mean=previous_registration.can_omit_base_mean,
                can_omit_base_variance=previous_registration.can_omit_base_variance,
            )(previous_callable)
        else:
            prior_bank_mod._ENCODERS[name] = replacement_encoder

        with pytest.raises(ValueError, match=r"canonical_content_projected.*built-in encoder"):
            _projected_cfg()
        with pytest.raises(ValueError, match=r"canonical_content_projected.*built-in encoder"):
            _projected_bank()
    finally:
        prior_bank_mod._ENCODERS[name] = previous_callable
        prior_bank_mod._ENCODER_REGISTRATIONS[name] = previous_registration
    assert prior_bank_mod._ENCODERS[name] is previous_callable
    assert prior_bank_mod._ENCODER_REGISTRATIONS[name] is previous_registration


@pytest.mark.registry_mutation
@pytest.mark.parametrize("decode_mode", ["full", "full_chunked"])
@pytest.mark.parametrize("override_kind", ["registration", "callable"])
def test_projected_construction_rejects_overridden_full_decoder_before_forward(
    decode_mode: str,
    override_kind: str,
) -> None:
    """A projected model must never reach a custom decoder lacking its frame contract."""
    from vfe3.model import prior_bank as prior_bank_mod

    previous = prior_bank_mod._DECODERS[decode_mode]

    def replacement_decode(pb, mu_q, sigma_q, tau_eff):
        return previous.callable(pb, mu_q, sigma_q, tau_eff)

    def replacement_fused_ce(
        pb,
        mu_q,
        sigma_q,
        targets,
        *,
        tau=None,
        chunk_size=None,
        ignore_index=-100,
    ):
        assert previous.fused_ce is not None
        return previous.fused_ce(
            pb,
            mu_q,
            sigma_q,
            targets,
            tau=tau,
            chunk_size=chunk_size,
            ignore_index=ignore_index,
        )

    decode_callable = previous.callable
    fused_callable = previous.fused_ce
    if override_kind == "callable":
        decode_callable = replacement_decode
        fused_callable = replacement_fused_ce if previous.supports_chunked else None
    try:
        prior_bank_mod.register_decode(
            decode_mode,
            supports_full=previous.supports_full,
            supports_chunked=previous.supports_chunked,
            fused_ce=fused_callable,
            family_consistent=previous.family_consistent,
            covariance_kinds=previous.covariance_kinds,
            can_omit_base_mean=previous.can_omit_base_mean,
            can_omit_base_variance=previous.can_omit_base_variance,
            override=True,
        )(decode_callable)

        with pytest.raises(
            ValueError,
            match=r"canonical_content_projected.*built-in analytic.*full",
        ):
            _projected_cfg(
                decode_mode=decode_mode,
                use_priorbank_head_evidence_mixer=False,
            )
        with pytest.raises(
            ValueError,
            match=r"canonical_content_projected.*built-in analytic.*full",
        ):
            PriorBank(
                vocab_size=9,
                K=4,
                n_gen=8,
                family="gaussian_diagonal",
                encode_mode="canonical_content_projected",
                decode_mode=decode_mode,
            )
    finally:
        # Restore the exact import-time object, including its original callable/fused identity.
        prior_bank_mod._DECODERS[decode_mode] = previous
    assert prior_bank_mod.get_decode_registration(decode_mode) is previous


@pytest.mark.registry_mutation
def test_ordinary_full_chunked_runtime_preserves_custom_fused_override() -> None:
    """The projected-only identity gate must not narrow ordinary registry extensibility."""
    from vfe3.model import prior_bank as prior_bank_mod

    decode_mode = "full_chunked"
    previous = prior_bank_mod._DECODERS[decode_mode]
    assert previous.fused_ce is not None

    def replacement_decode(pb, mu_q, sigma_q, tau_eff):
        return previous.callable(pb, mu_q, sigma_q, tau_eff)

    def replacement_fused_ce(
        pb,
        mu_q,
        sigma_q,
        targets,
        *,
        tau=None,
        chunk_size=None,
        ignore_index=-100,
    ):
        assert previous.fused_ce is not None
        return previous.fused_ce(
            pb,
            mu_q,
            sigma_q,
            targets,
            tau=tau,
            chunk_size=chunk_size,
            ignore_index=ignore_index,
        )

    try:
        prior_bank_mod.register_decode(
            decode_mode,
            supports_full=True,
            supports_chunked=True,
            fused_ce=replacement_fused_ce,
            covariance_kinds=frozenset({"full"}),
            can_omit_base_mean=True,
            can_omit_base_variance=True,
            override=True,
        )(replacement_decode)
        cfg = VFE3Config(
            vocab_size=11,
            embed_dim=4,
            n_heads=2,
            max_seq_len=3,
            batch_size=1,
            n_layers=1,
            n_e_steps=1,
            oracle_unroll_grad=True,
            family="gaussian_full",
            decode_mode=decode_mode,
            max_steps=1,
        )
        model = VFEModel(cfg).to(TASK6_DEVICE)
        tokens = torch.tensor([[2, 5]], device=TASK6_DEVICE)
        targets = torch.tensor([[7, 3]], device=TASK6_DEVICE)
        logits, loss, ce = model(tokens, targets)
        assert logits is None
        assert torch.isfinite(loss)
        assert torch.isfinite(ce)
    finally:
        prior_bank_mod._DECODERS[decode_mode] = previous
    assert prior_bank_mod.get_decode_registration(decode_mode) is previous


def test_direct_projected_prior_bank_rejects_locally_incompatible_family() -> None:
    """Direct bank construction must not bypass the projected encoder's family boundary."""
    with pytest.raises(ValueError, match="canonical_content_projected"):
        PriorBank(
            vocab_size=9,
            K=4,
            n_gen=8,
            family="gaussian_full",
            encode_mode="canonical_content_projected",
            decode_mode="full",
        )


def test_projected_tied_and_untied_banks_share_only_the_canonical_initial_values() -> None:
    """Untying decode must clone the canonical tables while encode retains its original leaves."""
    tied = _projected_bank()
    untied = _projected_bank(untie_decode_bank=True)

    assert torch.equal(tied.mu_embed, untied.mu_embed)
    assert torch.equal(tied.sigma_log_embed, untied.sigma_log_embed)
    assert torch.equal(untied.decode_mu_embed, untied.mu_embed)
    assert torch.equal(untied.decode_sigma_log_embed, untied.sigma_log_embed)
    assert untied.decode_mu_embed.data_ptr() != untied.mu_embed.data_ptr()
    assert untied.decode_sigma_log_embed.data_ptr() != untied.sigma_log_embed.data_ptr()


def _set_noncommuting_model_frames(model: VFEModel, token_ids: torch.Tensor) -> None:
    device = model.prior_bank.phi_embed.device
    token_blocks = torch.tensor(
        [
            [[[0.10, 0.28], [-0.16, 0.05]], [[-0.08, 0.19], [0.07, 0.12]]],
            [[[0.04, -0.21], [0.13, -0.09]], [[0.11, 0.06], [-0.18, 0.03]]],
        ],
        dtype=model.prior_bank.phi_embed.dtype,
        device=device,
    )
    position_blocks = torch.tensor(
        [
            [[[0.03, -0.17], [0.09, 0.06]], [[-0.12, 0.04], [0.15, 0.02]]],
            [[[-0.07, 0.14], [0.05, 0.08]], [[0.09, -0.13], [0.04, -0.05]]],
        ],
        dtype=model.prior_bank.phi_embed.dtype,
        device=device,
    )
    with torch.no_grad():
        model.prior_bank.phi_embed[token_ids[0]] = token_blocks.flatten(start_dim=1)
        model.pos_phi_free[: token_ids.shape[1]] = position_blocks.flatten(start_dim=1)
        model.prior_bank.mu_embed[token_ids[0]] = torch.tensor(
            [[0.35, -0.25, 0.55, 0.10], [-0.40, 0.65, 0.15, -0.30]],
            device=device,
        )
        model.prior_bank.sigma_log_embed[token_ids[0]] = torch.log(torch.tensor(
            [[0.7, 1.3, 0.5, 1.1], [1.4, 0.6, 0.9, 1.2]],
            device=device,
        ))


def test_projected_materialize_uses_one_shared_factored_frame_and_preserves_q0_equals_p(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebuilding transport or materializing q0 and p separately breaks their shared frame/value."""
    import vfe3.inference.e_step as e_step_module
    import vfe3.model.model as model_module

    model = VFEModel(_projected_cfg(
        pos_phi="learned",
        e_step_update="mm_exact",
    ))
    monkeypatch.setattr(model, "_compact_phi_blocks_enabled", lambda: False)
    token_ids = torch.tensor([[2, 5]])
    _set_noncommuting_model_frames(model, token_ids)
    canonical = model.prior_bank.encode(token_ids)

    original_builder = e_step_module.build_belief_transport
    built: list[object] = []

    def tracked_builder(*args: object, **kwargs: object) -> object:
        transport = original_builder(*args, **kwargs)
        built.append(transport)
        return transport

    original_stack = model_module.vfe_stack
    entry: dict[str, object] = {}

    def tracked_stack(
        belief: object,
        mu_p: torch.Tensor,
        sigma_p: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> object:
        entry.update(
            belief=belief,
            mu_p=mu_p,
            sigma_p=sigma_p,
            prebuilt=kwargs.get("prebuilt_transport"),
        )
        return original_stack(belief, mu_p, sigma_p, *args, **kwargs)

    monkeypatch.setattr(e_step_module, "build_belief_transport", tracked_builder)
    monkeypatch.setattr(model_module, "vfe_stack", tracked_stack)
    capture: dict = {}

    model.forward_beliefs(token_ids, capture=capture)

    assert len(built) == 1
    transport = built[0]
    assert isinstance(transport, FactoredTransport)
    assert entry["prebuilt"] is transport
    context = capture["canonical_frame"]
    assert context.forward is transport.exp_phi
    assert context.inverse is transport.exp_neg_phi

    token_matrix = torch.einsum(
        "...a,aij->...ij", canonical.phi, model.group.generators)
    position_matrix = torch.einsum(
        "...a,aij->...ij", model.pos_phi_free[: token_ids.shape[1]], model.group.generators)
    expected_frame = torch.matrix_exp(token_matrix) @ torch.matrix_exp(position_matrix)
    reversed_frame = torch.matrix_exp(position_matrix) @ torch.matrix_exp(token_matrix)
    assert torch.allclose(context.forward, expected_frame, atol=2e-6, rtol=2e-6)
    assert not torch.allclose(context.forward, reversed_frame, atol=1e-5, rtol=1e-5)

    expected_mu, expected_var = project_canonical_diagonal(
        canonical.mu, canonical.sigma, context.forward)
    entry_belief = entry["belief"]
    assert entry_belief.mu is entry["mu_p"]
    assert entry_belief.sigma is entry["sigma_p"]
    assert torch.equal(entry_belief.mu, expected_mu)
    assert torch.equal(entry_belief.sigma, expected_var)
    assert capture["prior"].mu is entry_belief.mu
    assert capture["prior"].sigma is entry_belief.sigma

    pulled_mu, pulled_cov = pullback_diagonal_query(
        entry_belief.mu, entry_belief.sigma, context.inverse)
    assert torch.equal(
        pulled_mu,
        torch.einsum("...ij,...j->...i", context.inverse, entry_belief.mu),
    )
    assert torch.equal(
        pulled_cov,
        context.inverse
        @ torch.diag_embed(entry_belief.sigma)
        @ context.inverse.transpose(-1, -2),
    )


def test_projected_materialize_expands_only_compact_vertex_factors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compact transport must supply dense vertices without materializing pairwise Omega."""
    import vfe3.inference.e_step as e_step_module

    model = VFEModel(_projected_cfg(
        pos_phi="learned",
        e_step_update="mm_exact",
    ))
    token_ids = torch.tensor([[2, 5]])
    _set_noncommuting_model_frames(model, token_ids)
    original_builder = e_step_module.build_belief_transport
    built: list[object] = []

    def tracked_builder(*args: object, **kwargs: object) -> object:
        transport = original_builder(*args, **kwargs)
        built.append(transport)
        return transport

    def pairwise_forbidden(_self: object) -> torch.Tensor:
        raise AssertionError("projected materialization built pairwise Omega")

    monkeypatch.setattr(e_step_module, "build_belief_transport", tracked_builder)
    monkeypatch.setattr(CompactFactoredTransport, "to_dense_omega", pairwise_forbidden)
    capture: dict = {}

    model.forward_beliefs(token_ids, capture=capture)

    assert len(built) == 1
    transport = built[0]
    assert isinstance(transport, CompactFactoredTransport)
    context = capture["canonical_frame"]
    assert torch.equal(
        context.forward,
        CompactBlockElement(transport.exp_blocks, transport.K).to_dense(),
    )
    assert torch.equal(
        context.inverse,
        CompactBlockElement(transport.inv_blocks, transport.K).to_dense(),
    )


def test_projected_fixed_depth_reuses_one_transport_across_all_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The context frame must remain the E-step frame across every fixed-depth iteration."""
    import vfe3.inference.e_step as e_step_module

    model = VFEModel(_projected_cfg(
        vocab_size=7,
        n_layers=2,
        n_e_steps=2,
        e_step_update="mm_exact",
    ))
    token_ids = torch.tensor([[2, 5]])
    original_builder = e_step_module.build_belief_transport
    original_iteration = e_step_module.e_step_iteration
    built: list[object] = []
    consumed: list[object] = []

    def tracked_builder(*args: object, **kwargs: object) -> object:
        transport = original_builder(*args, **kwargs)
        built.append(transport)
        return transport

    def tracked_iteration(*args: object, **kwargs: object) -> object:
        consumed.append(kwargs.get("_prebuilt_omega"))
        return original_iteration(*args, **kwargs)

    monkeypatch.setattr(e_step_module, "build_belief_transport", tracked_builder)
    monkeypatch.setattr(e_step_module, "e_step_iteration", tracked_iteration)
    capture: dict = {}

    model.forward_beliefs(token_ids, capture=capture)

    assert len(built) == 1
    assert len(consumed) == 4
    assert all(transport is built[0] for transport in consumed)
    context = capture["canonical_frame"]
    if isinstance(built[0], CompactFactoredTransport):
        assert torch.equal(
            context.forward,
            CompactBlockElement(built[0].exp_blocks, built[0].K).to_dense(),
        )
    else:
        assert isinstance(built[0], FactoredTransport)
        assert context.forward is built[0].exp_phi


@pytest.mark.parametrize("gauge_group", ["block_glk", "glk"])
def test_projected_single_head_groups_request_factored_vertex_transport(
    gauge_group: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-block projected groups still require exact vertex factors, never pairwise Omega."""
    import vfe3.inference.e_step as e_step_module

    model = VFEModel(_projected_cfg(
        gauge_group=gauge_group,
        n_heads=1,
        pos_phi="none",
        e_step_update="mm_exact",
    ))
    original_builder = e_step_module.build_belief_transport
    built: list[object] = []

    def tracked_builder(*args: object, **kwargs: object) -> object:
        transport = original_builder(*args, **kwargs)
        built.append(transport)
        return transport

    def pairwise_forbidden(_self: object) -> torch.Tensor:
        raise AssertionError("projected single-head path materialized pairwise Omega")

    monkeypatch.setattr(e_step_module, "build_belief_transport", tracked_builder)
    monkeypatch.setattr(FactoredTransport, "to_dense_omega", pairwise_forbidden)
    capture: dict = {}

    model.forward_beliefs(torch.tensor([[2, 5]]), capture=capture)

    assert len(built) == 1
    assert isinstance(built[0], FactoredTransport)
    assert capture["canonical_frame"].forward is built[0].exp_phi


def test_projected_model_materialization_preserves_float32_under_bf16_autocast() -> None:
    """The model's public projected prior remains float32 inside its AMP region."""
    model = VFEModel(_projected_cfg(
        amp_dtype="bf16",
        pos_phi="none",
        e_step_update="mm_exact",
    ))
    capture: dict = {}

    model.forward_beliefs(torch.tensor([[2, 5]]), capture=capture)

    canonical = model.prior_bank.encode(torch.tensor([[2, 5]]))
    context = capture["canonical_frame"]
    prior = capture["prior"]
    assert canonical.mu.dtype == torch.float32
    assert context.forward.dtype == torch.float32
    assert context.inverse.dtype == torch.float32
    assert prior.mu.dtype == torch.float32
    assert prior.sigma.dtype == torch.float32


def test_projected_phi_gradient_reaches_canonical_tables_and_realized_frames() -> None:
    """Detaching materialization severs at least one supervised mean/variance/frame gradient."""
    model = VFEModel(_projected_cfg(
        pos_phi="learned",
        e_step_update="gradient",
        e_q_mu_lr=0.03,
        e_q_sigma_lr=0.02,
    ))
    token_ids = torch.tensor([[2, 5]])
    _set_noncommuting_model_frames(model, token_ids)
    capture: dict = {}
    model.forward_beliefs(token_ids, capture=capture)
    materialized_prior = capture["prior"]
    target_mu = torch.tensor(
        [[[0.8, -0.1, 0.2, -0.5], [-0.2, 0.3, 0.9, 0.4]]],
        dtype=materialized_prior.mu.dtype,
    )
    target_var = torch.tensor(
        [[[0.6, 1.0, 1.4, 0.8], [1.1, 0.7, 0.5, 1.3]]],
        dtype=materialized_prior.sigma.dtype,
    )
    supervised_scalar = (
        (materialized_prior.mu - target_mu).square().sum()
        + 0.37 * (materialized_prior.sigma - target_var).square().sum()
    )

    supervised_scalar.backward()

    for parameter in (
        model.prior_bank.mu_embed,
        model.prior_bank.sigma_log_embed,
        model.prior_bank.phi_embed,
        model.pos_phi_free,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad).item() > 0


def _manual_canonical_full_logits(
    bank: PriorBank,
    mu_c: torch.Tensor,
    cov_c: torch.Tensor,
) -> torch.Tensor:
    """Independent full-Gaussian KL against the bank's canonical diagonal priors."""
    mu_v = bank.decode_mu_embed if bank.untie_decode_bank else bank.mu_embed
    log_var_v = (
        bank.decode_sigma_log_embed if bank.untie_decode_bank else bank.sigma_log_embed
    )
    var_v = torch.exp(log_var_v).clamp(min=bank.eps)
    inv_var_v = var_v.reciprocal()
    delta = mu_c.unsqueeze(-2) - mu_v
    trace = torch.diagonal(cov_c, dim1=-2, dim2=-1) @ inv_var_v.transpose(-1, -2)
    mahalanobis = (delta.square() * inv_var_v).sum(dim=-1)
    logdet_q = torch.linalg.slogdet(cov_c).logabsdet.unsqueeze(-1)
    logdet_v = torch.log(var_v).sum(dim=-1)
    kl_v = 0.5 * (trace + mahalanobis - bank.K + logdet_v - logdet_q)
    tau_eff = bank.decode_tau * torch.exp(-bank.decode_log_scale.clamp(-3.0, 3.0))
    logits = -kl_v / tau_eff
    if bank.decode_unigram_prior:
        logits = logits + bank.unigram_kappa * bank.unigram_log_prior
    return logits


def _set_decode_oracle_tables(model: VFEModel) -> None:
    bank = model.prior_bank
    mu_values = torch.linspace(
        -0.75, 0.85, steps=bank.vocab_size * bank.K,
        dtype=bank.mu_embed.dtype,
        device=bank.mu_embed.device,
    ).reshape(bank.vocab_size, bank.K)
    var_values = torch.linspace(
        0.45, 1.65, steps=bank.vocab_size * bank.K,
        dtype=bank.sigma_log_embed.dtype,
        device=bank.sigma_log_embed.device,
    ).reshape(bank.vocab_size, bank.K)
    with torch.no_grad():
        if bank.untie_decode_bank:
            bank.decode_mu_embed.copy_(mu_values)
            bank.decode_sigma_log_embed.copy_(var_values.log())
        else:
            bank.mu_embed.copy_(mu_values)
            bank.sigma_log_embed.copy_(var_values.log())
        bank.decode_log_scale.fill_(0.29)
    bank.set_unigram_log_prior(
        torch.tensor([2, 11, 5, 3, 17, 7, 13, 19, 23, 29, 31], dtype=torch.float32)
    )


@pytest.mark.parametrize("decode_mode", ["full", "full_chunked"])
@pytest.mark.parametrize("untie_decode_bank", [False, True])
def test_projected_manual_decode_matches_canonical_full_oracle(
    decode_mode: str,
    untie_decode_bank: bool,
) -> None:
    """Scoring the materialized diagonal query without pullback breaks this hand KL oracle."""
    torch.manual_seed(101)
    model = VFEModel(_projected_cfg(
        decode_mode=decode_mode,
        untie_decode_bank=untie_decode_bank,
        decode_unigram_prior=True,
        unigram_kappa=0.63,
        pos_phi="learned",
        e_step_update="mm_exact",
    )).to(TASK6_DEVICE)
    token_ids = torch.tensor([[2, 5]], device=TASK6_DEVICE)
    _set_decode_oracle_tables(model)
    _set_noncommuting_model_frames(model, token_ids)
    capture: dict = {}

    belief, logits = model.forward_beliefs(
        token_ids,
        return_logits=True,
        capture=capture,
    )

    context = capture["canonical_frame"]
    mu_c, cov_c = pullback_diagonal_query(
        belief.mu,
        belief.sigma,
        context.inverse,
    )
    expected = _manual_canonical_full_logits(model.prior_bank, mu_c, cov_c)
    token_only = torch.matrix_exp(torch.einsum(
        "...a,aij->...ij",
        model.prior_bank.encode(token_ids).phi,
        model.group.generators,
    ))
    assert not torch.allclose(context.forward, token_only, atol=1e-6, rtol=1e-6)
    assert logits is not None
    # The chunked KL decode preserves a float64 ranking escalation (_decode_analytic_kl_logits);
    # this hand oracle inherits float32 from its inputs. Compare values, not dtypes.
    assert torch.allclose(logits.to(expected.dtype), expected, atol=4e-4, rtol=4e-4)


def test_projected_missing_frame_fails_closed_at_priorbank_decode() -> None:
    """A projected query cannot silently be interpreted as already canonical."""
    bank = _projected_bank()
    mu_q = torch.zeros(1, 2, bank.K)
    var_q = torch.ones_like(mu_q)

    with pytest.raises(
        ValueError,
        match=r"canonical_content_projected.*requires canonical_frame.*same forward query",
    ):
        bank.decode(mu_q, var_q)


def test_projected_missing_frame_fails_closed_at_full_fused_ce() -> None:
    """The registered fused boundary must enforce the same same-forward context contract."""
    bank = _projected_gradient_bank("full_chunked")
    mu_q = torch.zeros(1, 2, bank.K, dtype=torch.float64)
    var_q = torch.ones_like(mu_q)
    targets = torch.tensor([[0, 8]])
    fused_ce = get_decode_registration("full_chunked").fused_ce
    assert fused_ce is not None

    with pytest.raises(
        ValueError,
        match=r"canonical_content_projected.*requires canonical_frame.*same forward query",
    ):
        fused_ce(bank, mu_q, var_q, targets)


@pytest.mark.parametrize(
    "incompatibility", ["shape", "dtype", "dtype_reduced", "dtype_variance", "device"])
def test_projected_incompatible_frame_fails_closed_at_priorbank_decode(
    incompatibility: str,
) -> None:
    """Broadcasting, promotion, or cross-device frame reuse would decode the wrong query.

    A float64 frame on a float32 query is NOT in this list: that is the escalated frame pair of
    audit 2026-09-23 #12 and is admitted (see the test below). Every other dtype mismatch fails.
    """
    bank = _projected_bank()
    mu_q = torch.zeros(1, 2, bank.K)
    var_q = torch.ones_like(mu_q)
    frame = torch.eye(bank.K).expand(1, 2, bank.K, bank.K).clone()
    if incompatibility == "shape":
        frame = frame[:, :1]
    elif incompatibility == "dtype":
        mu_q, var_q = mu_q.double(), var_q.double()        # float32 frame, float64 query
    elif incompatibility == "dtype_reduced":
        mu_q, var_q = mu_q.bfloat16(), var_q.bfloat16()    # float32 frame, bf16 query
    elif incompatibility == "dtype_variance":
        frame, var_q = frame.double(), var_q.double()       # float64 frame, mixed query moments
    else:
        frame = torch.eye(bank.K, device="meta").expand(1, 2, bank.K, bank.K)
    context = CanonicalFrameContext(forward=frame, inverse=frame.clone())

    with pytest.raises(ValueError):
        bank.decode(mu_q, var_q, canonical_frame=context)


def test_projected_escalated_float64_frame_is_admitted_at_priorbank_decode() -> None:
    """An escalated float64 frame pulls a float32 query back at float64 and returns float32."""
    bank = _projected_bank()
    torch.manual_seed(5)
    mu_q = torch.randn(1, 2, bank.K)
    var_q = torch.rand(1, 2, bank.K) + 0.5
    frame64 = torch.linalg.matrix_exp(0.3 * torch.randn(1, 2, bank.K, bank.K, dtype=torch.float64))
    inverse64 = torch.linalg.inv(frame64)
    escalated = CanonicalFrameContext(forward=frame64, inverse=inverse64)

    got = bank._query_in_decode_frame(mu_q, var_q, escalated)
    want = bank._query_in_decode_frame(mu_q.double(), var_q.double(), escalated)
    assert all(t.dtype == torch.float32 for t in got)
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w.float(), rtol=1e-6, atol=1e-6)
    assert torch.isfinite(bank.decode(mu_q, var_q, canonical_frame=escalated)).all()


def test_ordinary_decode_rejects_frame_context_and_is_unchanged_without_it() -> None:
    """The new frame keyword must remain illegal and value-inert for ordinary encoders."""
    torch.manual_seed(113)
    bank = PriorBank(vocab_size=7, K=4, n_gen=8, decode_mode="diagonal")
    mu_q = torch.randn(1, 2, 4)
    var_q = torch.rand(1, 2, 4) + 0.4
    context = CanonicalFrameContext(
        forward=torch.eye(4).expand(1, 2, 4, 4).clone(),
        inverse=torch.eye(4).expand(1, 2, 4, 4).clone(),
    )

    baseline = bank.decode(mu_q, var_q)
    assert torch.isfinite(baseline).all()
    with pytest.raises(ValueError, match=r"canonical_frame.*canonical_content_projected"):
        bank.decode(mu_q, var_q, canonical_frame=context)


def test_ordinary_full_fused_ce_rejects_supplied_frame_context() -> None:
    """Only the projected encoder may activate the fused frame keyword."""
    bank = PriorBank(
        vocab_size=7,
        K=4,
        n_gen=8,
        family="gaussian_full",
        diagonal_covariance=False,
        decode_mode="full_chunked",
    )
    mu_q = torch.zeros(1, 2, 4)
    cov_q = torch.eye(4).expand(1, 2, 4, 4).clone()
    targets = torch.tensor([[0, 6]])
    context = CanonicalFrameContext(
        forward=torch.eye(4).expand(1, 2, 4, 4).clone(),
        inverse=torch.eye(4).expand(1, 2, 4, 4).clone(),
    )
    fused_ce = get_decode_registration("full_chunked").fused_ce
    assert fused_ce is not None

    with pytest.raises(ValueError, match=r"canonical_frame.*canonical_content_projected"):
        fused_ce(bank, mu_q, cov_q, targets, canonical_frame=context)


def _projected_gradient_bank(decode_mode: str) -> PriorBank:
    torch.manual_seed(127)
    bank = PriorBank(
        vocab_size=9,
        K=4,
        n_gen=8,
        family="gaussian_diagonal",
        encode_mode="canonical_content_projected",
        decode_mode=decode_mode,
        decode_chunk_size=4,
        decode_ce_checkpoint=("always" if decode_mode == "full_chunked" else "auto"),
        irrep_dims=[2, 2],
        use_priorbank_head_evidence_mixer=True,
        decode_unigram_prior=True,
        unigram_kappa=0.57,
    ).double()
    with torch.no_grad():
        bank.mu_embed.copy_(torch.linspace(-0.8, 0.9, 36).reshape(9, 4))
        bank.sigma_log_embed.copy_(torch.linspace(0.55, 1.45, 36).reshape(9, 4).log())
        bank.decode_log_scale.fill_(0.31)
        bank.head_evidence_logits.copy_(torch.tensor([0.37, -0.22], dtype=torch.float64))
    bank.set_unigram_log_prior(torch.tensor([2, 11, 5, 3, 17, 7, 13, 19, 23]))
    return bank


def _projected_decode_loss_and_grads(
    bank: PriorBank,
    *,
    fused: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mu_q = torch.tensor(
        [[
            [0.2, -0.5, 0.7, 0.1],
            [-0.3, 0.8, -0.2, 0.4],
            [0.9, -0.1, 0.3, -0.6],
            [-0.4, 0.2, 0.5, 0.7],
        ]],
        dtype=torch.float64,
        requires_grad=True,
    )
    var_q = torch.tensor(
        [[
            [0.8, 1.2, 0.6, 1.1],
            [1.0, 0.7, 1.3, 0.9],
            [0.5, 1.4, 0.8, 1.2],
            [0.9, -0.25, 1.1, 0.7],
        ]],
        dtype=torch.float64,
        requires_grad=True,
    )
    frame = (
        torch.eye(4, dtype=torch.float64).expand(1, 4, 4, 4).clone()
        + torch.tensor(
            [[
                [[0.10, 0.07, 0.00, 0.00], [-0.04, -0.03, 0.00, 0.00],
                 [0.00, 0.00, 0.06, -0.08], [0.00, 0.00, 0.05, 0.02]],
                [[-0.05, 0.09, 0.00, 0.00], [0.03, 0.04, 0.00, 0.00],
                 [0.00, 0.00, -0.07, 0.04], [0.00, 0.00, 0.08, 0.01]],
                [[0.02, -0.06, 0.00, 0.00], [0.11, -0.02, 0.00, 0.00],
                 [0.00, 0.00, 0.03, 0.07], [0.00, 0.00, -0.05, -0.04]],
                [[-0.08, 0.03, 0.00, 0.00], [0.06, 0.05, 0.00, 0.00],
                 [0.00, 0.00, 0.09, -0.02], [0.00, 0.00, 0.04, -0.06]],
            ]],
            dtype=torch.float64,
        )
    ).requires_grad_()
    context = CanonicalFrameContext(forward=frame, inverse=torch.linalg.inv(frame))
    targets = torch.tensor([[0, -100, 8, 3]])
    if fused:
        fused_ce = get_decode_registration("full_chunked").fused_ce
        assert fused_ce is not None
        loss = fused_ce(
            bank,
            mu_q,
            var_q,
            targets,
            canonical_frame=context,
        )
    else:
        logits = bank.decode(mu_q, var_q, canonical_frame=context)
        degenerate = bank.decode_degenerate_positions(
            var_q,
            canonical_frame=context,
        )
        assert degenerate is not None
        dense_targets = torch.where(degenerate, targets.new_full((), -100), targets)
        flat_logits = logits.reshape(-1, bank.vocab_size)
        flat_targets = dense_targets.reshape(-1)
        valid = flat_targets != -100
        n_valid = valid.sum().clamp_min(1)
        loss = F.cross_entropy(
            flat_logits,
            flat_targets,
            ignore_index=-100,
            reduction="sum",
        ) / n_valid

    leaves = {
        "query_mean": mu_q,
        "query_variance": var_q,
        "frame": frame,
        "prior_mean": bank.mu_embed,
        "prior_variance": bank.sigma_log_embed,
        "temperature": bank.decode_log_scale,
        "evidence_logits": bank.head_evidence_logits,
    }
    gradients = torch.autograd.grad(loss, tuple(leaves.values()))
    return loss.detach(), dict(zip(leaves, gradients))


def test_projected_dense_nonfinite_variance_is_sanitized_before_frame_backward() -> None:
    """An excluded projected row must not poison its realized-frame gradient."""
    from vfe3.numerics import (
        decode_logdet_fallback_elements,
        reset_decode_logdet_fallback_elements,
    )

    bank = _projected_gradient_bank("full")
    mu = torch.tensor(
        [[[0.2, -0.5, 0.7, 0.1], [-0.3, 0.8, -0.2, 0.4]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    var = torch.tensor(
        [[[float("nan"), 1.2, 0.6, 1.1], [1.0, 0.7, 1.3, 0.9]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    perturb = torch.tensor(
        [[
            [[0.10, 0.07, 0.00, 0.00], [-0.04, -0.03, 0.00, 0.00],
             [0.00, 0.00, 0.06, -0.08], [0.00, 0.00, 0.05, 0.02]],
            [[-0.05, 0.09, 0.00, 0.00], [0.03, 0.04, 0.00, 0.00],
             [0.00, 0.00, -0.07, 0.04], [0.00, 0.00, 0.08, 0.01]],
        ]],
        dtype=torch.float64,
    )
    frame = (
        torch.eye(4, dtype=torch.float64).expand(1, 2, 4, 4).clone() + perturb
    ).requires_grad_()
    context = CanonicalFrameContext(forward=frame, inverse=torch.linalg.inv(frame))
    targets = torch.tensor([[1, 2]])

    reset_decode_logdet_fallback_elements()
    logits = bank.decode(mu, var, canonical_frame=context)
    mask = bank.decode_degenerate_positions(var, canonical_frame=context)
    assert mask is not None
    ce_targets = torch.where(mask, targets.new_full((), -100), targets)
    loss = F.cross_entropy(
        logits.reshape(-1, bank.vocab_size),
        ce_targets.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    ) / (ce_targets != -100).sum().clamp_min(1)
    loss.backward()

    assert torch.equal(mask, torch.tensor([[True, False]]))
    assert decode_logdet_fallback_elements() == 1
    assert torch.isfinite(loss)
    assert torch.equal(mu.grad[0, 0], torch.zeros_like(mu.grad[0, 0]))
    assert torch.equal(var.grad[0, 0], torch.zeros_like(var.grad[0, 0]))
    assert torch.isfinite(frame.grad).all()
    assert torch.equal(frame.grad[0, 0], torch.zeros_like(frame.grad[0, 0]))


def test_projected_dense_fused_value_gradient_parity_with_invalid_query() -> None:
    """Skipping pullback in either CE path breaks values, checkpoint grads, or exclusion."""
    dense = _projected_gradient_bank("full")
    fused = _projected_gradient_bank("full_chunked")
    fused.load_state_dict(dense.state_dict())

    dense_loss, dense_grads = _projected_decode_loss_and_grads(dense, fused=False)
    fused_loss, fused_grads = _projected_decode_loss_and_grads(fused, fused=True)

    assert torch.allclose(fused_loss, dense_loss, atol=2e-9, rtol=2e-9)
    for name in dense_grads:
        assert torch.isfinite(dense_grads[name]).all(), name
        assert torch.isfinite(fused_grads[name]).all(), name
        assert torch.count_nonzero(dense_grads[name]).item() > 0, name
        assert torch.count_nonzero(fused_grads[name]).item() > 0, name
        assert torch.allclose(fused_grads[name], dense_grads[name], atol=2e-8, rtol=2e-7), name


def _retain_forward_query(model: VFEModel, holder: dict[str, object]) -> None:
    original = model.forward_beliefs

    def tracked(*args: object, **kwargs: object):
        belief, logits = original(*args, **kwargs)
        belief.mu.retain_grad()
        belief.sigma.retain_grad()
        holder["belief"] = belief
        capture = kwargs.get("capture")
        if isinstance(capture, dict):
            holder["canonical_frame"] = capture.get("canonical_frame")
        return belief, logits

    model.forward_beliefs = tracked  # type: ignore[method-assign]


def _projected_training_model(decode_mode: str) -> VFEModel:
    torch.manual_seed(139)
    model = VFEModel(_projected_cfg(
        decode_mode=decode_mode,
        decode_chunk_size=4,
        decode_ce_checkpoint=("always" if decode_mode == "full_chunked" else "auto"),
        decode_unigram_prior=True,
        unigram_kappa=0.61,
        use_priorbank_head_evidence_mixer=True,
        pos_phi="learned",
        e_step_update="gradient",
        e_q_mu_lr=0.03,
        e_q_sigma_lr=0.02,
    ))
    token_ids = torch.tensor([[2, 5]])
    _set_noncommuting_model_frames(model, token_ids)
    model.prior_bank.set_unigram_log_prior(
        torch.tensor([2, 11, 5, 3, 17, 7, 13, 19, 23, 29, 31])
    )
    with torch.no_grad():
        model.prior_bank.decode_log_scale.fill_(0.23)
        model.prior_bank.head_evidence_logits.copy_(torch.tensor([0.31, -0.19]))
    return model.to(TASK6_DEVICE)


def test_projected_model_dense_fused_training_threads_context_and_gradients() -> None:
    """Dropping the same-forward frame from model training severs phi or decode gradients."""
    dense = _projected_training_model("full")
    fused = _projected_training_model("full_chunked")
    fused.load_state_dict(dense.state_dict())
    dense.train()
    fused.train()
    tokens = torch.tensor([[2, 5], [4, 7]], device=TASK6_DEVICE)
    targets = torch.tensor([[10, -100], [3, 1]], device=TASK6_DEVICE)
    dense_holder: dict[str, object] = {}
    fused_holder: dict[str, object] = {}
    _retain_forward_query(dense, dense_holder)
    _retain_forward_query(fused, fused_holder)

    dense_logits, dense_loss, dense_ce = dense(tokens, targets)
    fused_logits, fused_loss, fused_ce = fused(tokens, targets)
    dense_loss.backward()
    fused_loss.backward()

    assert dense_logits is not None
    assert fused_logits is None
    assert isinstance(dense_holder.get("canonical_frame"), CanonicalFrameContext)
    assert isinstance(fused_holder.get("canonical_frame"), CanonicalFrameContext)
    # Fused chunked CE preserves a float64 ranking escalation; the dense twin casts to mu_q.dtype.
    # Values agree to ~1e-7; compare across the intentional dtype boundary.
    assert torch.allclose(fused_ce.to(dense_ce.dtype), dense_ce, atol=7e-4, rtol=7e-4)
    assert torch.allclose(fused_loss.to(dense_loss.dtype), dense_loss, atol=7e-4, rtol=7e-4)

    dense_belief = dense_holder["belief"]
    fused_belief = fused_holder["belief"]
    gradient_pairs = {
        "query_mean": (dense_belief.mu.grad, fused_belief.mu.grad),
        "query_variance": (dense_belief.sigma.grad, fused_belief.sigma.grad),
        "canonical_prior_mean": (dense.prior_bank.mu_embed.grad, fused.prior_bank.mu_embed.grad),
        "canonical_prior_variance": (
            dense.prior_bank.sigma_log_embed.grad,
            fused.prior_bank.sigma_log_embed.grad,
        ),
        "token_phi": (dense.prior_bank.phi_embed.grad, fused.prior_bank.phi_embed.grad),
        "positional_phi": (dense.pos_phi_free.grad, fused.pos_phi_free.grad),
        "evidence_logits": (
            dense.prior_bank.head_evidence_logits.grad,
            fused.prior_bank.head_evidence_logits.grad,
        ),
        "temperature": (
            dense.prior_bank.decode_log_scale.grad,
            fused.prior_bank.decode_log_scale.grad,
        ),
    }
    for name, (dense_grad, fused_grad) in gradient_pairs.items():
        assert dense_grad is not None and fused_grad is not None, name
        assert torch.isfinite(dense_grad).all() and torch.isfinite(fused_grad).all(), name
        assert torch.count_nonzero(dense_grad).item() > 0, name
        assert torch.count_nonzero(fused_grad).item() > 0, name
        assert torch.allclose(fused_grad, dense_grad, atol=2e-3, rtol=6e-3), name
