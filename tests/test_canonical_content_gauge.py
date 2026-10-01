r"""Exact canonical-content gauge encoder: provenance, constraints, and reference behavior."""

import pytest
import torch

from tests.profiles import tiny_config
from vfe3.config import VFE3Config
from vfe3.families.frame_gaussian import FrameDiagonalGaussian
from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import build_factored_transport
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import PriorBank, get_encode


def _exact_cfg(**overrides: object) -> VFE3Config:
    r"""Small configuration on the exact canonical-content scientific control."""
    values = dict(
        vocab_size=7,
        embed_dim=4,
        n_heads=2,
        max_seq_len=5,
        batch_size=2,
        n_layers=1,
        n_e_steps=1,
        e_step_update="gradient",
        oracle_unroll_grad=True,
        family="gaussian_frame_diagonal",
        transport_mode="flat",
        attention_mode="transported",
        relational_input_scope="head",
        relational_output_scope="head",
        attention_residual=False,
        predictive_covariance="passthrough",
        decode_mode="diagonal_chunked",
        encode_mode="canonical_content_gauge",
        lambda_alpha_mode="constant",
        max_steps=1,
        decode_av_precision="fp32",
        use_block_mlp=False,
        norm_type_block="none",
        layernorm_affine=False,
        decode_unigram_prior=False,
    )
    values.update(overrides)
    return tiny_config(**values)


def _bank(
    *,
    encode_mode: str = "canonical_content_gauge",
    untie_decode_bank: bool = False,
) -> PriorBank:
    torch.manual_seed(17)
    return PriorBank(
        vocab_size=9,
        K=4,
        n_gen=6,
        decode_av_precision="fp32",
        family="gaussian_frame_diagonal",
        encode_mode=encode_mode,
        decode_mode="diagonal_chunked",
        untie_decode_bank=untie_decode_bank,
    )


def test_canonical_encoder_is_registered_and_matches_the_per_token_canonical_lookup() -> None:
    r"""Removing the named registration or adding a second table changes this control's result."""
    get_encode("canonical_content_gauge")
    canonical = _bank()
    per_token = _bank(encode_mode="per_token")
    token_ids = torch.tensor([[0, 4, 2], [7, 1, 8]])

    got = canonical.encode(token_ids)
    expected = per_token.encode(token_ids)

    assert torch.equal(got.mu, expected.mu)
    assert torch.equal(got.sigma, expected.sigma)
    assert torch.equal(got.phi, expected.phi)
    assert set(canonical._parameters) == set(per_token._parameters)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("family", "gaussian_diagonal"),
        ("decode_mode", "linear"),
        ("decode_mode", "family_chunked"),
    ],
)
def test_canonical_encoder_rejects_each_incompatible_cross_field_pair(
    field: str,
    value: object,
) -> None:
    r"""Relaxing any exact-control gate would silently change the experimental arm."""
    with pytest.raises(ValueError, match="canonical_content_gauge"):
        _exact_cfg(**{field: value})


def test_direct_prior_bank_construction_rejects_its_locally_knowable_incompatible_mode() -> None:
    r"""Changing the family beneath the named encoder must not bypass config validation."""
    with pytest.raises(ValueError, match="canonical_content_gauge"):
        PriorBank(
            vocab_size=9,
            K=4,
            n_gen=6,
        decode_av_precision="fp32",
            family="gaussian_diagonal",
            encode_mode="canonical_content_gauge",
        )


@pytest.mark.registry_mutation
@pytest.mark.parametrize("override_kind", ["registration", "callable_map"])
def test_exact_construction_rejects_overridden_encoder_identity(
    override_kind: str,
) -> None:
    """The exact-control name cannot authorize a replaced callable or registration record."""
    from vfe3.model import prior_bank as prior_bank_mod

    name = "canonical_content_gauge"
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

        with pytest.raises(ValueError, match=r"canonical_content_gauge.*built-in encoder"):
            _exact_cfg()
        with pytest.raises(ValueError, match=r"canonical_content_gauge.*built-in encoder"):
            _bank()
    finally:
        prior_bank_mod._ENCODERS[name] = previous_callable
        prior_bank_mod._ENCODER_REGISTRATIONS[name] = previous_registration
    assert prior_bank_mod._ENCODERS[name] is previous_callable
    assert prior_bank_mod._ENCODER_REGISTRATIONS[name] is previous_registration


@pytest.mark.registry_mutation
def test_exact_construction_rejects_overridden_frame_family_identity() -> None:
    """The exact intrinsic proof depends on the import-time frame-diagonal family."""
    from vfe3.families.base import _FAMILIES, register_family

    name = "gaussian_frame_diagonal"
    previous = _FAMILIES[name]
    try:
        @register_family(name, override=True)
        class _ReplacementFrameDiagonal(FrameDiagonalGaussian):
            pass

        with pytest.raises(ValueError, match=r"canonical_content_gauge.*gaussian_frame_diagonal"):
            _exact_cfg()
        with pytest.raises(ValueError, match=r"canonical_content_gauge.*gaussian_frame_diagonal"):
            _bank()
    finally:
        _FAMILIES[name] = previous
    assert _FAMILIES[name] is previous


def test_canonical_tied_and_untied_banks_keep_canonical_coordinates_and_clone_decode_tables() -> None:
    r"""Untying must begin from the exact canonical table, not a pushed-forward surrogate."""
    tied = _bank()
    untied = _bank(untie_decode_bank=True)
    token_ids = torch.tensor([[3, 0, 5]])

    assert torch.equal(tied.encode(token_ids).mu, tied.mu_embed[token_ids])
    assert torch.equal(tied.encode(token_ids).sigma, torch.exp(tied.sigma_log_embed[token_ids]))
    assert torch.equal(untied.encode(token_ids).mu, untied.mu_embed[token_ids])
    assert torch.equal(untied.encode(token_ids).sigma, torch.exp(untied.sigma_log_embed[token_ids]))
    assert torch.equal(untied.decode_mu_embed, untied.mu_embed)
    assert torch.equal(untied.decode_sigma_log_embed, untied.sigma_log_embed)
    assert untied.decode_mu_embed.data_ptr() != untied.mu_embed.data_ptr()
    assert untied.decode_sigma_log_embed.data_ptr() != untied.sigma_log_embed.data_ptr()


def test_canonical_frame_transport_and_decode_match_the_hand_computed_intrinsic_reference() -> None:
    r"""A dense frame sandwich or fixed-basis decoder would violate the intrinsic KL reference."""
    torch.manual_seed(23)
    bank = _bank()
    with torch.no_grad():
        bank.mu_embed.copy_(torch.tensor([
            [0.0, 0.5, -0.5, 1.0],
            [1.0, -1.0, 0.25, 0.5],
            [-0.25, 0.75, 1.0, -0.5],
            [0.5, 0.0, -1.0, 0.25],
            [-0.5, -0.25, 0.0, 0.5],
            [0.75, 0.5, -0.25, -1.0],
            [-1.0, 0.25, 0.5, 0.0],
            [0.25, -0.75, 1.0, 0.5],
            [0.5, 1.0, 0.75, -0.25],
        ]))
        bank.sigma_log_embed.copy_(torch.log(torch.tensor([
            [0.5, 0.75, 1.0, 1.25],
            [1.5, 1.0, 0.75, 0.5],
            [0.75, 1.25, 1.5, 1.0],
            [1.0, 0.5, 1.25, 0.75],
            [1.25, 1.5, 0.5, 1.0],
            [0.5, 1.0, 1.5, 0.75],
            [1.0, 1.25, 0.75, 1.5],
            [1.5, 0.75, 1.0, 0.5],
            [0.75, 1.0, 1.25, 1.5],
        ])))
    query_mu = torch.tensor([[[0.25, -0.5, 0.75, 1.0]]])
    query_sigma = torch.tensor([[[0.75, 1.25, 0.5, 1.5]]])

    group = get_group("block_glk")(4, 2)
    phi = 0.1 * torch.randn(2, group.generators.shape[0])
    transport = build_factored_transport(phi, group)
    transported = FrameDiagonalGaussian.transport_location(query_mu[0], transport)
    assert torch.equal(transported[0], query_mu[0])

    sigma_v = torch.exp(bank.sigma_log_embed)
    delta = bank.mu_embed - query_mu[0, 0]
    expected = -0.5 * (
        (query_sigma[0, 0] / sigma_v)
        + (delta.square() / sigma_v)
        - 1.0
        + torch.log(sigma_v)
        - torch.log(query_sigma[0, 0])
    ).sum(dim=-1)
    got = bank.decode(query_mu, query_sigma)
    assert torch.allclose(got[0, 0], expected, atol=1e-6, rtol=1e-6)


def test_canonical_model_has_no_supervised_phi_gradient_and_emits_one_provenance_notice() -> None:
    r"""Reconnects of token phi to the belief/decode objective invalidate this exact control."""
    with pytest.warns(UserWarning, match="frame-intrinsic.*phi_embed.*supervised"):
        cfg = _exact_cfg()

    torch.manual_seed(29)
    model = VFEModel(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    _, loss, _ = model(tokens, targets)
    loss.backward()
    assert model.prior_bank.phi_embed.grad is None


def test_canonical_config_accepts_unigram_base_rate_and_names_its_separate_role() -> None:
    r"""A unigram base-rate bias must not be mistaken for a violation of Gaussian exactness."""
    with pytest.warns(UserWarning, match="Gaussian divergence.*additive unigram base-rate"):
        cfg = _exact_cfg(decode_unigram_prior=True, unigram_kappa=1.75)
    assert cfg.decode_unigram_prior is True
    assert cfg.unigram_kappa == 1.75


def test_canonical_unigram_logits_add_the_configured_base_rate_after_intrinsic_kl_temperature() -> None:
    r"""Moving the bias inside the KL temperature or gauge map changes the canonical logits."""
    bank = PriorBank(
        vocab_size=3,
        K=2,
        n_gen=4,
        family="gaussian_frame_diagonal",
        encode_mode="canonical_content_gauge",
        decode_mode="diagonal_chunked",
        decode_tau=2.0,
        decode_unigram_prior=True,
        unigram_kappa=1.75,
    )
    with torch.no_grad():
        bank.mu_embed.copy_(torch.tensor([[0.0, 0.5], [1.0, -0.5], [-0.25, 0.75]]))
        bank.sigma_log_embed.copy_(torch.log(torch.tensor([[0.5, 1.0], [1.5, 0.75], [1.25, 2.0]])))
        bank.decode_log_scale.fill_(0.4)
    counts = torch.tensor([0.0, 3.0, 8.0])
    bank.set_unigram_log_prior(counts)
    query_mu = torch.tensor([[[0.25, -0.75]]])
    query_sigma = torch.tensor([[[0.75, 1.5]]])

    sigma_v = torch.exp(bank.sigma_log_embed)
    delta = bank.mu_embed - query_mu[0, 0]
    intrinsic_kl = 0.5 * (
        (query_sigma[0, 0] / sigma_v)
        + (delta.square() / sigma_v)
        - 1.0
        + torch.log(sigma_v)
        - torch.log(query_sigma[0, 0])
    ).sum(dim=-1)
    tau_eff = 2.0 * torch.exp(torch.tensor(-0.4))
    log_pi = torch.log((counts + 1.0) / (counts.sum() + counts.numel()))
    expected = -intrinsic_kl / tau_eff + 1.75 * log_pi

    got = bank.decode(query_mu, query_sigma)
    assert torch.allclose(got[0, 0], expected, atol=1e-6, rtol=1e-6)
