"""Noisy-query routing: the exact expected-likelihood source posterior on projected moments."""
import math

import pytest
import torch

from vfe3.relational_scoring import (
    projected_gaussian_kl, projected_moments, projected_noisy_query_cost, relational_mixture_mean,
)


def _moments(seed: int = 3, n: int = 4, heads: int = 2, r: int = 3,
             dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    """Random projected means (1, N, H, r) and exactly symmetric SPD covariances (1, N, H, r, r)."""
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(1, n, heads, r, generator=generator, dtype=dtype)
    root = torch.randn(1, n, heads, r, r, generator=generator, dtype=dtype)
    G = root @ root.transpose(-1, -2) + .3 * torch.eye(r, dtype=dtype)
    return a, .5 * (G + G.transpose(-1, -2))


def test_cost_matches_expected_likelihood_oracle() -> None:
    a, G = _moments()
    D = projected_noisy_query_cost(a, G, a, G)
    assert D.shape == (1, 2, 4, 4)
    for h in range(2):
        for i in range(4):
            for j in range(4):
                W = G[0, i, h] + G[0, j, h]
                log_likelihood = torch.distributions.MultivariateNormal(a[0, j, h], W).log_prob(a[0, i, h])
                row_constant = .5 * torch.logdet(2 * math.pi * G[0, i, h])
                torch.testing.assert_close(D[0, h, i, j], -log_likelihood - row_constant,
                                           rtol=1e-10, atol=1e-11)


def test_cost_is_nonnegative_and_invariant_under_a_common_comparison_map() -> None:
    a, G = _moments(seed=5)
    A = torch.tensor([[1.3, -.4, .2], [.1, .7, .5], [-.6, .2, 1.1]], dtype=a.dtype)  # det = 1.123
    D = projected_noisy_query_cost(a, G, a, G)
    Aa = (A @ a.unsqueeze(-1)).squeeze(-1)
    AG = A @ G @ A.transpose(-1, -2)
    AG = .5 * (AG + AG.transpose(-1, -2))
    torch.testing.assert_close(projected_noisy_query_cost(Aa, AG, Aa, AG), D, rtol=1e-9, atol=1e-10)
    assert float(D.min()) >= -1e-12


def test_point_query_limit_matches_gaussian_kl_routing() -> None:
    a, G = _moments(seed=7)
    tiny = 1e-9 * torch.eye(3, dtype=a.dtype).expand_as(G)
    noisy = (-projected_noisy_query_cost(a, tiny, a, G)).softmax(-1)
    kl = (-projected_gaussian_kl(a, tiny, a, G)).softmax(-1)
    torch.testing.assert_close(noisy, kl, rtol=0., atol=1e-6)


@pytest.mark.parametrize("variance, expected", [(1., .5622), (10., .5114)])
def test_manuscript_one_dimensional_example(variance: float, expected: float) -> None:
    a_q = torch.zeros(1, 1, 1, 1, dtype=torch.float64)                       # (1, Nq, H, r)
    G_q = torch.full((1, 1, 1, 1, 1), variance, dtype=torch.float64)
    a_k = torch.tensor([0., 1.], dtype=torch.float64).reshape(1, 2, 1, 1)    # keys b_1 = 0, b_2 = 1
    G_k = torch.ones(1, 2, 1, 1, 1, dtype=torch.float64)
    noisy = float((-projected_noisy_query_cost(a_q, G_q, a_k, G_k)).softmax(-1)[0, 0, 0, 0])
    kl = float((-projected_gaussian_kl(a_q, G_q, a_k, G_k)).softmax(-1)[0, 0, 0, 0])
    assert abs(noisy - expected) < 5e-5
    assert abs(noisy - 1 / (1 + math.exp(-1 / (2 + 2 * variance)))) < 1e-12
    assert abs(kl - 1 / (1 + math.exp(-.5))) < 1e-12


def test_shared_key_covariance_is_whitened_l2_with_the_query_covariance_added() -> None:
    a, G = _moments(seed=9)
    V = G[:, :1].expand_as(G)  # every key shares token 0's covariance
    noisy = (-projected_noisy_query_cost(a, G, a, V)).softmax(-1)
    for h in range(2):
        for i in range(4):
            evals, evecs = torch.linalg.eigh(G[0, i, h] + V[0, 0, h])
            whiten = evecs @ torch.diag(evals.rsqrt()) @ evecs.T    # (V_Q,i + V_*)^{-1/2}
            q = whiten @ a[0, i, h]
            k = a[0, :, h] @ whiten.T                                # (N, r)
            logits = k @ q - .5 * k.square().sum(-1)
            torch.testing.assert_close(noisy[0, h, i], logits.softmax(-1), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pair_tiles_match_dense_without_a_full_matrix_grid(dtype: torch.dtype) -> None:
    from torch.utils._python_dispatch import TorchDispatchMode

    mu = torch.randn(1, 35, 1, 2, generator=torch.Generator().manual_seed(5), dtype=dtype,
                     requires_grad=True)
    G = torch.eye(2, dtype=dtype).expand(1, 35, 1, 2, 2)
    matrix_grids = []

    class RecordSums(TorchDispatchMode):
        def __torch_dispatch__(self, func: object, types: tuple, args: tuple = (),
                               kwargs: dict | None = None) -> object:
            result = func(*args, **(kwargs or {}))
            if func == torch.ops.aten.add.Tensor and isinstance(result, torch.Tensor) and result.ndim == 6:
                matrix_grids.append(tuple(result.shape[-4:-2]))
            return result

    with RecordSums():
        actual = projected_noisy_query_cost(mu, G, mu, G)
        actual_grad, = torch.autograd.grad(actual.square().sum(), mu)
    # W = 2I and G_q = I with r = 2: D = |delta|^2 / 4 + log 2.
    delta = mu[:, :, 0, None, :] - mu[:, None, :, 0, :]
    expected = (.25 * delta.square().sum(-1) + math.log(2.))[:, None]
    expected_grad, = torch.autograd.grad(expected.square().sum(), mu)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-5)
    assert matrix_grids and (35, 35) not in matrix_grids


@pytest.mark.parametrize("role", ["q", "k"])
def test_ill_conditioned_sources_recover_through_float64_factors(role: str) -> None:
    from tests.test_relational_source_factorization import _inputs

    mu, sigma, inverse, S = _inputs()
    S = S.requires_grad_(True)
    identity = torch.eye(3)[None]
    S_q, S_k = (S, identity) if role == "q" else (identity, S)
    aq, Gq = projected_moments(mu, sigma, inverse, S_q)
    ak, Gk = projected_moments(mu, sigma, inverse, S_k)
    with pytest.raises(ValueError, match="positive definite"):
        projected_noisy_query_cost(aq, Gq, ak, Gk)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        score = projected_noisy_query_cost(
            aq, Gq, ak, Gk,
            projection_q=(sigma, inverse, S_q), projection_k=(sigma, inverse, S_k))
    aq64, Gq64 = projected_moments(mu.double(), sigma.double(), inverse.double(), S_q.double())
    ak64, Gk64 = projected_moments(mu.double(), sigma.double(), inverse.double(), S_k.double())
    reference = projected_noisy_query_cost(aq64, Gq64, ak64, Gk64)
    torch.testing.assert_close(score.double(), reference, rtol=2e-5, atol=2e-5)
    gradient, = torch.autograd.grad(score.sum(), S)
    reference_gradient, = torch.autograd.grad(reference.sum(), S)
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, reference_gradient, rtol=5e-4, atol=5e-3)


def test_non_spd_covariance_without_source_fails_explicitly() -> None:
    a, G = _moments(seed=11)
    G = G.clone()
    G[0, 1, 0] = torch.zeros(3, 3, dtype=G.dtype)
    with pytest.raises(ValueError, match="not positive definite"):
        projected_noisy_query_cost(a, G, a, G)


def test_autocast_inputs_score_in_float32() -> None:
    a, G = _moments(seed=13, dtype=torch.float32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        D = projected_noisy_query_cost(a, G, a, G)
    assert D.dtype == torch.float32 and bool(torch.isfinite(D).all())


def test_healthy_path_makes_one_list_transfer_and_one_failure_check(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    original_list, original_bool = torch.Tensor.tolist, torch.Tensor.__bool__

    def tolist(value: torch.Tensor) -> list:
        calls.append("tolist")
        return original_list(value)

    def tobool(value: torch.Tensor) -> bool:
        calls.append("bool")
        return original_bool(value)

    mu = torch.zeros(1, 3, 1, 2)
    covariance = torch.eye(2).expand(1, 3, 1, 2, 2)
    with monkeypatch.context() as scoped:
        scoped.setattr(torch.Tensor, "tolist", tolist)
        scoped.setattr(torch.Tensor, "__bool__", tobool)
        projected_noisy_query_cost(mu, covariance, mu, covariance)
    assert calls == ["tolist", "bool"]


def test_routing_gradients_match_finite_differences() -> None:
    generator = torch.Generator().manual_seed(17)
    mu = torch.randn(1, 3, 1, 2, generator=generator, dtype=torch.float64)   # (1, N, H, d)
    sigma = torch.rand(mu.shape, generator=generator, dtype=mu.dtype) + .7
    U = torch.eye(2, dtype=mu.dtype).expand(1, 3, 1, 2, 2).clone()
    U[..., 0, 1] = .4
    inverse = torch.linalg.inv(U)
    xi = (inverse @ mu.unsqueeze(-1)).squeeze(-1)
    S_q = torch.tensor([[[1., .2], [0., 1.]]], dtype=mu.dtype, requires_grad=True)
    S_k = torch.tensor([[[.9, -.1], [.3, 1.1]]], dtype=mu.dtype, requires_grad=True)
    eye = torch.eye(2, dtype=mu.dtype)[None]

    def routed(S_q: torch.Tensor, S_k: torch.Tensor) -> torch.Tensor:
        a_q, G_q = projected_moments(mu, sigma, inverse, S_q)
        a_k, G_k = projected_moments(mu, sigma, inverse, S_k)
        beta = (-projected_noisy_query_cost(a_q, G_q, a_k, G_k)).softmax(-1)
        return relational_mixture_mean(xi, U, beta, eye, eye)

    assert torch.autograd.gradcheck(routed, (S_q, S_k), eps=1e-6, atol=1e-5, rtol=1e-4)


@pytest.mark.usefixtures("condition_recovery_on")
def test_recovery_is_selected_per_pair_and_leaves_healthy_pairs_bit_identical() -> None:
    from tests.test_remediation_20260923_C import _projected_inputs

    def score(mu: torch.Tensor, sigma: torch.Tensor, inverse: torch.Tensor,
              S_q: torch.Tensor, S_k: torch.Tensor) -> torch.Tensor:
        a_q, G_q = projected_moments(mu, sigma, inverse, S_q)
        a_k, G_k = projected_moments(mu, sigma, inverse, S_k)
        return projected_noisy_query_cost(a_q, G_q, a_k, G_k,
                                          projection_q=(sigma, inverse, S_q, mu),
                                          projection_k=(sigma, inverse, S_k, mu))

    healthy = score(*_projected_inputs(ill=False))
    ill = score(*_projected_inputs(ill=True))  # sequence 1, last token: certificate-true, recovered
    assert torch.isfinite(ill).all() and ill.dtype == torch.float32
    # Another sequence's recovered token leaves sequence 0 bit-identical.
    assert torch.equal(ill[0], healthy[0])
    # A later recovered token leaves every earlier pair of its own sequence bit-identical.
    assert torch.equal(ill[1, :, :-1, :-1], healthy[1, :, :-1, :-1])
    # Pairs that touch the recovered token carry the float64 factor route's accuracy.
    reference = score(*(x.double() for x in _projected_inputs(ill=True)))
    torch.testing.assert_close(ill[1, :, -1].double(), reference[1, :, -1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ill[1, :, :, -1].double(), reference[1, :, :, -1], rtol=1e-5, atol=1e-5)


def test_config_local_validation_serialization_and_inactive_mark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect
    from dataclasses import asdict

    from tests.profiles import config_values, tiny_config
    from tests.test_relational_integration import _cfg
    from vfe3.config import VFE3Config, config_from_serialized, serialize_config

    predictive = dict(attention_mode="projected_predictive")
    assert _cfg(**predictive).relational_routing == "gaussian_kl"
    cfg = _cfg(**predictive, relational_routing="noisy_query")
    assert asdict(config_from_serialized(serialize_config(cfg), source="noisy query")) == asdict(cfg)
    legacy = serialize_config(_cfg(**predictive))
    legacy.pop("relational_routing")
    parameters = list(inspect.signature(VFE3Config.__init__).parameters)[1:]
    values = config_values()
    monkeypatch.setattr(VFE3Config.__init__, "__defaults__", tuple(values[name] for name in parameters))
    assert config_from_serialized(legacy, source="legacy").relational_routing == "gaussian_kl"
    with pytest.raises(ValueError, match="relational_routing"):
        _cfg(**predictive, relational_routing="dot_product")
    with pytest.raises(ValueError, match="requires attention_mode='projected_predictive'"):
        _cfg(relational_routing="noisy_query")  # _cfg defaults to projected_gradient
    assert "relational_routing" not in tiny_config().inactive_fields()


def test_operators_carry_routing_without_new_state() -> None:
    from vfe3.model.relational_operators import RelationalOperators

    default = RelationalOperators([2, 2], 11, readout_mode="predictive_content")
    noisy = RelationalOperators([2, 2], 11, readout_mode="predictive_content",
                                routing="noisy_query")
    assert (default.routing, noisy.routing) == ("gaussian_kl", "noisy_query")
    assert set(noisy.state_dict()) == set(default.state_dict())
    with pytest.raises(ValueError, match="predictive_content"):
        RelationalOperators([2, 2], 11, readout_mode="full_link_update", routing="noisy_query")
    with pytest.raises(ValueError, match="routing"):
        RelationalOperators([2, 2], 11, readout_mode="predictive_content", routing="dot_product")


def test_relational_score_dispatches_on_the_operator_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.gradients.relational as relational
    from tests.test_relational_filtering import _fixture
    from vfe3.model.relational_operators import RelationalOperators

    mu, sigma, _, frame, _, _ = _fixture()
    called = []
    import vfe3.relational_scoring as scoring
    for name in ("noisy_query", "gaussian_kl"):
        monkeypatch.setitem(scoring.ROUTING_COSTS, name,
                            lambda *args, _name=name, **kwargs: called.append(_name) or torch.zeros(()))
    for routing in ("gaussian_kl", "noisy_query"):
        ops = RelationalOperators([2, 2], 11, readout_mode="predictive_content",
                                  routing=routing).double()
        relational.relational_score(mu, sigma, mu, sigma, frame, ops)
    assert called == ["gaussian_kl", "noisy_query"]


def test_run_record_names_the_routing() -> None:
    from tests.test_relational_integration import _cfg
    from vfe3.run_artifacts import _relational_artifact_fields

    cfg = _cfg(attention_mode="projected_predictive",
               relational_routing="noisy_query")
    assert _relational_artifact_fields(cfg)["relational_operators"]["routing"] == "noisy_query"


def test_noisy_query_model_trains_and_loads_gaussian_kl_checkpoints() -> None:
    from tests.test_relational_integration import _cfg, _tokens
    from vfe3.model.model import VFEModel

    predictive = dict(attention_mode="projected_predictive",
                      attention_residual=True)
    torch.manual_seed(23)
    kl_model = VFEModel(_cfg(**predictive))
    noisy_model = VFEModel(_cfg(**predictive, relational_routing="noisy_query"))
    noisy_model.load_state_dict(kl_model.state_dict())  # strict: identical parameter names
    ids, targets = _tokens()
    captures = []
    for model in (kl_model, noisy_model):
        cap = {}
        _, logits = model.forward_beliefs(ids, return_logits=True, capture=cap)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1))
        loss.backward()
        assert torch.isfinite(loss)
        for parameter in model.relational_operators.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        captures.append(cap["relational_capture"])
    beta_kl, beta_noisy = (capture["beta_pre"] for capture in captures)
    assert not torch.allclose(beta_kl, beta_noisy)
    torch.testing.assert_close(beta_noisy.sum(-1), torch.ones_like(beta_noisy.sum(-1)))
    # D_ij >= log det(I + V_Q^{-1/2} V_K V_Q^{-1/2}) / 2 > 0 for every scored pair, so a zero
    # would mean the lower clamp had silently absorbed a negative cost.
    assert bool((captures[1]["score_energy_pre"] > 0).all())
