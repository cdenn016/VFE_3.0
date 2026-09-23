"""Independent Gaussian Q/K routing and optimization contracts."""
import pytest
import torch

from vfe3.gradients.relational import relational_score
from vfe3.model.relational_operators import RelationalOperators
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import _relational_artifact_fields
from vfe3.train import build_optimizer
from tests.test_relational_filtering import _fixture
from tests.test_relational_integration import _cfg, _tokens


def test_separate_maps_match_distribution_oracle_and_have_free_scales() -> None:
    mu, sigma, _, frame, _, _ = _fixture()
    ops = RelationalOperators([2, 2], 11, identity_content=True).double()
    S_q, S_k = ops.comparison_pair()
    assert S_q.data_ptr() != S_k.data_ptr()
    assert not torch.equal(S_q, S_k)
    assert {name for name, _ in ops.named_parameters()} == {"raw_score_q", "raw_score_k"}
    # Scaling Q is a learned degree of freedom, including at full rank.
    with torch.no_grad():
        ops.raw_score_q.mul_(1.3)
    S_q, S_k = ops.comparison_pair()
    actual = relational_score(mu, sigma, mu.detach(), sigma, frame, ops)
    expected = torch.empty_like(actual)
    for h in range(2):
        for i in range(3):
            for j in range(3):
                distributions = []
                for token, S in ((i, S_q[h]), (j, S_k[h])):
                    part = slice(2*h, 2*h+2)
                    L = S @ frame.inverse[0, token, part, part]
                    mean = L @ mu[0, token, part]
                    covariance = L @ torch.diag(sigma[0, token, part]) @ L.T
                    distributions.append(torch.distributions.MultivariateNormal(mean, covariance_matrix=covariance))
                expected[0, h, i, j] = torch.distributions.kl_divergence(*distributions)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    derivatives = torch.autograd.grad(actual.sum(), (ops.raw_score_q, ops.raw_score_k))
    assert all(g.abs().sum() > 1e-8 and torch.isfinite(g).all() for g in derivatives)
    assert actual.diagonal(dim1=-2, dim2=-1).max() > 0


def test_optimizer_updates_both_roles_and_reports_counts() -> None:
    cfg = _cfg(relational_identity_content=True)
    model = VFEModel(cfg)
    ops = model.relational_operators
    before = [p.detach().clone() for p in (ops.raw_score_q, ops.raw_score_k)]
    optimizer = build_optimizer(model, cfg)
    ids, targets = _tokens()
    _, logits = model.forward_beliefs(ids, return_logits=True)
    torch.nn.functional.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1)).backward()
    optimizer.step()
    for old, new in zip(before, (ops.raw_score_q, ops.raw_score_k)):
        assert not torch.equal(old, new)
    meta = _relational_artifact_fields(cfg)["relational_operators"]
    assert meta["parameter_counts"]["score"] == 2*2*2*2      # 2 roles x H x d_head x d_head
    assert meta["score_role_parameter_counts"] == {"q": 8, "k": 8}
    assert meta["content_mode"] == "identity"
