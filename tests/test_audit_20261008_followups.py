"""Follow-ups of the 2026-10-08 audit: chunked positional Fisher operators, measured SPD clearance,
the structure-constant budget of the pullback M-step, and the scaling-fit size threshold. Tiny CPU."""
import warnings

import pytest
import torch

from tests.profiles import test_config


def _fisher_step(monkeypatch: pytest.MonkeyPatch, chunk_bytes: int) -> torch.Tensor:
    import vfe3.geometry.phi_fisher as phi_fisher
    from tests.test_phi_fisher import _fisher_cfg
    from vfe3.model.model import VFEModel
    from vfe3.train import build_optimizer, train_step

    monkeypatch.setattr(phi_fisher, "POOLED_OPERATOR_CHUNK_BYTES", chunk_bytes)
    cfg = _fisher_cfg(m_phi_group_retraction="chart")
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg)
        optimizer = build_optimizer(model, cfg)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        tokens = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3], [3, 2, 1, 0, 3, 2, 1, 0]])
        train_step(model, optimizer, scheduler, tokens, torch.roll(tokens, -1, dims=1), grad_clip=1.0)
    return model.pos_phi_free.detach().clone()


def _initial_pos_phi() -> torch.Tensor:
    from tests.test_phi_fisher import _fisher_cfg
    from vfe3.model.model import VFEModel

    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFEModel(_fisher_cfg(m_phi_group_retraction="chart")).pos_phi_free.detach().clone()


def test_positional_fisher_operators_chunked_by_rows_are_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    """P2-22: forming and solving the pooled operators one row at a time gives the same step bitwise."""
    whole = _fisher_step(monkeypatch, 2**40)
    one_row = _fisher_step(monkeypatch, 1)                     # every chunk is a single row
    assert torch.equal(whole, one_row)
    assert not torch.equal(whole, _initial_pos_phi())                     # the step moved the positional rows


def test_pooled_operator_chunks_match_the_dense_operator() -> None:
    from vfe3.geometry.phi_fisher import FisherRowStatistics, pooled_operator_chunks, pooled_pos_operator

    g = torch.Generator().manual_seed(1)
    T, H, d, R, B = 3, 2, 3, 5, 4
    A = torch.randn(T, H, d, d, generator=g, dtype=torch.float64)
    C0 = A @ A.transpose(-1, -2) + torch.eye(d, dtype=torch.float64)
    xi0 = torch.randn(T, H, d, generator=g, dtype=torch.float64)
    gather = torch.randint(0, T, (R, B), generator=g)
    stats = FisherRowStatistics(kind="pos", count=torch.full((R,), float(B), dtype=torch.float64),
                                total=torch.tensor(9.0, dtype=torch.float64),
                                pooled_moments=(xi0, C0, torch.linalg.inv(C0), gather))
    dense = pooled_pos_operator(xi0[gather], C0[gather], torch.linalg.inv(C0)[gather], rotation_weight=2.0)
    chunks = list(pooled_operator_chunks(stats, rotation_weight=2.0, chunk_bytes=1))
    assert [(a, b) for a, b, _ in chunks] == [(r, r + 1) for r in range(R)]
    assert torch.equal(torch.cat([op for _, _, op in chunks]), dense)


def test_spd_repair_lands_near_eps_and_stays_certified() -> None:
    """P2-25: the measured clearance lifts lambda_min to about eps instead of ~1700 eps."""
    from vfe3.geometry.retraction import _certify_public_spd

    g = torch.Generator().manual_seed(0)
    K, eps, sigma_max = 32, 1e-6, 100.0
    Q = torch.linalg.qr(torch.randn(K, K, generator=g, dtype=torch.float64))[0]
    lam = torch.cat([torch.tensor([1e-8], dtype=torch.float64),
                     torch.linspace(1.0, 90.0, K - 1, dtype=torch.float64)])
    out = _certify_public_spd(((Q * lam) @ Q.T).float()[None], eps=eps, sigma_max=sigma_max)[0].double()
    lam_min = float(torch.linalg.eigvalsh(out)[0])
    assert eps < lam_min < 10 * eps
    eye = torch.eye(K, dtype=torch.float64)
    assert int(torch.linalg.cholesky_ex(out - eps * eye)[1]) == 0
    assert int(torch.linalg.cholesky_ex(sigma_max * eye - out)[1]) == 0


def test_pullback_structure_budget_replaces_the_flat_k12_cap() -> None:
    """geometer#2: block_glk is admitted to d_head 16 and refused at 17, at config time and at runtime."""
    from vfe3.geometry.phi_preconditioner import _strict_structure_constants

    base = dict(n_heads=2, gauge_group="block_glk", pos_phi="none", m_phi_update_mode="pullback_group",
                transport_chart_max_norm=6.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_config(embed_dim=32, **base)                                     # d_head = 16
        with pytest.raises(ValueError, match="structure-constant workspace"):
            test_config(embed_dim=34, **base)                                 # d_head = 17
        with pytest.raises(ValueError, match="structure-constant workspace"):
            test_config(embed_dim=32, **{**base, "gauge_tying": "shared"})    # 256 generators on 32 x 32
    with pytest.raises(ValueError, match="structure-constant workspace"):
        _strict_structure_constants(torch.zeros(4, 3, 3, dtype=torch.float64), structure_bytes=10)


def test_offset_law_needs_five_sizes_and_the_width_sweep_has_five() -> None:
    """gap-stats: the three-parameter offset law is testable from five sizes; the shipped sweep has five."""
    import scaling
    import scaling_analysis

    from vfe3.metric_contracts import scaling_min_testable_sizes

    assert scaling_min_testable_sizes(True) == 5 and scaling_min_testable_sizes(False) == 4
    assert scaling_analysis._min_testable_sizes() == scaling_min_testable_sizes(
        bool(scaling_analysis.CONFIG["with_offset"]))
    assert len(set(scaling.SWEEPS["widths"])) >= scaling_min_testable_sizes(True)
