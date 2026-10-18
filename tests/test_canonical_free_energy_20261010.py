r"""The canonical free energy as an opt-in log-step diagnostic (audit 2026-10-10, item 12; round-2 #22).

Tiny CPU models only (K = 4, two heads, N = 4, vocab 9). ``diagnose_canonical_free_energy`` evaluates, at each
layer's input beliefs, live prior and realized frames,

.. math::
    F_i = \alpha_iD(q_i\|p_i) + R(\alpha_i) + \lambda_\beta\sum_h\Big[\sum_j\beta^{*h}_{ij}D^h_{ij}
          + \tau_h\sum_j\beta^{*h}_{ij}\log\frac{\beta^{*h}_{ij}}{\pi^h_{ij}}\Big],
    \qquad D^h_{ij} = \mathrm{KL}\big(q^h_i\,\|\,(\Omega_{ij}q_j)^h\big),

and these tests pin it against an independent double loop over pairs with dense float64 matrices, the envelope
identity, the off path, the families and the metrics.csv wiring.
"""
from pathlib import Path
import csv
import logging
import math
from typing import Any, Dict, List, Optional

import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.free_energy as fe
import vfe3.run_artifacts as ra
import vfe3.train as T
from tests.profiles import test_config, tiny_config
from vfe3.data.datasets import TokenWindows
from vfe3.geometry.lie_ops import CompactBlockElement
from vfe3.model.model import DiagnosticSnapshot, VFEModel

F_KEYS = ("self_coupling", "self_divergence", "belief_coupling", "attention_entropy", "total")
TOKENS = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])


def _model(**overrides: Any) -> VFEModel:
    values: Dict[str, Any] = dict(vocab_size=9, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=2,
                                  phi_scale=0.4, diagnose_canonical_free_energy=True)
    values.update(overrides)
    torch.manual_seed(3)
    return VFEModel(tiny_config(**values)).eval()


def _dense_kl(mu_q: torch.Tensor, C_q: torch.Tensor, mu_t: torch.Tensor, C_t: torch.Tensor) -> float:
    r"""KL(N(mu_q, C_q) || N(mu_t, C_t)) with an explicit inverse and log-determinants."""
    precision = torch.linalg.inv(C_t)
    diff = mu_t - mu_q
    return 0.5 * float(torch.trace(precision @ C_q) + diff @ precision @ diff - mu_q.numel()
                       + torch.linalg.slogdet(C_t)[1] - torch.linalg.slogdet(C_q)[1])


def _brute_force(model: VFEModel, snapshot: DiagnosticSnapshot, layer: int) -> Dict[str, float]:
    r"""Per-token means of the canonical free energy of ``layer`` by a double loop over pairs (float64)."""
    cfg, group = model.cfg, model.group
    q = snapshot.initial_belief if layer == 0 else snapshot.layer_outputs[layer - 1]
    mu_p, sigma_p = snapshot.layer_priors[layer]
    B, N, K = q.mu.shape
    dense = lambda s: torch.diag_embed(s.double()) if s.dim() == 3 else s.double()   # (B, N, K, K)
    Sigma, Sigma_p = dense(q.sigma), dense(sigma_p)
    start = snapshot.initial_belief
    U = torch.linalg.matrix_exp(group.embed_algebra(start.phi.double(), dtype=torch.float64))
    if start.right_phi is not None:
        U = U @ torch.linalg.matrix_exp(group.embed_algebra(start.right_phi.double(), dtype=torch.float64))
    V = torch.linalg.inv(U)                                               # (B, N, K, K)
    if snapshot.rope is not None:
        # The folded pair (U_i R_i^T, R_i U_i^{-1}) of fold_rope_into_frame: the float32 rotation is
        # orthogonal only to ~4e-8, so inv(U_i R_i^T) would differ from the realized inverse at that level.
        V = snapshot.rope.double() @ V
    ops = model.relational_operators_by_layer[layer]
    delta = ops.routing_log_temperature
    dims = list(group.irrep_dims)
    tau = [cfg.kappa_beta * math.sqrt(d) * (1.0 if delta is None else math.exp(3.0 * math.tanh(float(delta[h]) / 3.0)))
           for h, d in enumerate(dims)]
    log_prior = snapshot.log_prior.double()
    while log_prior.dim() < 4:
        log_prior = log_prior.unsqueeze(0)
    beta_model = snapshot.layer_conditional_captures[layer]["beta_pre"].double()   # (B, H, N, N)
    mode = cfg.lambda_alpha_mode
    self_terms, raw_terms, coupling, entropy, tv = [], [], [], [], []
    for b in range(B):
        xi = [V[b, i] @ q.mu[b, i].double() for i in range(N)]
        C = [V[b, i] @ Sigma[b, i] @ V[b, i].T for i in range(N)]
        for i in range(N):
            mu_q, mu_t = q.mu[b, i].double(), mu_p[b, i].double()
            if mode == "state_dependent_per_coord":                     # diagonal family: one KL per coordinate
                s_q, s_t = q.sigma[b, i].double(), sigma_p[b, i].double()
                D_k = 0.5 * (s_q / s_t + (mu_t - mu_q) ** 2 / s_t - 1.0 + torch.log(s_t / s_q))
                alpha = cfg.c0 / (cfg.b0 + D_k)
                self_terms.append(float((alpha * D_k + cfg.b0 * alpha - cfg.c0 * torch.log(alpha)).sum()))
                raw_terms.append(float(D_k.sum()))
            else:
                D_self = _dense_kl(mu_q, Sigma[b, i], mu_t, Sigma_p[b, i])
                alpha = cfg.c0 / (cfg.b0 + D_self)                       # state_dependent
                self_terms.append(alpha * D_self + cfg.b0 * alpha - cfg.c0 * math.log(alpha))
                raw_terms.append(D_self)
            row_coupling, row_entropy = 0.0, 0.0
            offset = 0
            for h, d in enumerate(dims):
                block = slice(offset, offset + d)
                offset += d
                bias = log_prior[0 if log_prior.shape[0] == 1 else b, 0 if log_prior.shape[1] == 1 else h, i]
                allowed = [j for j in range(N) if math.isfinite(float(bias[j]))]
                pi = torch.softmax(bias[allowed], dim=0)
                D = torch.tensor([_dense_kl(xi[i][block], C[i][block, block], xi[j][block], C[j][block, block])
                                  for j in allowed], dtype=torch.float64)
                weights = pi * torch.exp(-D / tau[h])
                beta = weights / weights.sum()
                row_coupling += float((beta * D).sum())
                row_entropy += tau[h] * float((beta * torch.log(beta / pi)).sum())
                if len(allowed) >= 2:
                    star = torch.zeros(N, dtype=torch.float64)
                    star[allowed] = beta
                    tv.append(0.5 * float((beta_model[b, h, i] - star).abs().sum()))
            coupling.append(row_coupling)
            entropy.append(row_entropy)
    mean = lambda values: sum(values) / len(values)
    out = {"self_coupling": mean(self_terms), "self_divergence": mean(raw_terms),
           "belief_coupling": mean(coupling), "attention_entropy": mean(entropy), "canonical_beta_tv": mean(tv)}
    out["total"] = out["self_coupling"] + cfg.lambda_beta * (out["belief_coupling"] + out["attention_entropy"])
    return out


@pytest.mark.parametrize("overrides", [
    dict(),                                                               # block_glk frames, nonzero phi
    dict(gauge_transport="off"),                                          # U = I
    dict(rope_mode="frame_value", beta_attention_prior="causal_alibi"),       # RoPE folded, per-head prior
    dict(lambda_alpha_mode="state_dependent_per_coord", c0=0.7, b0=1.3),
    dict(lambda_alpha_mode="state_dependent", lambda_beta=0.6, kappa_beta=1.7),
    dict(learn_attention_temperature=True),
    dict(family="gaussian_full", beta_attention_prior="causal_noself"),
    dict(attention_mode="projected_gradient", predictive_covariance="passthrough",
         relational_receiver_update="euler",
         attention_residual=False),
], ids=["frames", "gauge_off", "rope_alibi", "per_coord_alpha", "state_alpha", "learned_tau", "full_noself",
        "gradient"])
def test_matches_independent_brute_force(overrides: Dict[str, Any]) -> None:
    model = _model(**overrides)
    if model.cfg.learn_attention_temperature:
        with torch.no_grad():
            for ops in model.relational_operators_by_layer:
                ops.routing_log_temperature.copy_(torch.tensor([0.4, -0.3]))
    snapshot = model.build_diagnostic_snapshot(TOKENS)
    per_layer = model.diagnostics_per_layer(TOKENS, snapshot=snapshot)
    last = model.diagnostics(TOKENS, snapshot=snapshot)
    for layer in range(model.cfg.n_layers):
        expected = _brute_force(model, snapshot, layer)
        for key, value in expected.items():
            assert per_layer[key][layer] == pytest.approx(value, rel=1e-9, abs=1e-12), (layer, key)
        assert per_layer["relational_legacy_global_f_available"][layer] == 1.0
    for key in (*F_KEYS, "canonical_beta_tv"):
        assert last[key] == per_layer[key][-1], key
    if model.cfg.gauge_transport == "on":
        assert per_layer["self_divergence"][0] == pytest.approx(0.0, abs=1e-12)   # layer 0 reads q = p
        assert per_layer["belief_coupling"][-1] > 0.0
    # The float64 frames are the realized routing frames of the forward (float32 rounding apart).
    start = snapshot.initial_belief
    _, frame, _ = model._relational_frames(start, snapshot.rope)
    inverse = (CompactBlockElement(frame.inverse, model.cfg.embed_dim).to_dense()
               if frame.block_diagonal else frame.inverse)
    U = torch.linalg.matrix_exp(model.group.embed_algebra(start.phi.double(), dtype=torch.float64))
    if start.right_phi is not None:
        U = U @ torch.linalg.matrix_exp(model.group.embed_algebra(start.right_phi.double(), dtype=torch.float64))
    if snapshot.rope is not None:
        U = U @ snapshot.rope.double().transpose(-1, -2)
    assert torch.allclose(torch.linalg.inv(U), inverse.double(), atol=1e-5)


def test_envelope_identity_and_query_tiling(monkeypatch: pytest.MonkeyPatch) -> None:
    generator = torch.Generator().manual_seed(11)
    B, N, K, dims = 2, 5, 5, [2, 3]
    mu = torch.randn(B, N, K, generator=generator, dtype=torch.float64)
    root = torch.randn(B, N, K, K, generator=generator, dtype=torch.float64)
    sigma = root @ root.transpose(-1, -2) + 0.3 * torch.eye(K, dtype=torch.float64)
    frame_inverse = torch.eye(K, dtype=torch.float64) + 0.4 * torch.randn(B, N, K, K, generator=generator,
                                                                          dtype=torch.float64)
    causal = torch.full((N, N), float("-inf"), dtype=torch.float64).triu(1)      # (N, N) causal mask
    noself = causal.clone().fill_diagonal_(float("-inf"))
    log_prior = torch.stack((causal - 0.5 * torch.arange(N).double(), noself))   # (H, N, N): alibi-like, noself
    tau = torch.tensor([0.7, 1.9], dtype=torch.float64)
    rows = fe.canonical_routing_rows(mu, sigma, frame_inverse, log_prior, tau=tau, irrep_dims=dims)
    live = torch.isfinite(rows.log_partition)                                    # (B, H, N)
    assert not bool(live[:, 1, 0].any()) and bool(live[:, 1, 1:].all()) and bool(live[:, 0].all())
    envelope = rows.belief_coupling + rows.attention_entropy                     # (B, H, N)
    reduced = -tau.reshape(1, -1, 1) * rows.log_partition
    assert torch.allclose(envelope[live], reduced[live], rtol=1e-12, atol=1e-12)
    assert torch.equal(envelope[~live], torch.zeros_like(envelope[~live]))     # an empty row contributes zero
    assert torch.allclose(rows.beta.sum(dim=-1)[live], torch.ones(int(live.sum()), dtype=torch.float64))
    assert torch.equal(rows.beta[:, 1, 0], torch.zeros_like(rows.beta[:, 1, 0]))
    # The Gibbs row minimizes A(beta) = sum beta D + tau sum beta log(beta/pi): every other row costs more.
    pi = torch.softmax(log_prior[0, 3, :4], dim=0)
    D, beta_star = rows.energy[0, 0, 3, :4], rows.beta[0, 0, 3, :4]
    cost = lambda beta: float((beta * D).sum() + tau[0] * (beta * torch.log(beta / pi)).sum())
    for weights in (torch.tensor([0.4, 0.3, 0.2, 0.1]), torch.full((4,), 0.25), 0.9 * beta_star + 0.025):
        assert cost(weights.double() / weights.sum()) > cost(beta_star)
    # One query row per Mahalanobis tile gives the same energies.
    monkeypatch.setattr(fe, "_CANONICAL_QUERY_TILE_ELEMENTS", 1)
    tiled = fe.canonical_routing_rows(mu, sigma, frame_inverse, log_prior, tau=tau, irrep_dims=dims)
    assert torch.allclose(tiled.energy, rows.energy, rtol=1e-13, atol=1e-13)


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def test_off_leaves_columns_nan_and_every_other_value_bitwise_unchanged() -> None:
    off, on = _model(diagnose_canonical_free_energy=False), _model()
    for name, tensor in off.state_dict().items():
        assert torch.equal(tensor, on.state_dict()[name]), name
    added = {"canonical_beta_tv", "canonical_self_weight", "canonical_total_output", "canonical_total_delta",
             "canonical_self_clamped_frac"}                                     # audit 2026-10-17 item 33
    replaced = {*F_KEYS, "relational_legacy_global_f_available"}
    for d_off, d_on in ((off.diagnostics(TOKENS), on.diagnostics(TOKENS)),
                        (off.diagnostics_per_layer(TOKENS), on.diagnostics_per_layer(TOKENS))):
        assert set(d_on) == set(d_off) | added and not added & set(d_off)
        for key in F_KEYS:
            values = d_off[key] if isinstance(d_off[key], list) else [d_off[key]]
            assert all(math.isnan(value) for value in values), key
        assert d_off["relational_legacy_global_f_available"] in (0.0, [0.0, 0.0])
        for key in set(d_off) - replaced:
            assert _same(d_off[key], d_on[key]), key


@pytest.mark.parametrize("overrides", [
    dict(),
    dict(family="gaussian_full"),
    dict(family="gaussian_block"),
    dict(gauge_group="so_n", group_n=3, irrep_spec=[("l1", 2)], embed_dim=6, phi_scale=0.3),
], ids=["diagonal", "full", "block", "so_n"])
def test_finite_on_the_families(overrides: Dict[str, Any]) -> None:
    model = _model(**overrides)
    per_layer = model.diagnostics_per_layer(TOKENS)
    for key in (*F_KEYS, "canonical_beta_tv"):
        assert all(math.isfinite(value) for value in per_layer[key]), key
    assert all(0.0 <= value <= 1.0 for value in per_layer["canonical_beta_tv"])
    assert per_layer["relational_legacy_global_f_available"] == [1.0, 1.0]


def _train_run(tmp_path: Path, flag: bool) -> tuple[List[Dict[str, str]], List[dict]]:
    B, N, V = 2, 4, 7
    cfg = test_config(vocab_size=V, embed_dim=4, n_heads=2, batch_size=B, max_seq_len=N, n_layers=2,
                      warmup_steps=0, max_steps=4, log_interval=0, eval_interval=0, checkpoint_interval=0,
                      generate_embedding_figures=False, attention_mode="projected_predictive",
                      attention_residual=True, phi_scale=0.3, diagnose_canonical_free_energy=flag)
    torch.manual_seed(5)
    model = VFEModel(cfg)
    recorded: List[dict] = []
    real = model.diagnostics

    def spy(token_ids: torch.Tensor, *, snapshot: Optional[DiagnosticSnapshot] = None, **kwargs: Any) -> dict:
        d = real(token_ids, snapshot=snapshot, **kwargs)
        if snapshot is None:                                              # the training log step
            recorded.append(dict(d))
        return d

    model.diagnostics = spy
    loaders = [DataLoader(TokenWindows(torch.arange(41) % V, N), batch_size=B, shuffle=False),
               DataLoader(TokenWindows((torch.arange(23) * 3) % V, N, pad_final=True), batch_size=2,
                          shuffle=False)]
    run_dir = tmp_path / ("on" if flag else "off")
    artifacts = ra.RunArtifacts(run_dir, cfg, model, save_model_artifacts=False)
    torch.manual_seed(11)
    T.train(model, loaders[0], cfg, n_steps=4, grad_clip=1.0, log_interval=1, eval_interval=2,
            val_loader=loaders[1], device=torch.device("cpu"), logger=logging.getLogger("canonical-fe"),
            artifacts=artifacts, generate_samples=False)
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle)), recorded


def test_metrics_csv_carries_the_per_token_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    rows, recorded = _train_run(tmp_path, True)
    assert len(rows) == len(recorded) == 4
    columns = {"self_coupling": "self_coupling", "self_divergence": "self_divergence",
               "belief_coupling": "belief_coupling", "attention_entropy": "attention_entropy",
               "inner_alignment_energy_total": "total", "canonical_beta_tv": "canonical_beta_tv",
               "relational_legacy_global_f_available": "relational_legacy_global_f_available"}
    for row, d in zip(rows, recorded):
        for column, key in columns.items():
            assert float(row[column]) == d[key], column                  # per-token means, not divided again
        evaluated = int(row["step"]) % 2 == 0
        for column in ("val_canonical_beta_tv", "val_self_coupling", "val_inner_alignment_energy_total"):
            assert (row[column] != "") == evaluated, (row["step"], column)
            if evaluated:
                assert math.isfinite(float(row[column]))
        if evaluated:
            assert float(row["val_relational_legacy_global_f_available"]) == 1.0
    off_rows, _ = _train_run(tmp_path, False)
    assert "canonical_beta_tv" not in off_rows[0] and "val_canonical_beta_tv" not in off_rows[0]
    added = ("canonical_beta_tv", "canonical_self_weight", "canonical_total_output", "canonical_total_delta",
             "canonical_self_clamped_frac")
    assert set(rows[0]) == set(off_rows[0]) | {*added, *(f"val_{key}" for key in added)}
    assert all(row[column] == "" for row in off_rows for column in ("self_coupling", "val_self_coupling"))
