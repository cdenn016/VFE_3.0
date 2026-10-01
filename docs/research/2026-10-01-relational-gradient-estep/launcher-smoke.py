"""Read-only launcher tracing and tiny CPU execution of its full-link conversion."""
import ast
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer

torch.set_num_threads(1)
canonical = Path(r"C:/Users/chris and christine/Desktop/VFE_3.0")
out = Path("docs/research/2026-10-01-relational-gradient-estep")
protected_names = ("train_vfe3.py", "vfe3/config.py", "ablation.py", "scaling.py", "zzzzz.py")
hashes = {name: hashlib.sha256((canonical / name).read_bytes()).hexdigest() for name in protected_names}
tree = ast.parse((canonical / "train_vfe3.py").read_text(encoding="utf-8-sig"))
assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "config" for target in node.targets))
values = {}
expressions = {}
for keyword in assignment.value.keywords:
    try:
        values[keyword.arg] = ast.literal_eval(keyword.value)
    except ValueError:
        expressions[keyword.arg] = ast.unparse(keyword.value)
        if keyword.arg != "vocab_size":
            raise
# Match the launcher's post-dict override before constructing the runtime config.
values["kl_max"] = 8 * values["embed_dim"]
values["vocab_size"] = 7
active = VFE3Config(**values)  # validate, but do not construct the production model
route_overrides = dict(attention_mode="projected_gradient",
                       predictive_covariance="passthrough",
                       lambda_alpha_mode="constant",
                       relational_receiver_noise=1.0)
tiny_overrides = dict(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=3,
                      batch_size=1, eval_stride=None, n_layers=2, block_mlp_expansion=1,
                      max_steps=2, warmup_steps=0, generate_embedding_figures=False,
                      checkpoint_interval=0, eval_interval=0, log_interval=0,
                      kl_max=32.0)
config = VFE3Config(**{**values, **route_overrides, **tiny_overrides})
model = VFEModel(config).cpu()
optimizer = build_optimizer(model, config)
ids = torch.tensor([[0, 1, 2]], device="cpu")
targets = torch.tensor([[1, 2, 3]], device="cpu")
before_maps = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
               if name.startswith("relational_operators")}
with torch.autocast("cpu", dtype=torch.bfloat16):
    _, loss, _ = model(ids, targets)
loss.backward()
map_gradients = {}
for name, parameter in model.named_parameters():
    if name.startswith("relational_operators"):
        assert torch.equal(before_maps[name], parameter.detach())
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        map_gradients[name] = float(parameter.grad.norm())
assert torch.isfinite(loss)
optimizer.step()
assert any(not torch.equal(before_maps[name], parameter.detach())
           for name, parameter in model.named_parameters() if name in before_maps)
after_hashes = {name: hashlib.sha256((canonical / name).read_bytes()).hexdigest() for name in protected_names}
assert after_hashes == hashes, "canonical launcher/config WIP changed during this read-only smoke"
summary = dict(
    device="cpu", torch_version=torch.__version__, threads=torch.get_num_threads(),
    active_launcher={key: getattr(active, key) for key in (
        "attention_mode", "family", "n_layers", "n_e_steps", "e_q_mu_lr", "e_q_sigma_lr",
        "lambda_alpha_mode", "predictive_covariance", "e_mu_q_trust", "e_sigma_q_trust")},
    omitted_dynamic_expressions=expressions,
    route_overrides=route_overrides, tiny_size_overrides=tiny_overrides,
    executed_config=asdict(config), finite_ce=float(loss.detach()), map_gradient_norms=map_gradients,
    optimizer_relational_groups=[dict(role=group.get("lr_aux_role"), lr=group["lr"],
                                      parameter_count=sum(p.numel() for p in group["params"]))
                                 for group in optimizer.param_groups
                                 if str(group.get("lr_aux_role", "")).startswith("relational_")],
    canonical_sha256=hashes, canonical_preserved=True,
    result="tiny forward/backward and one outer optimizer update, no production training")
(out / "launcher-smoke.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({key: summary[key] for key in (
    "device", "threads", "finite_ce", "optimizer_relational_groups", "canonical_preserved")}, indent=2))
