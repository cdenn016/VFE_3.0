"""Validate the fix hypothesis: transport types + multi-step optimizer training."""
import sys, warnings, importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import vfe3.config as cfgmod

spec = importlib.util.spec_from_file_location("train_vfe3_live", ROOT / "train_vfe3.py")
mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
LIVE = dict(mm.config)

SRC = Path(cfgmod.__file__).read_text(encoding="utf-8")
NEEDLE = '                "pos_rotation": "none",\n'
exec(compile(SRC.replace(NEEDLE, ""), cfgmod.__file__, "exec"), cfgmod.__dict__)
VFE3Config = cfgmod.VFE3Config

from vfe3.model.model import VFEModel
from vfe3.geometry.transport import RopeTransport, CompactFactoredTransport, FactoredTransport
from vfe3.run_artifacts import _relational_artifact_fields

USER = dict(LIVE)
USER.update(vocab_size=7, embed_dim=8, n_heads=2, n_layers=1, max_seq_len=8,
            eval_stride=4, batch_size=2, max_steps=2)

print("=== transport type produced by RopeTransport.score_operator() ===")
cfg = VFE3Config(**{**USER, "pos_rotation": "rope"})
torch.manual_seed(17)
model = VFEModel(cfg)
ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 0]])
beliefs = model.prior_bank.encode(ids)
beliefs = beliefs._replace(phi=model._apply_pos_phi(beliefs.phi),
                           right_phi=model._pos_phi_right(beliefs.phi))
r = model._rope_rotation(8, ids.device)
t = model._build_shared_flat_transport(beliefs, r, require_vertex_factors=True)
print("  base build with rope ->", type(t).__name__)
if isinstance(t, RopeTransport):
    so = t.score_operator()
    print("  score_operator() ->", type(so).__name__)
    print("  same_frame_flat_cocycle:", getattr(so, "same_frame_flat_cocycle", "<n/a>"))
    f = model._canonical_frame_context(t, fold_rope=True, compact=True)
    print("  frame forward shape:", tuple(f.forward.shape),
          " block_diagonal:", getattr(f, "block_diagonal", "<n/a>"))
    fn = model._canonical_frame_context(t, fold_rope=False, compact=True)
    same = torch.equal(f.forward.to_dense() if hasattr(f.forward, "to_dense") else f.forward,
                       fn.forward.to_dense() if hasattr(fn.forward, "to_dense") else fn.forward)
    print("  frame WITH fold_rope == frame WITHOUT? ", same)

print("\n=== multi-step training with the wiring fix (5 optimizer steps) ===")
orig = VFEModel._build_shared_flat_transport
def spy(self, b, rope, **kw):
    if rope is None:
        rope = self._rope_rotation(b.mu.shape[-2], b.phi.device)
    return orig(self, b, rope, **kw)

for rot in ("none", "rope"):
    VFEModel._build_shared_flat_transport = spy if rot == "rope" else orig
    c = VFE3Config(**{**USER, "pos_rotation": rot})
    torch.manual_seed(17)
    mdl = VFEModel(c)
    opt = torch.optim.AdamW([p for p in mdl.parameters() if p.requires_grad], lr=1e-4)
    losses = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for step in range(5):
            opt.zero_grad()
            out = mdl.forward(ids, targets=torch.tensor([[1, 2, 3, 4, 5, 6, 0, 1]]))
            loss = out[1] if isinstance(out, tuple) else out
            loss.backward()
            opt.step()
            losses.append(round(float(loss.detach()), 6))
    print(f"  pos_rotation={rot!r} wired={rot=='rope'}  losses = {losses}")
    print("   relational artifact fields:",
          {k: v for k, v in _relational_artifact_fields(c).items()
           if k in ("relational_operators",)}.get("relational_operators", {}).get(
               "total_parameters", "?"))
VFEModel._build_shared_flat_transport = orig
