"""One CPU integration run with alternate in-memory experiment settings."""
import inspect
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
Path(".test-output/config-isolation").mkdir(parents=True, exist_ok=True)
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import pytest
from vfe3.config import VFE3Config
from tests.profiles import config_values

changes = dict(vocab_size=23, embed_dim=8, n_heads=4, max_seq_len=11,
               n_layers=3, family="gaussian_full", decode_mode="full_chunked",
               norm_type_block="layernorm", use_block_mlp=False,
               block_mlp_expansion=3, decode_bias=False, untie_decode_bank=False,
               decode_temperature_learned=True, decode_unigram_prior=False,
               predictive_covariance="passthrough", attention_mode="transported",
               attention_residual=False, relational_output_scope="head",
               e_step_update="mm_exact", e_step_gradient="detach",
               lambda_alpha_mode="constant", sigma_init=0.4, pos_phi="none",
               m_decode_mu_lr=None, m_decode_sigma_lr=None, m_pos_phi_lr=None,
               phi_weight_decay=0.0, decode_av_precision="fp32",
               decode_ranking_refinement="full", frame_pair_fp64_escalation=True)
constructor = VFE3Config.__init__
names = list(inspect.signature(constructor).parameters)[1:]
defaults = list(constructor.__defaults__)
for name, value in changes.items():
    defaults[names.index(name)] = value
    setattr(VFE3Config, name, value)
    VFE3Config.__dataclass_fields__[name].default = value
constructor.__defaults__ = tuple(defaults)

import train_vfe3
import ablation
import scaling
train_vfe3.config = config_values(**changes)
ablation.BASELINE_CONFIG = config_values(**changes)
scaling.config = config_values(**changes)
scaling.BASELINE = scaling.config

raise SystemExit(pytest.main([
    "tests", "-m", "not cuda and not external", "--tb=short",
    "--basetemp=.test-output/config-isolation/alternate-temp",
    "--junitxml=.test-output/config-isolation/alternate-full.xml",
]))
