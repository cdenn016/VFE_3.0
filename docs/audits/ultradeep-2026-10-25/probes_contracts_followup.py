"""Coordinator-only CPU checks after refining targets and configuration reachability."""
from __future__ import annotations

import json
from pathlib import Path
import runpy
from typing import Any

scope = runpy.run_path(str(Path(__file__).with_name("probes_contracts_artifacts.py")))
torch, test_config, VFEModel = scope["torch"], scope["test_config"], scope["VFEModel"]
attempt = scope["attempt"]


def inference_loss(tokens: list[int], mode: str) -> None:
    model = VFEModel(test_config(vocab_size=7, embed_dim=4, n_heads=2, n_layers=1,
                                max_seq_len=3, use_block_mlp=False)).eval()
    ids = torch.tensor([tokens])
    scope = torch.inference_mode() if mode == "inference" else torch.no_grad()
    with scope:
        model(ids, targets=ids)


def rope_model_transition() -> None:
    model = VFEModel(test_config(vocab_size=7, embed_dim=4, n_heads=2, n_layers=1,
                                max_seq_len=3, use_block_mlp=False, rope_mode="frame"))
    ids = torch.tensor([[1, 2, 3]])
    model.eval()
    with torch.inference_mode():
        model(ids)
    model.train()
    _, loss, _ = model(ids, targets=ids)
    loss.backward()


def main() -> None:
    torch.set_num_threads(1)
    scope["warnings"].filterwarnings("ignore")
    torch.manual_seed(7)
    results: dict[str, Any] = {
        "loss_modes": {f"{mode}_{tokens}": attempt(lambda t=tokens, m=mode: inference_loss(t, m))
                       for mode in ("no_grad", "inference")
                       for tokens in ([1, 1, 2], [1, 2, 3])},
        "rope_model_transition": attempt(rope_model_transition),
        "parameter_counts": attempt(scope["parameter_counts"]),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
