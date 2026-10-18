"""
Agent 3: Export Rainbow DQN (C51 + Dueling + NoisyNet) weights to browser-compatible JSON.
"""

import os
import sys
import json
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))


def export(checkpoint_path: str = None, output_paths: list = None):
    checkpoints_dir = os.path.join(script_dir, "checkpoints")
    if checkpoint_path is None:
        best_path = os.path.join(checkpoints_dir, "best_model.pth")
        final_path = os.path.join(checkpoints_dir, "final_model.pth")
        checkpoint_path = best_path if os.path.exists(best_path) else final_path

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"[Rainbow Exporter] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_dict = ckpt.get("model_state_dict", ckpt.get("policy_state_dict", ckpt))
    meta = ckpt.get("meta", {})

    exportable_weights = {}
    total_params = 0

    for key, tensor in state_dict.items():
        if not key.startswith("target_") and not key.endswith("epsilon") and not key.startswith("noise_"):
            # Convert tensors to list for JSON serialization
            arr = tensor.detach().cpu().numpy()
            exportable_weights[key] = arr.tolist()
            total_params += arr.size

    num_atoms = int(meta.get("num_atoms", 51))
    v_min = float(meta.get("v_min", -30.0))
    v_max = float(meta.get("v_max", 150.0))

    support = [v_min + i * (v_max - v_min) / (num_atoms - 1) for i in range(num_atoms)]

    model_data = {
        "format": "rainbow_c51_dqn",
        "architecture": "Full Rainbow DQN (Spatial CNN + Kinematics + C51 Distributional + Dueling + NoisyNet)",
        "in_channels": 3,
        "dueling": True,
        "distributional": True,
        "num_atoms": num_atoms,
        "v_min": v_min,
        "v_max": v_max,
        "support": support,
        "state_dim": 134,
        "num_actions": 3,
        "total_parameters": total_params,
        "metadata": meta,
        "weights": exportable_weights,
    }

    if output_paths is None:
        output_paths = [
            os.path.join(checkpoints_dir, "model_weights_rainbow.json"),
            os.path.join(project_root, "frontend", "src", "assets", "model_weights_rainbow.json"),
        ]

    for p in output_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(model_data, f)
        print(f"✅ Exported Rainbow model to: {p} ({os.path.getsize(p) / 1024:.1f} KB)")


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    export(ckpt)
