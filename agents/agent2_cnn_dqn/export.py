"""
Export Agent 2 (Spatial CNN-DQN) weights to JSON for React browser inference.
"""

import os
import sys
import json
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def export(
    checkpoint_path="checkpoints/best_model.pth",
    output_path="../../frontend/src/assets/model_weights_cnn.json"
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_file = checkpoint_path if os.path.isabs(checkpoint_path) else os.path.join(script_dir, checkpoint_path)
    out_file = output_path if os.path.isabs(output_path) else os.path.join(script_dir, output_path)

    if not os.path.exists(ckpt_file):
        print(f"❌ Checkpoint not found: {ckpt_file}")
        return

    print(f"[Agent 2 Exporter] Loading: {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    policy_state = ckpt.get("policy_state_dict", ckpt.get("policy_net_state_dict", {}))

    layers = {k: np.round(v.cpu().numpy(), 6).tolist() for k, v in policy_state.items()}
    export_data = {
        "format": "spatial_cnn_v1",
        "agent_name": "Agent 2: Spatial CNN-DQN (88/88 Winner)",
        "architecture": "spatial_cnn",
        "num_actions": 3,
        "metadata": ckpt.get("meta", {}),
        "weights": layers,
    }

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f)
    print(f"✅ Exported Agent 2 to: {out_file} ({os.path.getsize(out_file) / 1024:.1f} KB)")

if __name__ == "__main__":
    export()
