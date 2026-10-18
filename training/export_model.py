"""
Export PyTorch Checkpoint to Browser-Ready JSON Weights.

Converts saved .pth models (VectorDQN or DuelingVectorDQN) into a lightweight JSON file
(frontend/src/assets/model_weights.json) so the React UI can perform real-time,
in-browser neural network inference with zero Python dependencies!
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="Export PyTorch DQN checkpoint to JSON for web browser inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to input .pth checkpoint")
    parser.add_argument("--output", "--output-json", dest="output_json", type=str, default="../frontend/src/assets/model_weights.json", help="Path to output JSON weights file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load model on")
    return parser.parse_args()


def export():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve checkpoint path: check cwd first, then script_dir
    if os.path.isabs(args.checkpoint):
        checkpoint_path = args.checkpoint
    elif os.path.exists(args.checkpoint):
        checkpoint_path = os.path.abspath(args.checkpoint)
    elif os.path.exists(os.path.join(script_dir, args.checkpoint)):
        checkpoint_path = os.path.abspath(os.path.join(script_dir, args.checkpoint))
    elif os.path.exists(os.path.join(script_dir, "checkpoints", os.path.basename(args.checkpoint))):
        checkpoint_path = os.path.abspath(os.path.join(script_dir, "checkpoints", os.path.basename(args.checkpoint)))
    else:
        checkpoint_path = os.path.abspath(os.path.join(script_dir, args.checkpoint))

    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        print("Please train a model first using 'python training/train.py' before exporting.")
        sys.exit(1)

    print(f"[Exporter] Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)

    policy_state = checkpoint.get("policy_state_dict", checkpoint.get("policy_net_state_dict"))
    if policy_state is None:
        raise KeyError("Could not find policy state dict in checkpoint.")
    dueling = checkpoint.get("dueling", True)
    state_dim = checkpoint.get("state_dim", 134)
    num_actions = checkpoint.get("num_actions", 3)
    extra_meta = checkpoint.get("meta", checkpoint.get("extra_meta", {}))

    layers = {}
    for key, tensor in policy_state.items():
        arr = tensor.cpu().numpy()
        layers[key] = np.round(arr, 6).tolist()

    arch = checkpoint.get("architecture", "spatial_cnn" if "conv.0.weight" in policy_state else "dueling")
    export_data = {
        "format": "spatial_cnn_v1" if arch in ["spatial_cnn", "cnn"] else "vector_dqn_v1",
        "architecture": arch,
        "dueling": dueling,
        "state_dim": state_dim,
        "num_actions": num_actions,
        "metadata": extra_meta,
        "weights": layers,
    }



    # Resolve output path
    if os.path.isabs(args.output_json):
        output_path = args.output_json
    elif os.path.exists(os.path.dirname(os.path.abspath(args.output_json))):
        output_path = os.path.abspath(args.output_json)
    else:
        output_path = os.path.abspath(os.path.join(script_dir, args.output_json))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f)

    file_size_kb = os.path.getsize(output_path) / 1024.0
    print(f"[Export Success] Web weights saved to: {output_path} ({file_size_kb:.1f} KB)")

    # Also save a copy inside checkpoints/
    ckpt_copy = os.path.join(script_dir, "checkpoints", "model_weights.json")
    with open(ckpt_copy, "w", encoding="utf-8") as f:
        json.dump(export_data, f)
    print(f"[Export Success] Checkpoint copy saved to: {ckpt_copy}")


if __name__ == "__main__":
    export()
