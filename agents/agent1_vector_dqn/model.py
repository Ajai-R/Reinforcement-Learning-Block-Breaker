"""
Agent 1: 134-Dimensional Dueling Vector MLP Q-Network.
Processes 6 continuous kinematic features + 128 brick layout indicators through separate streams.
"""

import os
import sys
import random
from typing import Dict, Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn


class DuelingVectorDQN(nn.Module):
    """
    Two-stream Dueling MLP architecture:
    - Kinematic Stream: 6 coordinates -> Linear(6, 128) -> ReLU -> Linear(128, 128) -> ReLU
    - Brick Stream: 128 brick states -> Linear(128, 128) -> ReLU -> Linear(128, 128) -> ReLU
    - Fusion: Concatenates kinematic (128) + brick (128) = 256 features
    - Value Stream: 256 -> 128 -> 1
    - Advantage Stream: 256 -> 128 -> 3
    """

    def __init__(
        self,
        state_dim: int = 134,
        num_actions: int = 3,
        kin_dim: int = 128,
        brick_h1: int = 64,
        brick_h2: int = 32,
    ):
        super().__init__()
        self.kin_stream = nn.Sequential(
            nn.Linear(6, kin_dim),
            nn.ReLU(),
            nn.Linear(kin_dim, kin_dim),
            nn.ReLU(),
        )
        self.brick_stream = nn.Sequential(
            nn.Linear(128, brick_h1),
            nn.ReLU(),
            nn.Linear(brick_h1, brick_h2),
            nn.ReLU(),
        )
        fused_dim = kin_dim + brick_h2

        self.value_stream = nn.Sequential(
            nn.Linear(fused_dim, kin_dim),
            nn.ReLU(),
            nn.Linear(kin_dim, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(fused_dim, kin_dim),
            nn.ReLU(),
            nn.Linear(kin_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kin = x[:, :6]
        bricks = x[:, 6:]
        k_feat = self.kin_stream(kin)
        b_feat = self.brick_stream(bricks)
        combined = torch.cat([k_feat, b_feat], dim=1)

        val = self.value_stream(combined)
        adv = self.advantage_stream(combined)
        return val + (adv - adv.mean(dim=1, keepdim=True))


class Agent1DQNAgent:
    """Agent 1 Managing Dueling Vector DQN."""

    def __init__(
        self,
        state_dim: int = 134,
        num_actions: int = 3,
        hidden_dim: int = 128,
        learning_rate: float = 1e-4,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.device = torch.device(device)

        self.policy_net = DuelingVectorDQN(state_dim, num_actions, hidden_dim).to(self.device)
        self.target_net = DuelingVectorDQN(state_dim, num_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        with torch.no_grad():
            st = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.policy_net(st).argmax(dim=1).item())

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath: str, extra_meta: Optional[Dict[str, Any]] = None):
        checkpoint = {
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "state_dim": self.state_dim,
            "num_actions": self.num_actions,
            "architecture": "vector_dqn",
            "dueling": True,
            "meta": extra_meta or {},
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str) -> Dict[str, Any]:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        return checkpoint.get("meta", {})
