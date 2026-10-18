"""
Agent 2: Compact Spatial Grid CNN (Spatial-Dueling-DQN).
Converts 134-dimensional observation into a 3x16x16 spatial tensor + continuous kinematics.
Winner of 88/88 stage clears!
"""

import os
import sys
import random
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import torch
import torch.nn as nn


def state_to_spatial_grid(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized conversion of 134-dim batch tensor into (B, 3, 16, 16) spatial tensor.
    - Channel 0: Bricks (top 8 rows of 16x16)
    - Channel 1: Paddle (Row 14)
    - Channel 2: Ball position
    """
    batch_size = x.shape[0]
    device = x.device

    kin = x[:, :6]
    bricks = x[:, 6:134]

    grid = torch.zeros((batch_size, 3, 16, 16), dtype=torch.float32, device=device)
    batch_idx = torch.arange(batch_size, device=device)

    # Channel 0: Bricks
    if bricks.shape[1] == 128:
        grid[:, 0, :8, :] = bricks.view(batch_size, 8, 16)

    # Channel 1: Paddle
    pad_norm_x = kin[:, 0]
    pad_col = ((pad_norm_x + 1.0) * 0.5 * 15.0).clamp(1, 14).long()
    grid[batch_idx, 1, 14, pad_col] = 1.0
    grid[batch_idx, 1, 14, pad_col - 1] = 0.8
    grid[batch_idx, 1, 14, pad_col + 1] = 0.8

    # Channel 2: Ball
    ball_norm_x = kin[:, 1]
    ball_norm_y = kin[:, 2]
    ball_col = ((ball_norm_x + 1.0) * 0.5 * 15.0).clamp(0, 15).long()
    ball_row = ((1.0 - ball_norm_y) * 0.5 * 15.0).clamp(0, 15).long()
    grid[batch_idx, 2, ball_row, ball_col] = 1.0

    return grid, kin


class SpatialDuelingDQN(nn.Module):
    """
    Compact Spatial Grid Convolutional Dueling Q-Network.
    """

    def __init__(self, in_channels: int = 3, num_actions: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
        )

        self.kin_net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        combined_dim = 128 + 64

        self.value_stream = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid, kin = state_to_spatial_grid(x)
        spatial_feat = self.conv(grid)
        kin_feat = self.kin_net(kin)
        combined = torch.cat([spatial_feat, kin_feat], dim=1)

        val = self.value_stream(combined)
        adv = self.advantage_stream(combined)
        return val + (adv - adv.mean(dim=1, keepdim=True))


class Agent2CNNAgent:
    """Agent 2 Managing Spatial Dueling CNN."""

    def __init__(
        self,
        num_actions: int = 3,
        learning_rate: float = 1e-4,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.num_actions = num_actions
        self.device = torch.device(device)

        self.policy_net = SpatialDuelingDQN(in_channels=3, num_actions=num_actions).to(self.device)
        self.target_net = SpatialDuelingDQN(in_channels=3, num_actions=num_actions).to(self.device)
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
            "architecture": "spatial_cnn",
            "dueling": True,
            "meta": extra_meta or {},
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str) -> Dict[str, Any]:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        return checkpoint.get("meta", {})
