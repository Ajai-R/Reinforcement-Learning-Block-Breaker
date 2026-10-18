"""
Agent 3: Rainbow DQN (C51 Distributional + Dueling + Double + NoisyNet).
Combines:
1. Double Q-Learning (decoupled action selection and evaluation)
2. Prioritized Experience Replay (PER with SumTree, in prioritized_replay.py)
3. Dueling Network Architecture (Value stream + Advantage stream per atom)
4. Multi-step Learning (n-step TD target, in train.py)
5. Distributional RL (Categorical C51 with 51 discrete atoms over [V_min, V_max])
6. Noisy Networks for Exploration (Factorized Gaussian NoisyLinear layers)
"""

import math
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """
    Factorized Gaussian Noisy Linear Layer (Fortunato et al., 2017).
    Replaces epsilon-greedy exploration with learned parametric weight noise.
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters (mean and std dev)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Non-learnable noise buffers
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.register_buffer("noise_in", torch.empty(in_features))
        self.register_buffer("noise_out", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int, device: torch.device) -> torch.Tensor:
        x = torch.randn(size, device=device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        self.noise_in.copy_(self._scale_noise(self.in_features, self.weight_mu.device))
        self.noise_out.copy_(self._scale_noise(self.out_features, self.weight_mu.device))
        self.weight_epsilon.copy_(self.noise_out.outer(self.noise_in))
        self.bias_epsilon.copy_(self.noise_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


def state_to_spatial_grid(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized conversion of 134-dim batch tensor into (B, 3, 16, 16) spatial grid + 6-D kinematics.
    """
    batch_size = x.shape[0]
    device = x.device

    kin = x[:, :6]
    bricks = x[:, 6:134]

    grid = torch.zeros((batch_size, 3, 16, 16), dtype=torch.float32, device=device)
    batch_idx = torch.arange(batch_size, device=device)

    # Channel 0: Bricks (top 8 rows of 16x16)
    if bricks.shape[1] == 128:
        grid[:, 0, :8, :] = bricks.view(batch_size, 8, 16)

    # Channel 1: Paddle (Row 14)
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


class RainbowDuelingC51Network(nn.Module):
    """
    Full Rainbow Deep Q-Network:
    - Spatial CNN Stream (Conv2D x3 + MaxPool -> 128)
    - Continuous Kinematics Stream (Linear x2 -> 64)
    - Dueling Categorical Distribution Heads (51 Atoms over [V_min, V_max]):
      * Value stream V(s): NoisyLinear(192, 128) -> NoisyLinear(128, 51)
      * Advantage stream A(s, a): NoisyLinear(192, 128) -> NoisyLinear(128, 3 * 51)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_actions: int = 3,
        num_atoms: int = 51,
        v_min: float = -30.0,
        v_max: float = 150.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Fixed discrete support atoms: z_i = v_min + i * delta_z
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        # 1. Spatial Grid CNN Stream
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

        # 2. Continuous Kinematics Stream
        self.kin_net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        combined_dim = 128 + 64  # 192

        # 3. Dueling Value Distribution Stream (Outputs 51 atom logits)
        self.val_noisy1 = NoisyLinear(combined_dim, 128)
        self.val_noisy2 = NoisyLinear(128, num_atoms)

        # 4. Dueling Advantage Distribution Stream (Outputs num_actions * 51 atom logits)
        self.adv_noisy1 = NoisyLinear(combined_dim, 128)
        self.adv_noisy2 = NoisyLinear(128, num_actions * num_atoms)

    def reset_noise(self):
        """Resets factorized Gaussian noise in all NoisyLinear layers."""
        self.val_noisy1.reset_noise()
        self.val_noisy2.reset_noise()
        self.adv_noisy1.reset_noise()
        self.adv_noisy2.reset_noise()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            probs: (batch_size, num_actions, num_atoms) valid categorical probability distributions
            q_values: (batch_size, num_actions) expected scalar Q-values (sum(p_i * z_i))
        """
        batch_size = x.shape[0]
        grid, kin = state_to_spatial_grid(x)

        spatial_feat = self.conv(grid)
        kin_feat = self.kin_net(kin)
        combined = torch.cat([spatial_feat, kin_feat], dim=1)

        # Value stream: (B, 192) -> (B, 128) -> (B, num_atoms)
        val = F.relu(self.val_noisy1(combined))
        val_logits = self.val_noisy2(val).view(batch_size, 1, self.num_atoms)

        # Advantage stream: (B, 192) -> (B, 128) -> (B, num_actions, num_atoms)
        adv = F.relu(self.adv_noisy1(combined))
        adv_logits = self.adv_noisy2(adv).view(batch_size, self.num_actions, self.num_atoms)

        # Dueling aggregation per atom: Q_i(s, a) = V_i(s) + (A_i(s, a) - mean_a(A_i(s, a)))
        q_logits = val_logits + (adv_logits - adv_logits.mean(dim=1, keepdim=True))

        # Softmax over discrete atoms to produce valid probability distribution
        probs = F.softmax(q_logits, dim=-1)  # (B, num_actions, num_atoms)

        # Expected Q-values: dot product of probabilities with support atoms
        q_values = torch.sum(probs * self.support, dim=-1)  # (B, num_actions)

        return probs, q_values

    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Selects action using epsilon-greedy over expected Q-values (hybrid with NoisyNet)."""
        if epsilon > 0.0 and np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)
        with torch.no_grad():
            t_state = torch.as_tensor(state, dtype=torch.float32, device=self.support.device).unsqueeze(0)
            _, q_values = self.forward(t_state)
            return int(q_values.argmax(dim=1).item())
