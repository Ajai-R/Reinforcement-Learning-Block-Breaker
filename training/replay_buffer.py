"""
Experience Replay Buffer for 134-Dimensional Vector DQN Breakout.

Stores transitions (s, a, r, s', done) in pre-allocated contiguous arrays for
maximum throughput and zero GPU-CPU memory fragmentation.
"""

from typing import Tuple, Union
import numpy as np
import torch


class ReplayBuffer:
    """
    Fixed-capacity circular replay buffer storing 1D vector transitions.
    """

    def __init__(self, capacity: int = 50000, state_dim: int = 134):
        self.capacity = capacity
        self.state_dim = state_dim
        self.ptr = 0
        self.size = 0

        # Pre-allocated arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.uint8)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Adds a transition to the buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = 1 if done else 0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, device: Union[str, torch.device] = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a random mini-batch of transitions as PyTorch tensors.

        Returns:
            states: (batch_size, 134) float32
            actions: (batch_size, 1) int64
            rewards: (batch_size, 1) float32
            next_states: (batch_size, 134) float32
            dones: (batch_size, 1) float32
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        states_t = torch.from_numpy(self.states[indices]).float().to(device)
        actions_t = torch.from_numpy(self.actions[indices]).long().unsqueeze(1).to(device)
        rewards_t = torch.from_numpy(self.rewards[indices]).float().unsqueeze(1).to(device)
        next_states_t = torch.from_numpy(self.next_states[indices]).float().to(device)
        dones_t = torch.from_numpy(self.dones[indices]).float().unsqueeze(1).to(device)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def can_sample(self, batch_size: int) -> bool:
        """Returns True if the buffer contains at least batch_size samples."""
        return self.size >= batch_size

    def __len__(self) -> int:
        return self.size
