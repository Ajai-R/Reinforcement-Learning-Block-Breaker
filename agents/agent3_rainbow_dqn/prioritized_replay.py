"""
Prioritized Experience Replay (PER) using Binary SumTree and MinTree (Schaul et al., 2015).
Provides O(log N) priority updates and sampling.
"""

import numpy as np
import torch


class SegmentTree:
    """Base Segment Tree data structure."""

    def __init__(self, capacity: int, operation, neutral_element: float):
        self.capacity = capacity
        self.operation = operation
        self.neutral_element = neutral_element
        self.tree = [neutral_element for _ in range(2 * capacity)]

    def _reduce(self, start: int, end: int) -> float:
        start += self.capacity
        end += self.capacity
        res = self.neutral_element
        while start < end:
            if start % 2 == 1:
                res = self.operation(res, self.tree[start])
                start += 1
            if end % 2 == 1:
                end -= 1
                res = self.operation(res, self.tree[end])
            start //= 2
            end //= 2
        return res

    def __setitem__(self, idx: int, val: float):
        idx += self.capacity
        self.tree[idx] = val
        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        return self.tree[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity: int):
        super().__init__(capacity=capacity, operation=lambda a, b: a + b, neutral_element=0.0)

    def total(self) -> float:
        return self.tree[1]

    def find_prefixsum_idx(self, prefixsum: float) -> int:
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if self.tree[left] > prefixsum:
                idx = left
            else:
                prefixsum -= self.tree[left]
                idx = left + 1
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity: int):
        super().__init__(capacity=capacity, operation=min, neutral_element=float("inf"))

    def min(self) -> float:
        return self.tree[1]


class PrioritizedReplayBuffer:
    """
    Proportional Prioritized Experience Replay Buffer (PER).
    Samples transitions proportionally to their TD error priority p^alpha.
    """

    def __init__(
        self,
        capacity: int = 100000,
        state_dim: int = 134,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        self.sum_tree = SumSegmentTree(capacity)
        self.min_tree = MinSegmentTree(capacity)
        self.max_priority = 1.0

        # Memory pre-allocated arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        idx = self.ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = 1.0 if done else 0.0

        # New transitions are inserted with max priority to guarantee they are visited at least once
        priority = self.max_priority ** self.alpha
        self.sum_tree[idx] = priority
        self.min_tree[idx] = priority

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        """
        Samples a batch with probability P(i) = p_i^alpha / sum(p_k^alpha)
        Returns importance sampling weights w_i = (N * P(i))^-beta / max(w)
        """
        self.frame += 1
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / max(1, self.beta_frames))

        indices = []
        p_total = self.sum_tree.total()
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            mass = np.random.uniform(a, b)
            idx = self.sum_tree.find_prefixsum_idx(mass)
            # Guard against edge-case out of bounds
            idx = min(idx, self.size - 1)
            indices.append(idx)

        # Compute importance sampling weights
        p_min = self.min_tree.min() / p_total
        max_weight = (p_min * self.size) ** (-beta) if p_min > 0 else 1.0

        weights = []
        for idx in indices:
            p_sample = self.sum_tree[idx] / p_total
            weight = (p_sample * self.size) ** (-beta)
            weights.append(weight / max_weight)

        weights = np.array(weights, dtype=np.float32).reshape(-1, 1)

        b_s = torch.as_tensor(self.states[indices], dtype=torch.float32, device=device)
        b_a = torch.as_tensor(self.actions[indices], dtype=torch.int64, device=device)
        b_r = torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=device)
        b_s_next = torch.as_tensor(self.next_states[indices], dtype=torch.float32, device=device)
        b_d = torch.as_tensor(self.dones[indices], dtype=torch.float32, device=device)
        b_w = torch.as_tensor(weights, dtype=torch.float32, device=device)

        return b_s, b_a, b_r, b_s_next, b_d, b_w, indices

    def update_priorities(self, indices, priorities):
        """Updates priorities in the SumTree and MinTree for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            p = float(priority) + 1e-6
            self.max_priority = max(self.max_priority, p)
            val = p ** self.alpha
            self.sum_tree[idx] = val
            self.min_tree[idx] = val

    def __len__(self):
        return self.size
