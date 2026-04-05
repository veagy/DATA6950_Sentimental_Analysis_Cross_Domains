"""
Phase 5: Paradigm Reinforcement Learning.
Provides native numpy contiguous ReplayBuffers, dynamic Welford running metrics and Frame stacks.
"""

import numpy as np
import torch
from collections import deque


# -----------------------------------------------------------------------------
# 1. RUNNING ONLINE WIERMANN-WELFORD SCALARS
# -----------------------------------------------------------------------------

class RunningNormalizer:
    """
    Online Welford's continuous mean and variance numerical scalarization exactly preserving 
    dynamic limits implicitly bounding RL observation inputs efficiently explicitly without O(N) memory sizes.
    """
    def __init__(self, shape: tuple, clip: float = 10.0):
        self.mean = np.zeros(shape, dtype="float64")
        self.var = np.ones(shape, dtype="float64")
        self.count = 0
        self.clip = clip

    def update(self, obs: np.ndarray):
        """Updates limits mathematically integrating incoming scalar representations exactly."""
        batch_mean = obs.mean(axis=0)
        batch_var = obs.var(axis=0)
        batch_n = obs.shape[0]
        
        if batch_n == 0:
            return
            
        total_n = self.count + batch_n
        delta = batch_mean - self.mean
        
        self.mean += delta * batch_n / total_n
        
        # M2 recursive scalar updates mathematically robust representation boundaries
        m_a = self.var * self.count
        m_b = batch_var * batch_n
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_n / total_n
        
        self.var = M2 / total_n
        self.count = total_n

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Standardizer loop representing standard gaussian mappings bound natively by explicit clipping limits."""
        return np.clip(
            (obs - self.mean) / (np.sqrt(self.var) + 1e-8),
            -self.clip, self.clip
        )


class RewardScaler:
    """
    Discounting scalar executing boundaries matching discounted reward sequences scaling identically without executing offset derivations explicitly preserving zeroes natively.
    """
    def __init__(self, gamma: float = 0.99, clip: float = 10.0):
        self.norm = RunningNormalizer((1,), clip=float("inf"))
        self.gamma = gamma
        self.ret = 0.0
        self.clip = clip

    def scale(self, reward: float, done: bool) -> float:
        self.ret = reward + self.gamma * self.ret * (1 - done)
        self.norm.update(np.array([[self.ret]]))
        std = np.sqrt(self.norm.var[0] + 1e-8)
        return float(np.clip(reward / std, -self.clip, self.clip))


# -----------------------------------------------------------------------------
# 2. CONTINUOUS ACTION-OBSERVATION BUFFERS
# -----------------------------------------------------------------------------

class ReplayBuffer:
    """
    Pre-allocated continuous contiguous FIFO mapping memory array explicitly bound purely to Numpy natively.
    Avoids O(N) PyTorch allocation tensor copies entirely over massive million-scale environments perfectly.
    """
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        # Explicit contiguous buffers
        self.obs = np.zeros((capacity, obs_dim), dtype="float32")
        self.actions = np.zeros((capacity, action_dim), dtype="float32")
        self.rewards = np.zeros((capacity, 1), dtype="float32")
        self.next_obs = np.zeros((capacity, obs_dim), dtype="float32")
        self.dones = np.zeros((capacity, 1), dtype="float32")

    def push(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str = "cpu") -> dict:
        """Generates random PyTorch bounded representations efficiently converting memory safely."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.from_numpy(self.obs[idx]).to(device),
            "actions": torch.from_numpy(self.actions[idx]).to(device),
            "rewards": torch.from_numpy(self.rewards[idx]).to(device),
            "next_obs": torch.from_numpy(self.next_obs[idx]).to(device),
            "dones": torch.from_numpy(self.dones[idx]).to(device)
        }


# -----------------------------------------------------------------------------
# 3. ATARI-STYLE SEQUENCE FRAME STACKING
# -----------------------------------------------------------------------------

class FrameStack:
    """Resolves partial environment observability boundaries securely by structurally extending input dimension bounds identically scaling history context matrices explicitly."""
    def __init__(self, n: int = 4):
        self.n = n
        self.frames = deque(maxlen=n)

    def reset(self, obs: np.ndarray) -> np.ndarray:
        self.frames.clear()
        for _ in range(self.n):
            self.frames.append(obs)
        return np.concatenate(list(self.frames), axis=0)

    def step(self, obs: np.ndarray) -> np.ndarray:
        self.frames.append(obs)
        return np.concatenate(list(self.frames), axis=0)
