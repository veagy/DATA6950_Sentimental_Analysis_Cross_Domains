"""
Temporal DBN variants: Recurrent, Temporal, Conditional.
Reference: docs/deep-learning/dbm/dbm.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
import math

from .._base import DBNModuleBase
from ..core.core import RBM
from ..foundational.foundational import StandardDBN


class RDBN(DBNModuleBase):
    """
    Recurrent DBN: recurrent connections for temporal dependencies.
    """

    def __init__(
        self,
        n_visible: int,
        n_hidden: Union[int, List[int]],
        n_steps: int,
        recurrent_connections: str = "full",
        learning_rate: float = 0.01,
        batch_size: int = 32,
        cd_k: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        nh = n_hidden if isinstance(n_hidden, list) else [n_hidden]
        self.n_visible = n_visible
        self.n_hidden = nh[-1]
        self.n_steps = n_steps
        self.recurrent_connections = recurrent_connections
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.cd_k = cd_k
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.rbm = RBM(
            n_visible,
            self.n_hidden,
            learning_rate=learning_rate,
            cd_k=cd_k,
            device=device,
            dtype=dtype,
        )
        self.W_rec = nn.Parameter(
            torch.randn(self.n_hidden, self.n_hidden, **self._fk()) * 0.01
        )
        nn.init.orthogonal_(self.W_rec)

    def _fk(self) -> dict:
        d = torch.device("cpu")
        dt = torch.float32
        try:
            p = next(self.parameters())
            d, dt = p.device, p.dtype
        except StopIteration:
            pass
        return {"device": d, "dtype": dt}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n_v) -> (B, T, n_h)."""
        B, T, _ = x.shape
        h_prev = torch.zeros(B, self.n_hidden, **self._fk()).to(x.device)
        outs = []
        for t in range(T):
            v_t = x[:, t, :]
            pre = F.linear(v_t, self.rbm.W, self.rbm.a) + h_prev @ self.W_rec
            h_t = torch.sigmoid(pre)
            h_prev = h_t
            outs.append(h_t)
        return torch.stack(outs, dim=1)

    def sample_sequence(self, n_samples: int, T: int) -> torch.Tensor:
        """Sample temporal sequence."""
        h = torch.bernoulli(torch.full((n_samples, self.n_hidden), 0.5, **self._fk()))
        seq = []
        for _ in range(T):
            v = self.rbm.sample_visible(h)
            h = self.rbm.sample_hidden(v)
            h = h + 0.1 * (h @ self.W_rec)
            h = torch.sigmoid(h)
            seq.append(v)
        return torch.stack(seq, dim=1)

    @property
    def recurrent_weights(self):
        return self.W_rec


class TDBN(DBNModuleBase):
    """
    Temporal DBN: generative model for sequences.
    """

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        sequence_length: int,
        transition_type: str = "directed",
        learning_rate: float = 0.01,
        batch_size: int = 32,
        cd_k: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.sequence_length = sequence_length
        self.transition_type = transition_type
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.cd_k = cd_k
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.W_vh = nn.Parameter(torch.randn(n_hidden, n_visible, **self._fk()) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(n_hidden, n_hidden, **self._fk()) * 0.01)
        self.W_hv = nn.Parameter(torch.randn(n_visible, n_hidden, **self._fk()) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(n_hidden, **self._fk()))
        self.b_v = nn.Parameter(torch.zeros(n_visible, **self._fk()))

    def _fk(self) -> dict:
        d, dt = torch.device("cpu"), torch.float32
        try:
            p = next(self.parameters())
            d, dt = p.device, p.dtype
        except StopIteration:
            pass
        return {"device": d, "dtype": dt}

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """sequence: (B, T, n_v) -> (B, T, n_h)."""
        B, T, _ = sequence.shape
        h_prev = torch.zeros(B, self.n_hidden, **self._fk()).to(sequence.device)
        outs = []
        for t in range(T):
            v_prev = sequence[:, t - 1, :] if t > 0 else sequence[:, 0, :]
            pre = F.linear(v_prev, self.W_vh.t(), self.b_h) + h_prev @ self.W_hh.t()
            h_t = torch.sigmoid(pre)
            h_prev = h_t
            outs.append(h_t)
        return torch.stack(outs, dim=1)

    def generate(self, T: int, n_samples: int) -> torch.Tensor:
        """Generate sequence of length T."""
        h = torch.zeros(n_samples, self.n_hidden, **self._fk())
        v = torch.bernoulli(torch.full((n_samples, self.n_visible), 0.5, **self._fk()))
        seq = []
        for _ in range(T):
            pre = F.linear(v, self.W_vh.t(), self.b_h) + h @ self.W_hh.t()
            h = torch.sigmoid(pre)
            v = torch.sigmoid(F.linear(h, self.W_hv.t(), self.b_v))
            v = torch.bernoulli(v.clamp(1e-7, 1 - 1e-7))
            seq.append(v)
        return torch.stack(seq, dim=1)

    @property
    def transition_weights(self):
        return self.W_hh, self.W_vh


class ConditionalDBN(DBNModuleBase):
    """
    Conditional DBN: generation conditioned on labels or past frames.
    """

    def __init__(
        self,
        n_visible: int,
        n_hidden: Union[int, List[int]],
        condition_size: int,
        n_classes: Optional[int] = None,
        layer_sizes: Optional[List[int]] = None,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        cd_k: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        if layer_sizes is None:
            nh = n_hidden if isinstance(n_hidden, list) else [n_hidden]
            layer_sizes = [n_visible] + list(nh)
        self.layer_sizes = layer_sizes
        self.n_visible = n_visible
        self.condition_size = condition_size
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.cd_k = cd_k
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.dbns = nn.ModuleList()
        self.U_layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            rbm = RBM(
                layer_sizes[i],
                layer_sizes[i + 1],
                learning_rate=learning_rate,
                cd_k=cd_k,
                device=device,
                dtype=dtype,
            )
            self.dbns.append(rbm)
            U = nn.Linear(condition_size, layer_sizes[i + 1])
            nn.init.normal_(U.weight, 0, 0.01)
            nn.init.zeros_(U.bias)
            self.U_layers.append(U)

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        """Conditional encode: x (B,n_v), condition (B,n_c) -> (B, n_L)."""
        h = x
        for rbm, U in zip(self.dbns, self.U_layers):
            pre = F.linear(h, rbm.W, rbm.a) + U(condition)
            h = torch.sigmoid(pre)
        return h

    def generate(self, condition: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Conditional sample."""
        B = condition.size(0) if condition.dim() > 1 else n_samples
        if condition.dim() == 1:
            condition = condition.unsqueeze(0).expand(n_samples, -1)
        h = torch.bernoulli(
            torch.full(
                (n_samples, self.layer_sizes[-1]), 0.5,
                device=condition.device, dtype=condition.dtype,
            )
        )
        for i in range(len(self.dbns) - 1, -1, -1):
            v = self.dbns[i].sample_visible(h)
            if i > 0:
                cond_contrib = self.U_layers[i - 1](condition[:v.size(0)])
                pre = F.linear(v, self.dbns[i].W.t(), self.dbns[i].b) + cond_contrib
                h = torch.sigmoid(pre)
            else:
                return v
        return v

    @property
    def condition_weights(self):
        return self.U_layers
