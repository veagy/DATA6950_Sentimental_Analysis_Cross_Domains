"""
Core building blocks for DBN: RBM and GaussianRBM.
Reference: docs/deep-learning/dbm/dbm.md
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple

from .._base import DBNModuleBase


class RBM(DBNModuleBase):
    """
    Binary-Binary Restricted Boltzmann Machine.
    Energy: E(v,h) = -b^T v - a^T h - h^T W v
    Conditional: P(h|v)=sigmoid(Wv+a), P(v|h)=sigmoid(W^T h+b)
    """

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        learning_rate: float = 0.01,
        cd_k: int = 1,
        momentum: float = 0.5,
        weight_decay: float = 0.0,
        persist: bool = False,
        sample_method: str = "bernoulli",
        gradient_clip: Optional[float] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.cd_k = cd_k
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.persist = persist
        self.sample_method = sample_method
        self.gradient_clip = gradient_clip
        self.factory_kwargs = {"device": device, "dtype": dtype}

        std = 0.01
        self.W = nn.Parameter(torch.empty(n_hidden, n_visible, **self._get_factory_kwargs()))
        self.b = nn.Parameter(torch.zeros(n_visible, **self._get_factory_kwargs()))
        self.a = nn.Parameter(torch.zeros(n_hidden, **self._get_factory_kwargs()))
        nn.init.normal_(self.W, 0, std)

        self._momentum_W = None
        self._momentum_b = None
        self._momentum_a = None
        self._positive_h = None

    def _get_factory_kwargs(self) -> dict:
        kw = dict(self.factory_kwargs)
        if kw.get("device") is None:
            kw["device"] = torch.device("cpu")
        if kw.get("dtype") is None:
            kw["dtype"] = torch.float32
        return {k: v for k, v in kw.items() if v is not None}

    def sample_hidden(self, v: torch.Tensor) -> torch.Tensor:
        """P(h=1|v) = sigmoid(Wv + a). Returns probabilities or sampled binary."""
        pre = F.linear(v, self.W, self.a)
        p = torch.sigmoid(pre)
        if self.training and self.sample_method == "bernoulli":
            return torch.bernoulli(p.clamp(1e-7, 1 - 1e-7))
        return p

    def sample_visible(self, h: torch.Tensor) -> torch.Tensor:
        """P(v=1|h) = sigmoid(W^T h + b). Returns probabilities or sampled binary."""
        pre = F.linear(h, self.W.t(), self.b)
        p = torch.sigmoid(pre)
        if self.training and self.sample_method == "bernoulli":
            return torch.bernoulli(p.clamp(1e-7, 1 - 1e-7))
        return p

    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """G(v) = -b^T v - sum_j log(1 + exp(a_j + W_j v))."""
        wx_b = F.linear(v, self.W, self.a)
        return -v @ self.b - F.softplus(wx_b).sum(dim=-1)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Bottom-up: return hidden probabilities (or samples in train)."""
        return self.sample_hidden(v)

    def cd_step(self, v: torch.Tensor, k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Contrastive Divergence step. Returns (v_neg, h_neg) from k-step Gibbs.
        If persist=True, uses persistent chains.
        """
        k = k if k is not None else self.cd_k
        h_pos = self.sample_hidden(v)
        v_neg = v
        h_neg = h_pos

        if self.persist and self._positive_h is not None and self._positive_h.shape[0] == v.shape[0]:
            h_neg = self._positive_h

        for _ in range(k):
            v_neg = self.sample_visible(h_neg)
            h_neg = self.sample_hidden(v_neg)

        if self.persist:
            self._positive_h = h_neg.detach()

        return v_neg, h_neg

    def reconstruct(self, v: torch.Tensor, n_steps: int = 1) -> torch.Tensor:
        """Reconstruct visible from hidden via n_steps Gibbs steps."""
        h = self.sample_hidden(v)
        for _ in range(n_steps):
            v_recon = self.sample_visible(h)
            h = self.sample_hidden(v_recon)
        return self.sample_visible(h)


class GaussianRBM(DBNModuleBase):
    """
    Gaussian-Bernoulli RBM for continuous visible, binary hidden.
    E(X,H) = ||X-b||^2/(2 sigma^2) - c^T H - X^T W H / sigma^2
    P(H_j=1|X) = sigmoid(W_j X / sigma^2 + c_j)
    P(X_i|H) = N(b_i + sigma^2 sum_j w_ij H_j, sigma^2)
    """

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        sigma: float = 1.0,
        learn_sigma: bool = False,
        learning_rate: float = 0.01,
        cd_k: int = 1,
        momentum: float = 0.5,
        weight_decay: float = 0.0,
        persist: bool = False,
        sample_method: str = "bernoulli",
        gradient_clip: Optional[float] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.sigma = sigma
        self.learn_sigma = learn_sigma
        self.learning_rate = learning_rate
        self.cd_k = cd_k
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.persist = persist
        self.sample_method = sample_method
        self.gradient_clip = gradient_clip
        self.factory_kwargs = {"device": device, "dtype": dtype}

        std = 0.01
        self.W = nn.Parameter(torch.empty(n_hidden, n_visible, **self._get_factory_kwargs()))
        self.b = nn.Parameter(torch.zeros(n_visible, **self._get_factory_kwargs()))
        self.c = nn.Parameter(torch.zeros(n_hidden, **self._get_factory_kwargs()))
        nn.init.normal_(self.W, 0, std)

        if learn_sigma:
            self.log_sigma = nn.Parameter(
                torch.tensor(math.log(sigma), **self._get_factory_kwargs())
            )
        else:
            self.register_buffer("log_sigma", torch.tensor(math.log(sigma)))

        self._momentum_W = None
        self._momentum_b = None
        self._momentum_c = None
        self._positive_h = None

    def _get_factory_kwargs(self) -> dict:
        kw = dict(self.factory_kwargs)
        if kw.get("device") is None:
            kw["device"] = torch.device("cpu")
        if kw.get("dtype") is None:
            kw["dtype"] = torch.float32
        return {k: v for k, v in kw.items() if v is not None}

    @property
    def sigma_val(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)

    def sample_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """P(H=1|X) = sigmoid(W X / sigma^2 + c)."""
        sigma2 = self.sigma_val**2
        pre = F.linear(x, self.W / sigma2, self.c)
        p = torch.sigmoid(pre)
        if self.training and self.sample_method == "bernoulli":
            return torch.bernoulli(p.clamp(1e-7, 1 - 1e-7))
        return p

    def sample_visible(self, h: torch.Tensor) -> torch.Tensor:
        """P(X|H) = N(b + sigma^2 W^T h, sigma^2). Returns mean (or sampled)."""
        sigma2 = self.sigma_val**2
        mean = self.b + sigma2 * F.linear(h, self.W.t())
        if self.training and self.sample_method == "bernoulli":
            return mean + self.sigma_val * torch.randn_like(mean, device=mean.device, dtype=mean.dtype)
        return mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Bottom-up: return hidden probabilities (or samples)."""
        return self.sample_hidden(x)

    def reconstruct(self, x: torch.Tensor, n_steps: int = 1) -> torch.Tensor:
        """Reconstruct visible from hidden."""
        h = self.sample_hidden(x)
        for _ in range(n_steps):
            x_recon = self.sample_visible(h)
            h = self.sample_hidden(x_recon)
        return self.sample_visible(h)
