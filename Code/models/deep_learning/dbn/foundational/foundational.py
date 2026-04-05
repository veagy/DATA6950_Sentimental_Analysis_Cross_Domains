"""
Foundational DBN architectures.
Reference: docs/deep-learning/dbm/dbm.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Union, List, Any
import math

from .._base import DBNModuleBase
from ..core.core import RBM, GaussianRBM


def _ensure_dataloader(data: Any, batch_size: int) -> DataLoader:
    """Wrap tensor or Dataset in DataLoader."""
    if isinstance(data, DataLoader):
        return data
    if isinstance(data, torch.Tensor):
        return DataLoader(data, batch_size=batch_size, shuffle=True)
    if hasattr(data, "__getitem__") and hasattr(data, "__len__"):
        return DataLoader(data, batch_size=batch_size, shuffle=True)
    raise TypeError("data must be Tensor, DataLoader, or Dataset-like")


class StandardDBN(DBNModuleBase):
    """
    Standard Deep Belief Network: stack of RBMs trained greedily via CD.
    """

    def __init__(
        self,
        layer_sizes: Optional[List[int]] = None,
        n_visible: Optional[int] = None,
        n_hidden: Optional[Union[int, List[int]]] = None,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        n_epochs: int = 100,
        cd_k: int = 1,
        momentum: float = 0.5,
        weight_decay: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        if layer_sizes is None:
            if n_visible is None or n_hidden is None:
                raise ValueError("Provide layer_sizes or (n_visible, n_hidden)")
            nh = n_hidden if isinstance(n_hidden, list) else [n_hidden]
            layer_sizes = [n_visible] + nh
        self.layer_sizes = layer_sizes
        self.n_visible = layer_sizes[0]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.cd_k = cd_k
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            rbm = RBM(
                layer_sizes[i],
                layer_sizes[i + 1],
                learning_rate=learning_rate,
                cd_k=cd_k,
                momentum=momentum,
                weight_decay=weight_decay,
                device=device,
                dtype=dtype,
            )
            self.layers.append(rbm)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Bottom-up pass through stack; returns top-layer representation."""
        h = x
        for rbm in self.layers:
            h = rbm.sample_hidden(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def reconstruct(self, x: torch.Tensor, n_steps: int = 1) -> torch.Tensor:
        """Reconstruct visible from hidden via top layer down."""
        h = self.encode(x)
        for i in range(len(self.layers) - 1, -1, -1):
            v = self.layers[i].sample_visible(h)
            if i > 0:
                h = self.layers[i - 1].sample_hidden(v)
            else:
                return v
        return v

    def sample(self, n_samples: int, n_steps: int = 10) -> torch.Tensor:
        """Gibbs sample from top layer down to visible."""
        n_top = self.layer_sizes[-1]
        h = torch.bernoulli(torch.full((n_samples, n_top), 0.5, **self._fk())).to(
            self._get_device_dtype()[0]
        )
        for i in range(len(self.layers) - 1, -1, -1):
            v = self.layers[i].sample_visible(h)
            if i > 0:
                h = self.layers[i - 1].sample_hidden(v)
            else:
                return v
        return v

    def _fk(self) -> dict:
        d, dt = self._get_device_dtype()
        return {"device": d, "dtype": dt}

    def _get_device_dtype(self):
        try:
            p = next(self.parameters())
            return p.device, p.dtype
        except StopIteration:
            return torch.device("cpu"), torch.float32

    def pretrain(self, data: Any) -> None:
        """Greedy layer-wise pre-training."""
        loader = _ensure_dataloader(data, self.batch_size)
        d, dt = self._get_device_dtype()
        for i, rbm in enumerate(self.layers):
            rbm.train()
            opt = torch.optim.SGD(
                rbm.parameters(),
                lr=rbm.learning_rate,
                momentum=rbm.momentum,
                weight_decay=rbm.weight_decay,
            )
            for epoch in range(self.n_epochs):
                for batch in loader:
                    if isinstance(batch, (list, tuple)):
                        v = batch[0]
                    else:
                        v = batch
                    v = v.to(d, dtype=dt)
                    with torch.no_grad():
                        inp = v
                        for j in range(i):
                            inp = self.layers[j].sample_hidden(inp)
                    v_neg, h_neg = rbm.cd_step(inp, self.cd_k)
                    h_pos = rbm.sample_hidden(inp)
                    dw = (inp.t() @ h_pos - v_neg.t() @ h_neg) / inp.size(0)
                    db = (inp - v_neg).mean(0)
                    da = (h_pos - h_neg).mean(0)
                    rbm.W.grad = -dw
                    rbm.b.grad = -db
                    rbm.a.grad = -da
                    opt.step()

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=True)


class DBNDNNHybrid(DBNModuleBase):
    """
    DBN-DNN Hybrid: DBN pre-training + unrolled DNN for supervised fine-tuning.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        n_classes: int,
        learning_rate: float = 0.01,
        supervised_epochs: int = 50,
        supervised_lr: float = 0.001,
        dropout: float = 0.0,
        batch_size: int = 32,
        cd_k: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.dbn_stack = StandardDBN(
            layer_sizes=layer_sizes,
            learning_rate=learning_rate,
            batch_size=batch_size,
            cd_k=cd_k,
            device=device,
            dtype=dtype,
        )
        self.n_classes = n_classes
        self.supervised_epochs = supervised_epochs
        self.supervised_lr = supervised_lr
        self.dropout_val = dropout
        self.batch_size = batch_size
        self._finetuned = False

        n_top = layer_sizes[-1]
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_top, n_classes),
        )
        self._init_classifier()

    def _init_classifier(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def pretrain(self, data: Any) -> None:
        self.dbn_stack.pretrain(data)

    def finetune(self, data: Any, labels: torch.Tensor) -> None:
        from torch.utils.data import TensorDataset
        if not isinstance(data, DataLoader):
            if isinstance(data, torch.Tensor) and isinstance(labels, torch.Tensor):
                loader = DataLoader(
                    TensorDataset(data, labels),
                    batch_size=self.batch_size,
                    shuffle=True,
                )
            else:
                raise TypeError("data must be DataLoader or (Tensor, Tensor) for finetune")
        else:
            loader = data
        opt = torch.optim.Adam(
            list(self.dbn_stack.parameters()) + list(self.classifier.parameters()),
            lr=self.supervised_lr,
        )
        self.train()
        dev = next(self.parameters()).device
        for epoch in range(self.supervised_epochs):
            for batch in loader:
                x, y = batch[0], batch[1]
                x = x.to(dev)
                y = y.to(dev).long()
                h = self.dbn_stack.encode(x)
                logits = self.classifier(h)
                loss = F.cross_entropy(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self._finetuned = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dbn_stack.encode(x)
        return self.classifier(h)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).argmax(dim=-1)


class GaussianDBN(DBNModuleBase):
    """
    Gaussian DBN: Gaussian-Bernoulli RBM in first layer, binary RBMs above.
    """

    def __init__(
        self,
        n_visible: Optional[int] = None,
        n_hidden: Optional[List[int]] = None,
        sigma: float = 1.0,
        layer_sizes: Optional[List[int]] = None,
        gaussian_first: bool = True,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        n_epochs: int = 100,
        cd_k: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        if layer_sizes is None:
            if n_visible is None or n_hidden is None:
                raise ValueError("Provide layer_sizes or (n_visible, n_hidden)")
            layer_sizes = [n_visible] + list(n_hidden)
        self.layer_sizes = layer_sizes
        self.n_visible = layer_sizes[0]
        self.gaussian_first = gaussian_first
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.gaussian_layer = None
        self.binary_layers = nn.ModuleList()
        if gaussian_first:
            self.gaussian_layer = GaussianRBM(
                layer_sizes[0],
                layer_sizes[1],
                sigma=sigma,
                learning_rate=learning_rate,
                cd_k=cd_k,
                device=device,
                dtype=dtype,
            )
            for i in range(2, len(layer_sizes)):
                self.binary_layers.append(
                    RBM(
                        layer_sizes[i - 1],
                        layer_sizes[i],
                        learning_rate=learning_rate,
                        cd_k=cd_k,
                        device=device,
                        dtype=dtype,
                    )
                )
        else:
            for i in range(len(layer_sizes) - 1):
                self.binary_layers.append(
                    RBM(
                        layer_sizes[i],
                        layer_sizes[i + 1],
                        learning_rate=learning_rate,
                        cd_k=cd_k,
                        device=device,
                        dtype=dtype,
                    )
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        if self.gaussian_layer is not None:
            h = self.gaussian_layer.sample_hidden(h)
        for rbm in self.binary_layers:
            h = rbm.sample_hidden(h)
        return h

    def reconstruct(self, x: torch.Tensor, n_steps: int = 1) -> torch.Tensor:
        """Reconstruct visible from top hidden; n_steps Gibbs on gaussian layer."""
        h = self.forward(x)
        for rbm in reversed(list(self.binary_layers)):
            h = rbm.sample_visible(h)
        if self.gaussian_layer is not None:
            for _ in range(n_steps):
                v = self.gaussian_layer.sample_visible(h)
                h = self.gaussian_layer.sample_hidden(v)
            return self.gaussian_layer.sample_visible(h)
        return self.binary_layers[0].reconstruct(h, n_steps) if self.binary_layers else h

    def sample(self, n_samples: int) -> torch.Tensor:
        n_top = self.layer_sizes[-1]
        d, dt = self._dd()
        h = torch.bernoulli(torch.full((n_samples, n_top), 0.5, device=d, dtype=dt))
        for rbm in reversed(list(self.binary_layers)):
            h = rbm.sample_visible(h)
        if self.gaussian_layer is not None:
            return self.gaussian_layer.sample_visible(h)
        return h

    def _dd(self):
        try:
            p = next(self.parameters())
            return p.device, p.dtype
        except StopIteration:
            return torch.device("cpu"), torch.float32


class ReluDBN(DBNModuleBase):
    """
    ReLU-DBN: Hidden units use ReLU/softplus instead of sigmoid.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = "relu",
        learning_rate: float = 0.01,
        batch_size: int = 32,
        n_epochs: int = 100,
        cd_k: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.cd_k = cd_k
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            lin = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            nn.init.xavier_uniform_(lin.weight, gain=0.5)
            nn.init.zeros_(lin.bias)
            self.layers.append(lin)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "sigmoid":
            return torch.sigmoid(x)
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "softplus":
            return F.softplus(x)
        return F.relu(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for lin in self.layers:
            h = self._act(lin(h))
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class SparseDBN(DBNModuleBase):
    """
    Sparse DBN: Sparsity regularization on hidden activations.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        sparsity_target: float = 0.05,
        sparsity_weight: float = 0.1,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        n_epochs: int = 100,
        cd_k: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.layer_sizes = layer_sizes
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.cd_k = cd_k
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                RBM(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    learning_rate=learning_rate,
                    cd_k=cd_k,
                    device=device,
                    dtype=dtype,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for rbm in self.layers:
            h = rbm.sample_hidden(h)
        return h

    def get_sparsity(self) -> dict:
        """Per-layer mean activations (for monitoring)."""
        return {"sparsity_target": self.sparsity_target}


class BayesianDBN(DBNModuleBase):
    """
    Bayesian DBN: Variational weight posteriors for uncertainty estimates.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        prior_scale: float = 1.0,
        num_samples: int = 10,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        n_epochs: int = 100,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(device=device, dtype=dtype)
        self.layer_sizes = layer_sizes
        self.prior_scale = prior_scale
        self.num_samples = num_samples
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.mu_layers = nn.ModuleList()
        self.logvar_layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
            mu = nn.Linear(n_in, n_out)
            nn.init.normal_(mu.weight, 0, 0.01)
            nn.init.zeros_(mu.bias)
            logvar = nn.Linear(n_in, n_out)
            nn.init.constant_(logvar.weight, -3.0)
            nn.init.constant_(logvar.bias, -3.0)
            self.mu_layers.append(mu)
            self.logvar_layers.append(logvar)

    def _sample_weights(self, i: int) -> torch.Tensor:
        mu = self.mu_layers[i].weight
        logvar = self.logvar_layers[i].weight
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(mu, device=mu.device, dtype=mu.dtype)

    def forward(self, x: torch.Tensor, n_samples: Optional[int] = None) -> torch.Tensor:
        n = n_samples or self.num_samples
        outs = []
        for _ in range(n):
            h = x
            for i in range(len(self.mu_layers)):
                W = self._sample_weights(i)
                b = self.mu_layers[i].bias
                h = torch.sigmoid(F.linear(h, W, b))
            outs.append(h)
        return torch.stack(outs, dim=0).mean(dim=0)

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 10
    ) -> tuple:
        outs = []
        for _ in range(n_samples):
            h = x
            for i in range(len(self.mu_layers)):
                W = self._sample_weights(i)
                b = self.mu_layers[i].bias
                h = torch.sigmoid(F.linear(h, W, b))
            outs.append(h)
        stacked = torch.stack(outs, dim=0)
        return stacked.mean(dim=0), stacked.std(dim=0)

    def kl_divergence(self) -> torch.Tensor:
        """KL(q || p) over all variational layers."""
        kl = 0.0
        for mu_lin, logvar_lin in zip(self.mu_layers, self.logvar_layers):
            mu, logvar = mu_lin.weight, logvar_lin.weight
            var = torch.exp(logvar)
            kl = kl + 0.5 * (mu.pow(2) + var - logvar - 1).sum() / (self.prior_scale**2)
        return kl
