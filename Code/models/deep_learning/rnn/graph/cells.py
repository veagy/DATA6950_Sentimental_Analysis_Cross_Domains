"""Graph RNN cells: GraphRecurrentUnitCell, DynamicGraphRecurrentUnitCell."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Callable, List, Tuple, Dict
from .....models.utils import DLModule

from ..base import RNNCell, LSTMCell, GRUCell


class GraphRecurrentUnitCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 non_linearity: Union[str, Callable, nn.Module] = 'tanh',
                 funcs: Union[List, Tuple, Dict] = None,
                 bias: bool = True,
                 proj_size: int = None,
                 gcn_type: str = "gcn",
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.gcn_type = gcn_type.lower()

        self.Wx = nn.Linear(input_size, hidden_size, bias=bias, **self.factory_kwargs)
        self.Wh = nn.Linear(hidden_size, hidden_size, bias=bias, **self.factory_kwargs)

        rnn_input_dim = hidden_size
        cell_kwargs = {
            "input_size": rnn_input_dim,
            "hidden_size": hidden_size,
            "non_linearity": non_linearity,
            "funcs": funcs,
            "bias": bias,
            "proj_size": proj_size,
            **self.factory_kwargs,
            **kwargs,
        }

        if "input_size" in kwargs:
            del kwargs["input_size"]
        if "proj_size" in cell_kwargs:
            del cell_kwargs["proj_size"]

        for k in ["cell_type", "gcn_type", "sim_type", "dist_type", "p", "gamma", "beta", "max_iter", "num_nodes",
                  "dynamic_neighbours"]:
            if k in cell_kwargs:
                del cell_kwargs[k]

        cell_type = kwargs.get('cell_type', kwargs.get('rnn_type', 'gru'))
        if isinstance(cell_type, str):
            cell_type = cell_type.lower()
        self.cell_type = cell_type

        if cell_type == "lstm":
            self.cell = LSTMCell(**cell_kwargs)
        elif cell_type == "gru":
            self.cell = GRUCell(**cell_kwargs)
        else:
            self.cell = RNNCell(**cell_kwargs)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, h_prev: Union[torch.Tensor, Tuple] = None):
        if adj.dim() == 2:
            x_agg = torch.matmul(adj, x)
        else:
            x_agg = torch.bmm(adj, x)

        if self.cell_type == 'lstm':
            h, c = h_prev if h_prev is not None else (None, None)
        else:
            h = h_prev

        x_agg = self.Wx(x_agg)
        h_agg = None
        if h is not None:
            if adj.dim() == 2:
                h_agg = torch.matmul(adj, h)
            else:
                h_agg = torch.bmm(adj, h)
            h_agg = self.Wh(h_agg)
        else:
            B, N, _ = x.shape
            h_agg = torch.zeros(B, N, self.hidden_size, **self.factory_kwargs)
            if self.cell_type == 'lstm':
                c = torch.zeros(B, N, self.hidden_size, **self.factory_kwargs)

        B, N, F = x_agg.shape
        x_flat = x_agg.reshape(B * N, -1)
        h_flat = h_agg.reshape(B * N, -1)

        if self.cell_type == 'lstm':
            c_flat = c.reshape(B * N, -1) if c is not None else None
            h_out_flat, c_out_flat = self.cell(x_flat, h_flat, c_flat)
            h_out = h_out_flat.reshape(B, N, -1)
            c_out = c_out_flat.reshape(B, N, -1)
            return h_out, c_out
        else:
            h_out_flat = self.cell(x_flat, h_flat)
            h_out = h_out_flat.reshape(B, N, -1)
            return h_out


class DynamicGraphRecurrentUnitCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 non_linearity: Union[str, Callable, nn.Module] = 'tanh',
                 funcs: Union[List, Tuple, Dict] = None,
                 bias: bool = True,
                 proj_size: int = None,
                 sim_type: str = "attn",
                 dist_type: str = "dist",
                 p: int = 2,
                 gamma: float = 1.0,
                 beta: float = None,
                 max_iter: int = 100,
                 soft_threshold: float = 0.5,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.soft_threshold = soft_threshold

        self.sim_kwargs = {
            "sim_type": sim_type.lower(),
            "dist_type": dist_type.lower(),
            "p": p,
            "gamma": gamma,
            "beta": beta,
            "max_iter": max_iter
        }

        self.Wq = nn.Linear(hidden_size, hidden_size, bias=bias, **self.factory_kwargs)
        self.Wk = nn.Linear(hidden_size, hidden_size, bias=bias, **self.factory_kwargs)
        self.Wx = nn.Linear(input_size, hidden_size, bias=bias, **self.factory_kwargs)
        self.Wh = nn.Linear(hidden_size, hidden_size, bias=bias, **self.factory_kwargs)

        rnn_input_dim = hidden_size
        if "input_size" in kwargs:
            del kwargs["input_size"]

        cell_kwargs = {
            "input_size": rnn_input_dim,
            "hidden_size": hidden_size,
            "non_linearity": non_linearity,
            "funcs": funcs,
            "bias": bias,
            "proj_size": proj_size,
            **self.factory_kwargs,
            **kwargs,
        }
        if "proj_size" in cell_kwargs:
            del cell_kwargs["proj_size"]

        for k in ["cell_type", "gcn_type", "sim_type", "dist_type", "p", "gamma", "beta", "max_iter", "soft_threshold",
                  "num_nodes", "dynamic_neighbours"]:
            if k in cell_kwargs:
                del cell_kwargs[k]

        cell_type = kwargs.get('cell_type', kwargs.get('rnn_type', 'gru'))
        if isinstance(cell_type, str):
            cell_type = cell_type.lower()
        self.cell_type = cell_type

        if cell_type == "lstm":
            self.cell = LSTMCell(**cell_kwargs)
        elif cell_type == "gru":
            self.cell = GRUCell(**cell_kwargs)
        else:
            self.cell = RNNCell(**cell_kwargs)

    def _calc_attn_sim(self, h: torch.Tensor):
        d_sqrt = math.sqrt(self.hidden_size)
        Q = self.Wq(h)
        K = self.Wk(h)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / d_sqrt
        scores = F.softmax(scores, dim=-1)
        scores = 2 * scores - 1
        return scores

    def _calc_dist(self, h: torch.Tensor, p: int = 2):
        Q = self.Wq(h)
        K = self.Wk(h)
        dist_sim = torch.cdist(Q, K, p=p)
        return dist_sim

    def _calc_mahalanobis_dist(self, h: torch.Tensor):
        Q = self.Wq(h)
        K = self.Wk(h)
        B, N, H = Q.size()
        comb = torch.cat([Q, K], dim=-2)
        comb_centered = comb - comb.mean(dim=-2, keepdim=True)
        cov = torch.bmm(comb_centered.transpose(1, 2), comb_centered) / (2 * N - 1)
        eps = 1e-6 * torch.eye(H, device=Q.device).unsqueeze(0)
        cov = cov + eps
        inv_cov = torch.linalg.inv(cov)
        term1 = (torch.matmul(Q, inv_cov) * Q).sum(dim=-1, keepdim=True)
        term2 = (torch.matmul(K, inv_cov) * K).sum(dim=-1, keepdim=True).transpose(-1, -2)
        term3 = 2 * torch.matmul(torch.matmul(Q, inv_cov), K.transpose(-1, -2))
        dist_sq = term1 + term2 - term3
        dist = torch.sqrt(torch.clamp(dist_sq, min=0.0))
        return dist

    def _calc_jaccard_sim(self, h: torch.Tensor):
        Q = self.Wq(h)
        K = self.Wk(h)
        Q_exp = Q.unsqueeze(2)
        K_exp = K.unsqueeze(1)
        int_map = Q_exp * K_exp
        union_map = Q_exp + K_exp - int_map
        int_sum = int_map.sum(dim=-1)
        union_sum = union_map.sum(dim=-1)
        iou = int_sum / (union_sum + 1e-8)
        sim = 2 * iou - 1
        return sim

    def _calc_rbf_sim(self, h: torch.Tensor, dist_type: str = "dist", gamma: float = 1.0, p: int = 2):
        if dist_type in ["mahalanobis", "mahalanobis_dist"]:
            dist = self._calc_mahalanobis_dist(h)
        else:
            dist = self._calc_dist(h, p)
        if gamma == 0.0:
            gamma = 1.0
        rbf = torch.exp(-gamma * (dist ** 2))
        return 2 * rbf - 1

    def _calc_kl_div_sim(self, h: torch.Tensor, beta: float = None):
        Q = F.softmax(self.Wq(h), dim=-1)
        K = F.softmax(self.Wk(h), dim=-1)
        Q_exp = Q.unsqueeze(2)
        K_exp = K.unsqueeze(1)
        kl = (Q_exp * (Q_exp.log() - K_exp.log())).sum(dim=-1)
        if beta is None:
            beta = torch.mean(kl) + 1e-6
        sim = torch.exp(-kl / beta)
        return sim

    def _calc_pearson_corr(self, h: torch.Tensor):
        Q = self.Wq(h)
        K = self.Wk(h)
        Q_c = Q - Q.mean(dim=-1, keepdim=True)
        K_c = K - K.mean(dim=-1, keepdim=True)
        Q_n = F.normalize(Q_c, p=2, dim=-1)
        K_n = F.normalize(K_c, p=2, dim=-1)
        sim = torch.matmul(Q_n, K_n.transpose(-1, -2))
        return sim

    def _calc_mi_sim(self, h: torch.Tensor):
        Q = F.softmax(self.Wq(h), dim=-1)
        K = F.softmax(self.Wk(h), dim=-1)
        scores = torch.matmul(Q, K.transpose(-1, -2))
        lse_q = torch.logsumexp(scores, dim=-1, keepdim=True)
        lse_k = torch.logsumexp(scores, dim=-2, keepdim=True)
        pmi = scores - lse_q - lse_k
        pmi_mean = pmi.mean(dim=-1, keepdim=True)
        pmi_std = pmi.std(dim=-1, keepdim=True) + 1e-6
        pmi_norm = (pmi - pmi_mean) / pmi_std
        sim = torch.tanh(pmi_norm)
        return sim

    def _calc_sinkhorn_similarity(self, h: torch.Tensor, beta: float = None, max_iter: int = 100, eps: float = 0.1):
        Q = F.softmax(self.Wq(h), dim=-1)
        K = F.softmax(self.Wk(h), dim=-1)
        C = torch.cdist(Q, K, p=2)
        k = torch.exp(-C / eps)
        B, N, H = Q.shape
        u = torch.ones((B, N), device=Q.device) / N
        for _ in range(max_iter):
            kt_u = torch.bmm(k.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)
            v = 1.0 / (kt_u + 1e-8)
            k_v = torch.bmm(k, v.unsqueeze(-1)).squeeze(-1)
            u = 1.0 / (k_v + 1e-8)
        P = u.unsqueeze(-1) * k * v.unsqueeze(-2)
        sim = P * N
        sim = torch.clamp(sim, 0, 1)
        sim = 2 * sim - 1
        return sim

    def _calc_linear_cka_similarity(self, h: torch.Tensor):
        Q = F.softmax(self.Wq(h), dim=-1)
        K = F.softmax(self.Wk(h), dim=-1)
        GramQ = torch.bmm(Q, Q.transpose(1, 2))
        GramK = torch.bmm(K, K.transpose(1, 2))
        B, N, _ = Q.shape
        H_mat = torch.eye(N, device=Q.device) - torch.ones((N, N), device=Q.device) / N
        GramQ_c = torch.bmm(torch.bmm(H_mat.unsqueeze(0).expand(B, -1, -1), GramQ),
                            H_mat.unsqueeze(0).expand(B, -1, -1))
        GramK_c = torch.bmm(torch.bmm(H_mat.unsqueeze(0).expand(B, -1, -1), GramK),
                            H_mat.unsqueeze(0).expand(B, -1, -1))
        sim = GramQ_c * GramK_c
        denom = torch.norm(GramQ_c, dim=(-1, -2)) * torch.norm(GramK_c, dim=(-1, -2))
        sim = sim / (denom.view(B, 1, 1) + 1e-8)
        return sim

    def _calc_dtw_sim(self, h: torch.Tensor, gamma: float = 0.1, max_dist=10.0):
        try:
            from ...models import SoftDTWMatrix
            sdtw = SoftDTWMatrix(gamma, max_dist)
            Q = F.softmax(self.Wq(h), dim=-1)
            K = F.softmax(self.Wk(h), dim=-1)
            dist = sdtw(Q, K)
            sim = torch.exp(-dist)
            return 2 * sim - 1
        except ImportError:
            return self._calc_attn_sim(h)

    def calc_adj(self, h: torch.Tensor):
        sim_type = self.sim_kwargs["sim_type"]
        p = self.sim_kwargs["p"]
        gamma = self.sim_kwargs["gamma"]
        beta = self.sim_kwargs["beta"]

        if sim_type in ["attn", "attention"]:
            sim = self._calc_attn_sim(h)
        elif sim_type in ["dist", "distance"]:
            dist = self._calc_dist(h, p=p)
            gamma = gamma or 1.0
            sim = torch.exp(-gamma * (dist ** 2))
            sim = 2 * sim - 1
        elif sim_type in ["mahalanobis", "mahalanobis_dist"]:
            val = self._calc_mahalanobis_dist(h)
            if beta is None:
                beta = val.mean() + 1e-6
            sim = 1 - 2 * torch.tanh(val / beta)
        elif sim_type == "jaccard":
            sim = self._calc_jaccard_sim(h)
        elif sim_type == "rbf":
            sim = self._calc_rbf_sim(h, self.sim_kwargs["dist_type"], gamma, p)
        elif sim_type in ["kl_div", "kl_divergence"]:
            sim = self._calc_kl_div_sim(h, beta)
        elif sim_type in ["corr", "pearson_corr"]:
            sim = self._calc_pearson_corr(h)
        elif sim_type in ["dtw", "soft_dtw"]:
            sim = self._calc_dtw_sim(h, gamma)
        elif sim_type in ["mi", "mutual_information"]:
            sim = self._calc_mi_sim(h)
        elif sim_type == "sinkhorn":
            sim = self._calc_sinkhorn_similarity(h, beta)
        elif sim_type == "cka":
            sim = self._calc_linear_cka_similarity(h)
        else:
            sim = self._calc_attn_sim(h)

        adj = (sim >= self.soft_threshold).float()
        return adj

    def forward(self, x: torch.Tensor, h_prev: Union[torch.Tensor, Tuple] = None):
        if self.cell_type == 'lstm':
            h, c = h_prev if h_prev is not None else (None, None)
        else:
            h = h_prev

        if h is None:
            B, N, _ = x.shape
            h = torch.zeros(B, N, self.hidden_size, **self.factory_kwargs)
            if self.cell_type == 'lstm':
                c = torch.zeros(B, N, self.hidden_size, **self.factory_kwargs)

        adj = self.calc_adj(h)

        if adj.dim() == 2:
            x_agg = torch.matmul(adj, x)
            h_agg = torch.matmul(adj, h)
        else:
            x_agg = torch.bmm(adj, x)
            h_agg = torch.bmm(adj, h)

        x_agg = self.Wx(x_agg)
        h_agg = self.Wh(h_agg)

        B, N, F = x_agg.shape
        x_flat = x_agg.reshape(B * N, -1)
        h_flat = h_agg.reshape(B * N, -1)

        if self.cell_type == 'lstm':
            c_flat = c.reshape(B * N, -1)
            h_out_flat, c_out_flat = self.cell(x_flat, h_flat, c_flat)
            h_out = h_out_flat.reshape(B, N, -1)
            c_out = c_out_flat.reshape(B, N, -1)
            return h_out, c_out
        else:
            h_out_flat = self.cell(x_flat, h_flat)
            h_out = h_out_flat.reshape(B, N, -1)
            return h_out
