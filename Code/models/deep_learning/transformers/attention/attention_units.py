import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Union, Any, Tuple, List, Dict, Callable
from .....models.utils import DLModule
import warnings


class GeneralAttentionBlock(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 context_length: int,
                 heads: int = None,
                 bias: bool = False,
                 causal: bool = True,
                 cross: bool = False,
                 hard: bool = False,
                 additive: bool = False,
                 multiheaded: bool = True,
                 out_proj: bool = False,
                 temperature: float = None,
                 func: Union[str, Callable, nn.Module, DLModule] = "tanh",
                 caching: bool = False,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        lin_kwargs = {
            "bias": bias,
            **self.factory_kwargs
        }
        self.lin_kwargs = lin_kwargs
        self.Wq = nn.Linear(
            in_features=input_size,
            out_features=hidden_size,
            **lin_kwargs
        )
        self.Wk = nn.Linear(
            in_features=input_size,
            out_features=hidden_size,
            **lin_kwargs
        )
        self.Wv = nn.Linear(
            in_features=input_size,
            out_features=hidden_size,
            **lin_kwargs
        )
        self.Wo = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            **lin_kwargs
        ) if out_proj else None
        if not multiheaded:
            if additive:
                self.vi = nn.Linear(
                    in_features=hidden_size,
                    out_features=1,
                    **lin_kwargs
                )
        else:
            if heads is None:
                target_heads: int = 64
                h = hidden_size // target_heads
                h = max(1, h)
                if hidden_size % h != 0:
                    divisors = [i for i in range(1, hidden_size + 1) if hidden_size % i == 0]
                    h = min(divisors, key=lambda x: abs(x - h))
                heads = h
            if hidden_size % heads != 0:
                raise ValueError(f"Dimensional mismatch.\n"
                                 f"The given dimension: {hidden_size} is not divisible by heads: {heads}.\n")
            self.heads = heads
            self.head_dim = hidden_size // heads
            if additive:
                self.vi = nn.Linear(
                    in_features=self.head_dim,
                    out_features=1,
                    **lin_kwargs
                )
        self.causal = causal
        if self.causal:
            self.register_buffer(
                "causal_mask",
                torch.triu(torch.ones((context_length, context_length), **self.factory_kwargs), diagonal=1).bool()
            )
        self.context_len = context_length
        self.cross = cross
        self.hard = hard
        self.additive = additive
        self.caching = caching
        self.multiheaded = multiheaded
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.out_proj = out_proj
        self.func = self._resolve_funcs(func, *args, **kwargs)

    def _general_attn(self, x: torch.Tensor, y: torch.Tensor = None):
        b, n, _ = x.size()
        original_n = n
        if n > self.context_len:
            warnings.warn(f"The given embeddings tensor are beyond the described context length {self.context_len}.\n"
                          f"Truncating it to the defined context length", UserWarning)
            x = x[:, :self.context_len, :]
        elif n < self.context_len:
            pad_zeros = torch.zeros((b, self.context_len - n, x.size(-1)), device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad_zeros], dim=-2)

        b, n, _ = x.size()
        if y is None and self.cross:
            y = torch.zeros_like(x)
        if self.cross:
            Q = self.Wq(y)
        else:
            Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        if not self.multiheaded:
            if not self.additive:
                scores: torch.Tensor = Q @ K.transpose(-1, -2)
            else:
                scores = self.func(Q.unsqueeze(-2) + K.unsqueeze(-3))
                scores = self.vi(scores).squeeze(-1)
            scores = scores / math.sqrt(self.hidden_size)
            if self.causal:
                scores = scores.masked_fill(self.causal_mask, -torch.inf)
            if not self.hard:
                scores = F.softmax(scores, dim=-1)
            else:
                scores = F.gumbel_softmax(scores, tau=self.temperature, hard=True, dim=-1)
            context_vectors = scores @ V
            if self.out_proj:
                context_vectors = self.Wo(context_vectors)

            if context_vectors.size(1) > original_n:
                context_vectors = context_vectors[:, :original_n, :]

            # Slice K/V back to original length
            if K.size(2) > original_n:
                K = K[:, :, :original_n, :]
                V = V[:, :, :original_n, :]

            return context_vectors, scores, K, V
        else:
            Q = Q.view(b, n, self.heads, self.head_dim).transpose(1, 2)
            K = K.view(b, n, self.heads, self.head_dim).transpose(1, 2)
            V = V.view(b, n, self.heads, self.head_dim).transpose(1, 2)
            if not self.additive:
                scores = Q @ K.transpose(-1, -2)
            else:
                scores = self.func(Q.unsqueeze(-2) + K.unsqueeze(-3))
                scores = self.vi(scores).squeeze()
            scores = scores / math.sqrt(self.hidden_size)
            if self.causal:
                scores = scores.masked_fill(self.causal_mask, -torch.inf)
            if not self.hard:
                scores = F.softmax(scores, dim=-1)
            else:
                scores = F.gumbel_softmax(scores, tau=self.temperature, hard=True, dim=-1)
            context_vectors = scores @ V
            context_vectors = context_vectors.transpose(1, 2).contiguous().view(b, n, self.hidden_size)
            if self.out_proj:
                context_vectors = self.Wo(context_vectors)

            if context_vectors.size(1) > original_n:
                context_vectors = context_vectors[:, :original_n, :]

            # Slice K/V back to original length
            if K.size(2) > original_n:
                K = K[:, :, :original_n, :]
                V = V[:, :, :original_n, :]

            return context_vectors, scores, K, V

    def forward(self, x: torch.Tensor, y: torch.Tensor = None,
                scores: torch.Tensor = None,
                K: torch.Tensor = None, V: torch.Tensor = None):
        if K is None or V is None:
            return self._general_attn(x, y)
        else:
            return self._cached_general_attn(x, scores, K, V, y)

    def _cached_general_attn(self, x, scores, K, V, y=None):
        b, n, _ = x.size()
        if y is None and self.cross:
            y = torch.zeros_like(x)

        if self.cross:
            Q = self.Wq(y)
        else:
            Q = self.Wq(x)

        K_new = self.Wk(x)
        V_new = self.Wv(x)

        if not self.multiheaded:
            if K is not None:
                K = torch.cat([K, K_new], dim=1)
                V = torch.cat([V, V_new], dim=1)
            else:
                K = K_new
                V = V_new

            # Check context length
            if K.size(1) > self.context_len:
                warnings.warn(f"The total sequence length {K.size(1)} exceeds the context length {self.context_len}.",
                              UserWarning)

            if not self.additive:
                current_scores = Q @ K.transpose(-1, -2)
            else:
                current_scores = self.func(Q.unsqueeze(-2) + K.unsqueeze(-3))
                current_scores = self.vi(current_scores).squeeze(-1)

            current_scores = current_scores / math.sqrt(self.hidden_size)

            if self.causal:
                total_len = K.size(1)
                # We need to mask positions that effectively shouldn't be attended to.
                # In strict autoregressive generation (n=1), we attend to everything in K (0...t).
                # If n > 1, we need to respect causality within the new chunk and between new chunk and past.
                # Construct mask for the current query chunk.
                # Q indices: [past_len ... past_len + n]
                # K indices: [0 ... total_len]
                past_len = total_len - n
                # Slice the causal mask
                # self.causal_mask is (context, context)
                # We want rows [past_len : total_len] and cols [0 : total_len]
                if total_len <= self.context_len:
                    mask_chunk = self.causal_mask[past_len:total_len, :total_len]
                    current_scores = current_scores.masked_fill(~mask_chunk, -torch.inf)

            if not self.hard:
                current_scores = F.softmax(current_scores, dim=-1)
            else:
                current_scores = F.gumbel_softmax(current_scores, tau=self.temperature, hard=True, dim=-1)

            context_vectors = current_scores @ V
            if self.out_proj:
                context_vectors = self.Wo(context_vectors)

            return context_vectors, current_scores, K, V

        else:
            Q = Q.view(b, n, self.heads, self.head_dim).transpose(1, 2)
            K_new = K_new.view(b, n, self.heads, self.head_dim).transpose(1, 2)
            V_new = V_new.view(b, n, self.heads, self.head_dim).transpose(1, 2)

            if K is not None:
                K = torch.cat([K, K_new], dim=2)
                V = torch.cat([V, V_new], dim=2)
            else:
                K = K_new
                V = V_new

            # Check context length
            if K.size(2) > self.context_len:
                warnings.warn(f"The total sequence length {K.size(2)} exceeds the context length {self.context_len}.",
                              UserWarning)

            if not self.additive:
                current_scores = Q @ K.transpose(-1, -2)
            else:
                current_scores = self.func(Q.unsqueeze(-2) + K.unsqueeze(-3))
                current_scores = self.vi(current_scores).squeeze(-1)

            current_scores = current_scores / math.sqrt(self.hidden_size)

            if self.causal:
                total_len = K.size(2)
                past_len = total_len - n
                if total_len <= self.context_len:
                    mask_chunk = self.causal_mask[past_len:total_len, :total_len]
                    # current_scores shape: (b, heads, n, total_len)
                    # mask_chunk shape: (n, total_len)
                    # It broadcasts correctly.
                    current_scores = current_scores.masked_fill(mask_chunk, -torch.inf)

            if not self.hard:
                current_scores = F.softmax(current_scores, dim=-1)
            else:
                current_scores = F.gumbel_softmax(current_scores, tau=self.temperature, hard=True, dim=-1)

            context_vectors = current_scores @ V
            context_vectors = context_vectors.transpose(1, 2).contiguous().view(b, n, self.hidden_size)

            if self.out_proj:
                context_vectors = self.Wo(context_vectors)

            return context_vectors, current_scores, K, V


SentenceAttention = GeneralAttentionBlock
WordAttention = GeneralAttentionBlock
ContextAttention = GeneralAttentionBlock
GlobalAttention = GeneralAttentionBlock
