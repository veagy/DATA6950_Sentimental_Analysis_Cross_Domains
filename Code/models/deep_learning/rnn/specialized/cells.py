"""Specialized RNN cells extracted from RNNFamilyCell."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from typing import Optional, Any, Union, Callable, List, Tuple, Dict
import importlib.util
from pathlib import Path

from .....models.utils import DLModule
from ..base import RNNCell, LSTMCell, GRUCell

# Load Complex without importing ``activations`` package __init__ (optional submodules may be missing).
_cp = Path(__file__).resolve().parents[3] / "deep_learning" / "activations" / "Complex" / "complex_.py"
_spec = importlib.util.spec_from_file_location("_hrm_cells_complex", _cp)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load Complex from {_cp}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Complex = _mod.Complex


class LMUCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 memory_size: int,
                 theta: int = 1,
                 min_discrete_time: float = 1.0,
                 func: Union[str, Callable, nn.Module] = None,
                 bias: bool = True,
                 proj_size: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        if memory_size > 256:
            dtype = torch.float64
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        self.A = nn.Parameter(torch.zeros((memory_size, memory_size), **self.factory_kwargs), requires_grad=False)
        self.B = nn.Parameter(torch.zeros((memory_size, 1), **self.factory_kwargs), requires_grad=False)
        self.get_lmu_mat(memory_size, theta, min_discrete_time)

        self.Wh = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=bias,
            **self.factory_kwargs
        )
        self.Wx = nn.Linear(
            in_features=input_size,
            out_features=hidden_size,
            bias=bias,
            **self.factory_kwargs
        )
        self.Wm = nn.Linear(
            in_features=memory_size,
            out_features=hidden_size,
            bias=bias,
            **self.factory_kwargs
        )

        self.Wout = nn.Linear(
            in_features=hidden_size,
            out_features=proj_size,
            bias=bias,
            **self.factory_kwargs
        ) if proj_size is not None else None
        self.proj = False if proj_size is None else True

        self.ex = nn.Parameter(torch.ones((1, input_size), **self.factory_kwargs))
        self.eh = nn.Parameter(torch.zeros((1, hidden_size), **self.factory_kwargs))
        self.em = nn.Parameter(torch.ones((1, memory_size), **self.factory_kwargs))

        self.func = self._resolve_funcs(func, *args, **kwargs)

        self.dim = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "memory_size": memory_size
        }

    def get_lmu_mat(self, memory_size: int, theta: int, dt: float = 1.0, eps: Optional[float] = 1e-8):
        if memory_size < 0 or theta < 0:
            warnings.warn(f"The given values must be positive but got order: {memory_size}, theta: {theta}.\n"
                          f"Forcefully converting them to positive values", UserWarning)
            memory_size = math.fabs(memory_size)
            theta = math.fabs(theta)

        indices = torch.arange(memory_size, dtype=self.factory_kwargs["dtype"])
        Q = indices.unsqueeze(1)
        R = indices.unsqueeze(0)

        A_cont = (2 * Q + 1) * torch.where(Q < R, -torch.ones_like(Q), (-1.0) ** (Q - R + 1.0))
        A_cont = A_cont / (theta + eps)

        B_cont = (2 * Q + 1) * ((-1.0) ** Q)
        B_cont = B_cont / (theta + eps)

        self.A = torch.matrix_exp(A_cont * dt)

        I_ = torch.eye(memory_size)
        self.B = torch.linalg.solve(A_cont, (self.A - I_) @ B_cont)

    def forward(self, x: torch.Tensor, m_prev: Optional[torch.Tensor] = None, h_prev: Optional[torch.Tensor] = None):
        if x.dtype != self.factory_kwargs['dtype']:
            x = x.to(dtype=self.factory_kwargs['dtype'])

        if m_prev is None:
            m_prev = torch.zeros((x.size(0), self.dim["memory_size"]), **self.factory_kwargs)
        if h_prev is None:
            h_prev = torch.zeros((x.size(0), self.dim["hidden_size"]), **self.factory_kwargs)

        u_t = (x @ self.ex.T) + (h_prev @ self.eh.T) + (m_prev @ self.em.T)

        m_t = (m_prev @ self.A.T) + (u_t @ self.B.T)

        h_t = self.func(self.Wh(h_prev) + self.Wm(m_t) + self.Wx(x))
        if self.proj:
            h_t = self.Wout(h_t)
        return h_t, m_t


class IndRNNCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 func: Union[str, Callable, nn.Module] = None,
                 bias: bool = True,
                 proj_size: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        self.W = nn.Linear(
            in_features=input_size,
            out_features=hidden_size,
            bias=bias,
            **self.factory_kwargs
        )
        self.u = nn.Parameter(torch.ones((hidden_size,), **self.factory_kwargs))
        self.bias = nn.Parameter(torch.zeros((hidden_size,), **self.factory_kwargs))

        self.Wout = nn.Linear(
            in_features=hidden_size,
            out_features=proj_size,
            bias=bias,
            **self.factory_kwargs
        ) if proj_size is not None else None
        self.proj = False if proj_size is None else True

        if func is None:
            func = "sigmoid"
        self.func = self._resolve_funcs(func, *args, **kwargs)

        self.dim = {
            "input_size": input_size,
            "hidden_size": hidden_size
        }

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor]):
        if h_prev is None:
            h_prev = torch.zeros((self.dim["hidden_size"],), **self.factory_kwargs)

        pre_act = self.W(x) + (self.u * h_prev) + self.bias
        h_t = self.func(pre_act)

        if self.proj:
            h_t = self.Wout(h_t)
        return h_t, h_t


class PhasedLSTMCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 funcs: Union[List[Tuple[str, Callable, nn.Module]],
                 Tuple[Union[str, Callable, nn.Module]],
                 Dict[str, Union[str, Callable, nn.Module]]] = None,
                 bias: bool = True,
                 use_feature_timestamp: bool = False,
                 alpha: float = 0.0001,
                 proj_size: int = 0,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.use_feature_timestamp = use_feature_timestamp

        real_hidden_size = proj_size if proj_size > 0 else hidden_size

        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }

        in_lin_kwargs = {
            "in_features": input_size,
            "out_features": hidden_size,
            "bias": bias,
            **self.factory_kwargs
        }
        h_lin_kwargs = {
            "in_features": real_hidden_size,
            "out_features": hidden_size,
            "bias": bias,
            **self.factory_kwargs
        }
        out_lin_kwargs = {
            "in_features": hidden_size,
            "out_features": proj_size,
            "bias": bias,
            **self.factory_kwargs
        }

        self.Wii = nn.Linear(**in_lin_kwargs)
        self.Wif = nn.Linear(**in_lin_kwargs)
        self.Wig = nn.Linear(**in_lin_kwargs)
        self.Wio = nn.Linear(**in_lin_kwargs)

        self.Whi = nn.Linear(**h_lin_kwargs)
        self.Whf = nn.Linear(**h_lin_kwargs)
        self.Whg = nn.Linear(**h_lin_kwargs)
        self.Who = nn.Linear(**h_lin_kwargs)

        t_kwargs = {
            "size": (1, hidden_size),
            **self.factory_kwargs
        }
        self.tau = nn.Parameter(torch.ones(**t_kwargs))
        self.r = nn.Parameter(torch.zeros(**t_kwargs))
        self.r_on = nn.Parameter(torch.ones(**t_kwargs))
        nn.init.constant_(self.r_on, 0.05)
        self.alpha = alpha

        if proj_size > 0:
            self.Wo = nn.Linear(**out_lin_kwargs)
        else:
            self.Wo = None

        if funcs is None:
            funcs = ["sigmoid", "sigmoid", "tanh", "sigmoid", "tanh"]
        self.funcs = self._resolve_funcs(funcs, *args, **kwargs)

    def calc_time_gate(self, t: Union[int, float, torch.Tensor]):
        phi_t = torch.remainder(t - self.r, self.tau) / (self.tau + 1e-8)

        low = 2 * phi_t / (self.r_on + 1e-8)
        mid = 2 - low
        high = self.alpha * phi_t

        k_t = torch.where(
            phi_t < 0.5 * self.r_on,
            low,
            torch.where(
                (0.5 * self.r_on <= phi_t) & (phi_t < self.r_on),
                mid,
                high
            )
        )

        return k_t, phi_t

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor,
                t: Optional[Union[int, float, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_feature_timestamp:
            t_val = x[:, 0:1]
            x_in = x[:, 1:]
        else:
            x_in = x
            if t is None:
                t_val = torch.zeros((x.size(0), 1), device=x.device, dtype=x.dtype)
            else:
                if isinstance(t, (int, float)):
                    t_val = torch.full((x.size(0), 1), t, device=x.device, dtype=x.dtype)
                else:
                    t_val = t.view(-1, 1)

        f_sig_i, f_sig_f, f_tanh_g, f_sig_o, f_tanh_h = self.funcs

        i_t = f_sig_i(self.Wii(x_in) + self.Whi(h_prev))
        f_t = f_sig_f(self.Wif(x_in) + self.Whf(h_prev))
        g_t = f_tanh_g(self.Wig(x_in) + self.Whg(h_prev))
        o_t = f_sig_o(self.Wio(x_in) + self.Who(h_prev))
        k_t, _ = self.calc_time_gate(t_val)

        c_t = (k_t * ((f_t * c_prev) + (i_t * g_t))) + ((1 - k_t) * c_prev)

        h_t = (k_t * (o_t * f_tanh_h(c_t))) + ((1 - k_t) * h_prev)

        if self.proj_size is not None and self.proj_size > 0:
            h_t = self.Wo(h_t)
        return h_t, c_t


class mRNNCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 funcs: Union[str, Callable, nn.Module, DLModule] = None,
                 bias: bool = True,
                 proj_size: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        in_kwargs = {
            "in_features": input_size,
            "out_features": hidden_size,
            "bias": bias,
            **self.factory_kwargs
        }
        h_kwargs = {
            "in_features": hidden_size,
            "out_features": hidden_size,
            "bias": bias,
            **self.factory_kwargs
        }
        self.Wfx = nn.Linear(**in_kwargs)
        self.Whx = nn.Linear(**in_kwargs)

        self.Wfh = nn.Linear(**h_kwargs)
        self.Whh = nn.Linear(**h_kwargs)

        self.bh = nn.Parameter(
            torch.zeros((hidden_size, 1), **self.factory_kwargs)
        )

        if proj_size is not None:
            self.Wo = nn.Linear(
                in_features=hidden_size,
                out_features=proj_size,
                bias=bias,
                **self.factory_kwargs
            )
        else:
            self.Wo = None
        self.proj_size = proj_size
        self.hidden_size = hidden_size

        if 'proj_size' in kwargs:
            kwargs = dict(kwargs)
            kwargs.pop('proj_size')
        self.func = self._resolve_func(funcs, *args, **kwargs)

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor]):
        if h_prev is None:
            h_prev = torch.zeros(
                (self.hidden_size, 1), **self.factory_kwargs
            )
        f_t = self.Wfx(x) * self.Wfh(h_prev)

        h_t = self.func(self.Whx(x) + self.Whh(f_t) + self.bh.view(1, -1))

        if self.proj_size is not None:
            h_t = self.Wo(h_t)
        return h_t


class FastWeightsRNCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 funcs: Union[str, Callable, nn.Module, DLModule] = None,
                 bias: bool = True,
                 proj_size: int = None,
                 device: str = "cpu",
                 delta_rule: bool = False,
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        in_kwargs = {
            "in_features": input_size,
            "out_features": hidden_size,
            "bias": bias,
            **self.factory_kwargs
        }
        self.Wq = nn.Linear(**in_kwargs)
        self.Wk = nn.Linear(**in_kwargs)
        self.Wv = nn.Linear(**in_kwargs)

        if delta_rule:
            self.beta = nn.Parameter(
                torch.ones((1, hidden_size), **self.factory_kwargs)
            )
        self.W_slow = nn.Linear(**in_kwargs)
        self.W_fast = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=bias,
            **self.factory_kwargs
        )
        if proj_size is not None:
            self.Wo = nn.Linear(
                in_features=hidden_size,
                out_features=proj_size,
                bias=bias,
                **self.factory_kwargs
            )
        self.proj_size = proj_size
        self.func = self._resolve_funcs(funcs, *args, **kwargs)
        self.hidden_size = hidden_size
        self.delta_rule = delta_rule

    def forward(self, x: torch.Tensor, W_prev: Optional[torch.Tensor] = None):
        if W_prev is None:
            batch_size = x.size(0)
            W_prev = torch.zeros(
                (batch_size, self.hidden_size, self.hidden_size), **self.factory_kwargs
            )
        q_t = self.Wq(x)
        k_t = self.Wk(x)
        v_t = self.Wv(x)

        if not self.delta_rule:
            update = v_t.unsqueeze(2) @ k_t.unsqueeze(1)
            W_t = W_prev + update
        else:
            k_t_unsq = k_t.unsqueeze(2)
            W_k = torch.bmm(W_prev, k_t_unsq).squeeze(2)
            err = v_t - W_k
            beta = (self.beta * err).sum(dim=1, keepdim=True)
            update = beta.unsqueeze(2) * k_t.unsqueeze(1)
            W_t = W_prev + update

        h_t = W_t @ q_t
        y_t = self.func(self.W_slow(x) + self.W_fast(h_t))
        if self.proj_size is not None:
            y_t = self.Wo(y_t)
        return y_t, W_t


class SkipRNNCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 funcs: Union[List[Tuple[str, Callable, nn.Module]],
                 Tuple[Union[str, Callable, nn.Module]],
                 Dict[str, Union[str, Callable, nn.Module]]] = None,
                 func: Union[nn.Module, DLModule, str, Callable] = None,
                 bias: bool = True,
                 update: Union[str, DLModule, nn.Module] = "rnn",
                 proj_size: int = 0,
                 threshold: float = 1.0,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        self.Wp = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=bias,
            **self.factory_kwargs
        )
        self.bp = nn.Parameter(
            torch.zeros((1, hidden_size), **self.factory_kwargs)
        )
        if func is None:
            func = torch.sigmoid
        self.func = self._resolve_funcs(func, *args, **kwargs)
        self.threshold = threshold

        update_kwargs = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "funcs": funcs,
            "bias": bias,
            "proj_size": proj_size,
            "args": args,
            "kwargs": kwargs,
            **self.factory_kwargs
        }

        self.hidden_size = hidden_size

        if proj_size is not None:
            self.Wo = nn.Linear(
                in_features=hidden_size,
                out_features=proj_size,
                bias=bias,
                **self.factory_kwargs
            )
        else:
            self.Wo = None
        self.proj_size = proj_size

        if isinstance(update, str):
            if update == "rnn":
                self.update = RNNCell(**update_kwargs)
            elif update == "lstm":
                self.update = LSTMCell(**update_kwargs)
            elif update == "gru":
                self.update = GRUCell(**update_kwargs)
            else:
                self.update = RNNCell(**update_kwargs)
        else:
            self.update = update

    def skip_gate(self, h_prev: torch.Tensor, G_prev: torch.Tensor):
        du_t = self.func(self.Wp(h_prev) + self.bp)
        G_t = G_prev + du_t
        u_t = (G_t >= self.threshold).float()
        return u_t, G_t

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None, G_prev: Optional[torch.Tensor] = None):
        if h_prev is None:
            h_prev = torch.zeros((self.hidden_size, 1), **self.factory_kwargs)
        if G_prev is None:
            G_prev = torch.zeros((1, self.hidden_size), **self.factory_kwargs)

        h_bar = self.update(x, h_prev)
        u_t, G_t = self.skip_gate(h_prev, G_prev)

        h_t = (u_t * h_bar) + ((1 - u_t) * h_prev)
        if self.proj_size is not None:
            h_t = self.Wo(h_t)

        return h_t, G_t


class JumpLSTMCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 funcs: Union[List[Tuple[str, Callable, nn.Module]],
                 Tuple[Union[str, Callable, nn.Module]],
                 Dict[str, Union[str, Callable, nn.Module]]] = None,
                 bias: bool = True,
                 hard: bool = False,
                 tau: float = 1.0,
                 proj_size: int = 0,
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
        self.hard = hard
        self.tau = tau

        real_hidden_size = proj_size if proj_size > 0 else hidden_size

        in_lin_kwargs = {
            "in_features": input_size,
            "out_features": hidden_size,
            "bias": bias,
            "device": device,
            "dtype": dtype
        }
        h_lin_kwargs = {
            "in_features": real_hidden_size,
            "out_features": hidden_size,
            "bias": bias,
            "device": device,
            "dtype": dtype
        }
        out_lin_kwargs = {
            "in_features": hidden_size,
            "out_features": proj_size,
            "bias": bias,
            "device": device,
            "dtype": dtype
        }

        self.Wii = nn.Linear(**in_lin_kwargs)
        self.Wif = nn.Linear(**in_lin_kwargs)
        self.Wig = nn.Linear(**in_lin_kwargs)
        self.Wio = nn.Linear(**in_lin_kwargs)

        self.Whi = nn.Linear(**h_lin_kwargs)
        self.Whf = nn.Linear(**h_lin_kwargs)
        self.Whg = nn.Linear(**h_lin_kwargs)
        self.Who = nn.Linear(**h_lin_kwargs)
        self.Wa = nn.Linear(**h_lin_kwargs)
        self.Wu = nn.Linear(**h_lin_kwargs)

        if proj_size > 0:
            self.Wo = nn.Linear(**out_lin_kwargs)
        else:
            self.Wo = None

        if funcs is None:
            funcs = ["sigmoid", "sigmoid", "tanh", "sigmoid", "tanh", "softplus"]
        self.funcs = self._resolve_funcs(funcs, *args, **kwargs)

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None, c_prev: Optional[torch.Tensor] = None):
        if h_prev is None:
            h_prev = torch.zeros((self.hidden_size, 1), **self.factory_kwargs)
        if c_prev is None:
            c_prev = torch.zeros((self.hidden_size, 1), **self.factory_kwargs)

        f_sig_i, f_sig_f, f_tanh_g, f_sig_o, f_tanh_h, f_soft = self.funcs

        i_t = f_sig_i(self.Wii(x) + self.Whi(h_prev))
        f_t = f_sig_f(self.Wif(x) + self.Whf(h_prev))
        g_t = f_tanh_g(self.Wig(x) + self.Whg(h_prev))
        o_t = f_sig_o(self.Wio(x) + self.Who(h_prev))

        c_t = (f_t * c_prev) + (i_t * g_t)
        h_t = o_t * f_tanh_h(c_t)

        a_t = self.Wa(h_t)
        u_t = self.Wu(h_t)
        a_t = torch.cat([a_t, u_t], dim=-1)
        gate_dist = F.gumbel_softmax(a_t, tau=self.tau, hard=self.hard)
        u_t = gate_dist[:, 1:2]

        h_t = u_t * h_t + (1 - u_t) * h_prev
        c_t = u_t * c_t + (1 - u_t) * c_prev

        if self.Wo is not None:
            h_t = self.Wo(h_t)

        return h_t, c_t, u_t


class ACTRNNCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 non_linearity: Union[str, Callable, nn.Module] = 'tanh',
                 funcs: Union[List, Tuple, Dict] = None,
                 halting_func: Union[str, Callable, nn.Module, DLModule] = 'sigmoid',
                 bias: bool = True,
                 max_step_size: int = 50,
                 cell_type: str = 'rnn',
                 proj_size: int = None,
                 epsilon: float = 0.0001,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        self.epsilon = epsilon
        self.max_step_size = max_step_size
        self.W_halt = nn.Linear(
            in_features=hidden_size,
            out_features=1,
            bias=bias,
            **self.factory_kwargs
        )
        in_kwargs = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "non_linearity": non_linearity,
            "funcs": funcs,
            "bias": bias,
            "proj_size": proj_size,
            "args": args,
            "kwargs": kwargs,
            **self.factory_kwargs
        }
        cell_type = cell_type.lower()
        if cell_type == 'rnn':
            self.cell = RNNCell(**in_kwargs)
        elif cell_type == 'lstm':
            self.cell = LSTMCell(**in_kwargs)
        elif cell_type == 'gru':
            self.cell = GRUCell(**in_kwargs)
        else:
            self.cell = RNNCell(**in_kwargs)

        if proj_size is not None:
            self.Wo = nn.Linear(
                in_features=hidden_size,
                out_features=proj_size,
                bias=bias,
                **self.factory_kwargs
            )
        else:
            self.Wo = None
        self.proj_size = proj_size
        self.hidden_size = hidden_size
        self.halted_func = self._resolve_func(halting_func, *args, **kwargs)

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None):
        batch_size = x.size(0)
        device = x.device

        h_accumulate = torch.zeros(batch_size, self.hidden_size).to(device)
        p_accumulate = torch.zeros(batch_size, 1).to(device)
        halted = torch.zeros(batch_size, 1).to(device).bool()
        n_steps = torch.zeros(batch_size, 1).to(device)
        remainder = torch.zeros(batch_size, 1).to(device)

        h_n = h_prev

        step = 0
        while not halted.all():
            step += 1
            is_first_flag = torch.ones(batch_size, 1).to(device) if step == 1 else torch.zeros(batch_size, 1).to(device)
            x_aug = torch.cat([x, is_first_flag], dim=1)

            h_n = self.cell(x_aug, h_n)
            prob = self.halted_func(self.W_halt(h_n))

            still_running = ~halted

            reached_threshold = (p_accumulate + prob) >= (1 - self.epsilon)

            p_n = torch.where(reached_threshold, 1 - p_accumulate, prob)

            h_accumulate = h_accumulate + (p_n * h_n * still_running.float())
            p_accumulate = p_accumulate + (p_n * still_running.float())
            n_steps = n_steps + still_running.float()

            halted = halted | reached_threshold | (step >= self.max_step_size)

            if reached_threshold.any():
                mask = reached_threshold & still_running
                remainder = torch.where(mask, 1 - (p_accumulate - p_n), remainder)

            if step >= self.max_step_size:
                break

        ponder_cost = n_steps + remainder
        if self.proj_size is not None:
            h_accumulate = self.Wo(h_accumulate)
        return h_accumulate, ponder_cost


class uRNNCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 proj_size: int,
                 num_reflections: int,
                 func: Union[str, nn.Module, DLModule, Callable, Complex],
                 bias: bool,
                 dim: int,
                 arrangement: str,
                 is_stacked_flag: bool,
                 device: str,
                 dtype: torch.dtype,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        from .....models.deep_learning.activations.Complex.complex_ import ComplexLinear
        self.V = ComplexLinear(
            in_features=input_size,
            out_features=hidden_size,
            bias=bias,
            **self.factory_kwargs
        )
        self.U = ComplexLinear(
            in_features=hidden_size,
            out_features=proj_size,
            bias=bias,
            **self.factory_kwargs
        )

        self.c_kwargs = {
            "dim": dim,
            "arrangement": arrangement,
            "is_stacked_flag": is_stacked_flag,
            "args": args,
            "kwargs": kwargs
        }

        self.thetas = nn.ParameterList([nn.Parameter(
            torch.zeros((hidden_size, 1), **self.factory_kwargs)
        ) for _ in range(3)])

        self.v = []
        for _ in range(num_reflections):
            real = nn.Parameter(torch.zeros((hidden_size, 1), **self.factory_kwargs))
            imag = nn.Parameter(torch.zeros((hidden_size, 1), **self.factory_kwargs))
            tensor = torch.stack([real, imag], dim=0)
            tensor = Complex(tensor, dim=0, *args, **self.factory_kwargs, **kwargs)
            self.v.append(tensor)
        self.hidden_size = hidden_size
        self.func = self._resolve_func(func, *args, **kwargs)
        self.permute_indices = torch.randperm(hidden_size, device=device)

    def calc_D(self):
        D_out = []
        for theta in self.thetas:
            real, imag = torch.cos(theta).squeeze(), torch.sin(theta).squeeze()
            real_diag = torch.diag(real)
            imag_diag = torch.diag(imag)
            tensor = torch.stack([real_diag, imag_diag], dim=0)
            tensor = Complex(tensor, dim=0, **self.factory_kwargs)
            D_out.append(tensor)
        return D_out

    def get_ortho_mat(self):
        hidden_size = self.hidden_size
        real, imag = torch.eye(hidden_size), torch.zeros((hidden_size, hidden_size), **self.factory_kwargs)
        tensor_I = torch.stack([real, imag], dim=0)
        tensor_I = Complex(tensor_I, dim=0, **self.factory_kwargs)
        M = tensor_I
        for v in self.v:
            sq_norm = v.mag() ** 2
            v_star = v.hermitian()
            H = tensor_I - 2 * (v * v_star) / sq_norm
            M = M @ H
        return M

    def calc_fourier_matrices(self):
        n = self.hidden_size
        indices = torch.arange(n)
        phi = -2 * torch.pi * indices.view(-1, 1) * indices.view(1, -1) / n
        real, imag = torch.cos(phi), torch.sin(phi)
        tensor = torch.stack([real, imag], dim=0)
        W = Complex(tensor, dim=0, **self.factory_kwargs)
        W_inv = W.inv()
        return W, W_inv

    def calc_permute(self, indices: Union[list, torch.Tensor]):
        if isinstance(indices, list):
            indices = torch.tensor(indices, device=self.factory_kwargs['device'], dtype=torch.long)
        n = self.hidden_size
        I_ = torch.eye(n)
        I_ = I_[indices]
        tensor = torch.stack([I_, I_], dim=0)
        tensor = Complex(tensor, dim=0, **self.factory_kwargs)
        return tensor

    def calc_W(self, indices: Union[list, torch.Tensor]):
        D1, D2, D3 = self.calc_D()
        P = self.calc_permute(indices)
        B = self.get_ortho_mat()
        Fourier, Fourier_inv = self.calc_fourier_matrices()
        W = D3 @ P @ B @ Fourier_inv @ D2 @ Fourier @ P @ B @ D1
        return W

    def forward(self, x: Union[torch.Tensor, Complex], h_prev: Optional[Complex] = None):
        if h_prev is None:
            h_prev = Complex.zeros(self.hidden_size, 1, **self.factory_kwargs, dim=0)
        if not isinstance(x, Complex):
            x = Complex(x, **self.c_kwargs)

        W = self.calc_W(self.permute_indices)
        Wh = h_prev.linear(W)
        Vx = self.V(x)
        pre_act = Wh + Vx
        h_t = self.func(pre_act)
        y_t = self.U(h_t)
        return y_t, h_t, W


class AntiSymRNNCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 func: Union[str, nn.Module, DLModule, Callable] = "sigmoid",
                 bias: bool = True,
                 proj_size: int = None,
                 epsilon: float = 0.5,
                 gamma: float = 0.01,
                 tensor_type: str = "real",
                 dim: Optional[int] = None,
                 arrangement: str = "split",
                 is_stacked_flag: bool = False,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        if tensor_type.lower() not in ["real", "complex"]:
            warnings.warn(f"The given type does not exists. Changing it to default type.", UserWarning)
            tensor_type = "real"
        self.tensor_type = tensor_type.lower()
        if tensor_type == "real":
            self.V = nn.Linear(
                in_features=input_size,
                out_features=hidden_size,
                bias=bias,
                **self.factory_kwargs
            )

            self.weight = nn.Parameter(torch.randn((hidden_size, hidden_size), **self.factory_kwargs) * 0.01)

            if proj_size is not None:
                self.Wo = nn.Linear(
                    in_features=hidden_size,
                    out_features=proj_size,
                    bias=bias,
                    **self.factory_kwargs
                )

        else:
            self.c_kwargs = {
                "dim": dim,
                "arrangement": arrangement,
                "is_stacked_flag": is_stacked_flag,
                "args": args,
                "kwargs": kwargs,
                **self.factory_kwargs
            }
            from .....models.deep_learning.activations.Complex.complex_ import ComplexLinear
            self.V = ComplexLinear(
                in_features=input_size,
                out_features=hidden_size,
                bias=bias,
                **self.c_kwargs
            )
            self.weight = [
                nn.Parameter(torch.randn((hidden_size, hidden_size), **self.factory_kwargs) * 0.01) for _ in range(2)
            ]
            self.weight = torch.stack(self.weight, dim=self.c_kwargs["dim"])

            self.weight = Complex(self.weight, **self.c_kwargs)

            if proj_size is not None:
                self.Wo = ComplexLinear(
                    in_features=hidden_size,
                    out_features=proj_size,
                    bias=bias,
                    **self.c_kwargs
                )

        self.proj_size = proj_size

        self.func = self._resolve_funcs(func, *args, **kwargs)

        self.register_buffer('epsilon', torch.tensor(epsilon, **self.factory_kwargs))
        self.register_buffer('gamma', torch.tensor(gamma, **self.factory_kwargs))
        self.hidden_size = hidden_size

    def calc_Wh(self):
        if self.tensor_type == "real":
            W = torch.triu(self.weight) - torch.triu(self.weight).T
            return W
        else:
            W = self.weight.triu_() - self.weight.triu_().t()
            return W

    def forward(self, x: Union[torch.Tensor, Complex], h_prev: Optional[Union[torch.Tensor, Complex]] = None):
        if self.tensor_type == "real":
            if h_prev is None:
                h_prev = torch.zeros((self.hidden_size, 1), **self.factory_kwargs)
            W = self.calc_Wh() - self.gamma * torch.eye(self.hidden_size)
            Vx = self.V(x)
            h_t = h_prev + self.epsilon * self.func(W @ h_prev + Vx)
            if self.proj_size is not None:
                output = self.Wo(h_t)
            else:
                output = h_t
            return output, h_t, self.calc_Wh()
        else:
            if h_prev is None:
                h_prev = Complex.zeros((self.hidden_size, self.hidden_size), **self.c_kwargs)
            W = self.calc_Wh() - self.gamma * Complex.eye(self.hidden_size, **self.c_kwargs)
            Vx = self.V(x)
            h_t = h_prev + self.epsilon * self.func(W @ h_prev + Vx)
            if self.proj_size is not None:
                output = self.Wo(h_t)
            else:
                output = h_t
            return output, h_t, self.calc_Wh()


class CTRNNCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 time_delta: float = 0.01,
                 time_bounds: Union[List[float], Tuple[float, float]] = None,
                 func: Union[str, Callable, nn.Module, DLModule] = None,
                 bias: bool = True,
                 proj_size: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        self.Wx = nn.Linear(
            in_features=input_size,
            out_features=hidden_size,
            bias=bias,
            **self.factory_kwargs
        )
        self.Wh = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=bias,
            **self.factory_kwargs
        )
        self.theta = nn.Parameter(torch.zeros((hidden_size, 1), **self.factory_kwargs))
        self.tau = nn.Parameter(torch.ones((hidden_size, 1), **self.factory_kwargs))
        self.y_init = nn.Parameter(torch.randn((hidden_size, 1), **self.factory_kwargs) * 0.01)

        self.Wo = nn.Linear(
            in_features=hidden_size,
            out_features=proj_size,
            bias=bias,
            **self.factory_kwargs
        ) if proj_size is not None else None
        self.proj_size = proj_size
        self.dt = time_delta

        if time_bounds is None:
            time_bounds = [0.0, time_delta]
        if len(time_bounds) > 2:
            time_bounds = time_bounds[:2]
        elif len(time_bounds) == 1:
            temp = math.fabs(float(time_bounds[0]))
            time_bounds = [temp - self.dt, temp + self.dt]
        elif len(time_bounds) == 0:
            time_bounds = [0.0, self.dt]
        t_min, t_max = time_bounds
        self.soft_clamp = lambda z: t_min + ((t_max - t_min) / 2.0) * (
                1.0 + torch.tanh((2 * z - (t_min + t_max)) / (t_max - t_min)))
        self.func = self._resolve_funcs(func, *args, **kwargs)

    def forward(self, x: torch.Tensor, y_prev: Optional[torch.Tensor] = None):
        if y_prev is None:
            y_prev = self.y_init

        tau = self.soft_clamp(self.tau) + self.dt
        slope = (-y_prev + self.func(self.Wx(x) + self.Wh(y_prev) + self.theta)) / tau

        y_next = y_prev + slope * self.dt
        if self.proj_size is not None:
            y_next = self.Wo(y_next)
        return y_next


class StackRNNCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 stack_size: int = None,
                 funcs: Union[List[Tuple[str, Callable, nn.Module]],
                 Tuple[Union[str, Callable, nn.Module]],
                 Dict[str, Union[str, Callable, nn.Module]]] = None,
                 max_stack_vectors: int = 256,
                 bias: bool = True,
                 k: int = 1,
                 proj_size: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        if stack_size is None:
            stack_size = hidden_size
        if funcs is None:
            funcs = ["sigmoid", "tanh", "sigmoid", "sigmoid"]

        self.Wxh = nn.Linear(
            in_features=input_size,
            out_features=hidden_size,
            bias=bias,
            **self.factory_kwargs
        )
        self.Whh = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=bias,
            **self.factory_kwargs
        )
        self.Wsh = nn.Linear(
            in_features=k * stack_size,
            out_features=hidden_size,
            bias=bias,
            **self.factory_kwargs
        )
        self.Whd = nn.Linear(
            in_features=hidden_size,
            out_features=stack_size,
            bias=bias,
            **self.factory_kwargs
        )
        self.Whu = nn.Linear(
            in_features=hidden_size,
            out_features=1,
            bias=bias,
            **self.factory_kwargs
        )
        self.Whv = nn.Linear(
            in_features=hidden_size,
            out_features=1,
            bias=bias,
            **self.factory_kwargs
        )
        self.bh = nn.Parameter(torch.zeros((hidden_size,), **self.factory_kwargs))
        self.bd = nn.Parameter(torch.zeros((stack_size,), **self.factory_kwargs))
        self.bu = nn.Parameter(torch.zeros((1,), **self.factory_kwargs))
        self.bv = nn.Parameter(torch.zeros((1,), **self.factory_kwargs))
        self.k = k
        self.stack_size = stack_size
        self.hidden_size = hidden_size
        self.Wo = nn.Linear(
            in_features=hidden_size,
            out_features=proj_size,
            bias=bias,
            **self.factory_kwargs
        ) if proj_size is not None else None
        self.proj_size = proj_size
        self.funcs = self._resolve_funcs(funcs)
        self.max_stack_vectors = max_stack_vectors

    def forward(self, x: torch.Tensor,
                h_prev: Optional[torch.Tensor] = None,
                s_prev: Optional[torch.Tensor] = None,
                stack_content: Optional[torch.Tensor] = None,
                stack_strengths: Optional[torch.Tensor] = None):
        if x.dim() == 2:
            batch, _ = x.size()
            seq_len = 1
        else:
            batch, seq_len, _ = x.size()
        if h_prev is None:
            h_prev = torch.zeros((batch, self.hidden_size), **self.factory_kwargs)
        if s_prev is None:
            s_prev = torch.zeros((batch, self.k * self.stack_size), **self.factory_kwargs)
        if stack_content is None:
            stack_content = torch.zeros((batch, seq_len, self.stack_size), **self.factory_kwargs)
        if stack_strengths is None:
            stack_strengths = torch.zeros((batch, seq_len), **self.factory_kwargs)

        sig_h, tanh_d, sig_u, sig_v = self.funcs

        h_t = sig_h(self.Wxh(x) + self.Whh(h_prev) + self.Wsh(s_prev) + self.bh)

        d_t = tanh_d(self.Whd(h_t) + self.bd)
        u_t = sig_u(self.Whu(h_t) + self.bu)
        v_t = sig_v(self.Whv(h_t) + self.bv)

        if stack_strengths.size(1) > 0:
            strengths_above = torch.cumsum(stack_strengths.flip(dims=[1]), dim=1).flip(dims=[1])
            strengths_above = torch.cat([strengths_above[:, 1:], torch.zeros_like(v_t)], dim=1)

            pop_pressure = torch.clamp(v_t - strengths_above, min=0)
            new_strengths = torch.clamp(stack_strengths - pop_pressure, min=0)
        else:
            new_strengths = stack_strengths

        new_stack_content = torch.cat([stack_content, d_t.unsqueeze(1)], dim=1)
        new_stack_strengths = torch.cat([new_strengths, u_t], dim=1)

        if new_stack_strengths.size(1) > self.max_stack_vectors:
            new_stack_content = new_stack_content[:, -self.max_stack_vectors, ...]
            new_stack_strengths = new_stack_strengths[:, -self.max_stack_vectors, ...]

        read_vectors = []
        curr_above = torch.cumsum(new_stack_strengths.flip(dims=[1]), dim=1).flip(dims=[1])
        curr_above = torch.cat([curr_above[:, 1:], torch.zeros_like(v_t)], dim=1)

        for j in range(1, self.k + 1):
            lower_bound = j - 1
            upper_bound = j

            weight_j = torch.clamp(torch.min(new_stack_strengths + curr_above, torch.tensor(float(upper_bound))) -
                                   torch.max(curr_above, torch.tensor(float(lower_bound))), min=0)
            vec_j = torch.sum(weight_j.unsqueeze(-1) * new_stack_content, dim=1)
            read_vectors.append(vec_j)
        s_t = torch.cat(read_vectors, dim=1)

        if self.proj_size is not None:
            h_t = self.Wo(h_t)

        return h_t, s_t, new_stack_content, new_stack_strengths


class VariationalRecurrentUnitCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 latent_size: int,
                 z_dim: int,
                 non_linearity: Union[str, Callable, nn.Module] = 'tanh',
                 funcs: Union[List[Tuple[str, Callable, nn.Module]],
                 Tuple[Union[str, Callable, nn.Module]],
                 Dict[str, Union[str, Callable, nn.Module]]] = None,
                 enc_funcs: Union[List[Tuple[str, Callable, nn.Module]],
                 Tuple[Union[str, Callable, nn.Module]],
                 Dict[str, Union[str, Callable, nn.Module]]] = None,
                 dec_funcs: Union[List[Tuple[str, Callable, nn.Module]],
                 Tuple[Union[str, Callable, nn.Module]],
                 Dict[str, Union[str, Callable, nn.Module]]] = None,
                 bias: bool = True,
                 cell_type: str = "rnn",
                 generative: bool = False,
                 proj_size: int = 0,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        cell_kwargs = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "non_linearity": non_linearity,
            "funcs": funcs,
            "bias": bias,
            "proj_size": proj_size,
            **kwargs,
            **self.factory_kwargs
        }
        cell_type = cell_type.lower()
        self.cell_type = cell_type
        if cell_type == "lstm":
            self.cell = LSTMCell(**cell_kwargs)
        elif cell_type == "gru":
            self.cell = GRUCell(**cell_kwargs)
        else:
            self.cell = RNNCell(**cell_kwargs)
        lin_kwargs = {
            "bias": bias,
            **self.factory_kwargs
        }
        self.Wfx = nn.Linear(
            in_features=input_size,
            out_features=latent_size,
            **lin_kwargs
        )
        self.Wfz = nn.Linear(
            in_features=z_dim,
            out_features=latent_size,
            **lin_kwargs
        )
        if generative:
            self.W_start = nn.Linear(
                in_features=hidden_size,
                out_features=z_dim * 2,
                **lin_kwargs
            )
            self.Wz = nn.Linear(
                in_features=2 * z_dim,
                out_features=z_dim,
                **lin_kwargs
            )
        self.generative = generative
        if enc_funcs is None:
            enc_funcs = ["relu", "relu", "relu"]
        enc_funcs = self._resolve_funcs(enc_funcs, *args, **kwargs)
        self.enc_layers = nn.ModuleList([
            nn.Linear(
                in_features=latent_size + hidden_size,
                out_features=hidden_size,
                **lin_kwargs
            )
        ])
        for i, func in enumerate(enc_funcs):
            self.enc_layers.append(func)
            if i == len(enc_funcs) - 1:
                self.enc_layers.append(
                    nn.Linear(
                        in_features=hidden_size,
                        out_features=z_dim * 2,
                        **lin_kwargs
                    )
                )
                continue
            self.enc_layers.append(
                nn.Linear(
                    in_features=hidden_size,
                    out_features=hidden_size,
                    **lin_kwargs
                )
            )
        if dec_funcs is None:
            dec_funcs = ["relu", "relu", "relu"]
        dec_funcs = self._resolve_funcs(dec_funcs, *args, **kwargs)
        self.dec_layers = nn.ModuleList([
            nn.Linear(
                in_features=latent_size + hidden_size,
                out_features=hidden_size,
                **lin_kwargs
            )
        ])
        for i, func in enumerate(dec_funcs):
            self.dec_layers.append(func)
            if i == len(dec_funcs) - 1:
                self.dec_layers.append(
                    nn.Linear(
                        in_features=hidden_size,
                        out_features=2 * input_size,
                        **lin_kwargs
                    )
                )
                continue
            self.dec_layers.append(
                nn.Linear(
                    in_features=hidden_size,
                    out_features=hidden_size,
                    **lin_kwargs
                )
            )
        self.W_pre_rn = nn.Linear(
            in_features=2 * latent_size,
            out_features=input_size,
            **lin_kwargs
        )
        self.Wfx_new = nn.Linear(
            in_features=input_size,
            out_features=latent_size,
            **lin_kwargs
        )
        self.hidden_size = hidden_size
        self.z_dim = z_dim
        self.Wo = nn.Linear(
            in_features=hidden_size,
            out_features=proj_size,
            bias=bias,
            **self.factory_kwargs
        ) if proj_size is not None else None
        self.proj_size = proj_size

    def _calc_enc_out(self, phi_x: torch.Tensor, h_prev: torch.Tensor):
        x_tot = torch.cat([phi_x, h_prev], dim=-1)
        for layer in self.enc_layers:
            x_tot = layer(x_tot)
        return x_tot

    def _calc_dec_out(self, phi_z: torch.Tensor, h_prev: torch.Tensor):
        x_tot = torch.cat([phi_z, h_prev], dim=-1)
        for layer in self.dec_layers:
            x_tot = layer(x_tot)
        return x_tot

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None,
                c_prev: Optional[torch.Tensor] = None):
        if h_prev is None:
            h_prev = torch.zeros((x.size(0), self.hidden_size), **self.factory_kwargs)
        if c_prev is None:
            c_prev = torch.zeros((x.size(0), self.hidden_size), **self.factory_kwargs)
        phi_x_old = self.Wfx(x)
        z_comp = self._calc_enc_out(phi_x_old, h_prev)
        z_mu, z_logvar = torch.chunk(z_comp, chunks=2, dim=-1)
        z_sig = torch.exp(0.5 * z_logvar)
        eps = torch.rand_like(z_sig)
        z_t = z_mu + (eps * z_sig)
        if self.generative:
            z_gen = self.W_start(h_prev)
            z_gen_mu, z_gen_logvar = torch.chunk(z_gen, chunks=2, dim=-1)
            z_gen_sig = torch.exp(0.5 * z_gen_logvar)
            z_gen_eps = torch.rand_like(z_gen_sig)
            z_gen_t = z_gen_mu + (z_gen_eps * z_gen_sig)
            z_t = torch.stack([z_t, z_gen_t], dim=-1)
            z_t = self.Wz(z_t)
        phi_z = self.Wfz(z_t)
        x_comp = self._calc_dec_out(phi_z, h_prev)
        x_mu, x_logvar = torch.chunk(x_comp, chunks=2, dim=-1)
        x_sig = torch.exp(x_logvar)
        x_eps = torch.rand_like(x_sig)
        x_new = x_mu + (x_eps * x_sig)
        phi_x_new = self.Wfx_new(x_new)
        x = torch.stack([phi_x_new, phi_z], dim=-1)
        x = self.W_pre_rn(x)
        if self.cell_type == "lstm":
            h_t, c_t = self.cell(x, h_prev, c_prev)
            return h_t, c_t
        else:
            h_t = self.cell(x, h_prev)
            if self.proj_size is not None:
                h_t = self.Wo(h_t)
                return h_t


class NARXCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 proj_size: int = None,
                 non_linearity: Union[str, Callable, nn.Module, DLModule] = 'tanh',
                 bias: bool = True,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()

        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        self.hidden_size = hidden_size
        lin_kwargs = {
            "bias": bias,
            **self.factory_kwargs
        }
        if proj_size is None:
            proj_size = hidden_size

        self.Wz = nn.Linear(
            in_features=input_size + hidden_size,
            out_features=hidden_size,
            **lin_kwargs
        )

        self.Wout = nn.Linear(
            in_features=hidden_size,
            out_features=proj_size,
            **lin_kwargs
        )

        self.func = self._resolve_funcs(non_linearity)

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None):
        if h_prev is None:
            h_prev = torch.zeros((x.size(0), self.hidden_size), **self.factory_kwargs)
        z_t = torch.cat([x, h_prev], dim=-1)
        a_t = self.Wz(z_t)
        h_t = self.func(a_t)
        y_t = self.Wout(h_t)
        return y_t, h_t


class MorgifierRecurrentUnitCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_rounds: int = 3,
                 non_linearity: Union[str, Callable, nn.Module] = 'tanh',
                 funcs: Union[List[Tuple[str, Callable, nn.Module]],
                 Tuple[Union[str, Callable, nn.Module]],
                 Dict[str, Union[str, Callable, nn.Module]]] = None,
                 bias: bool = True,
                 cell_type: str = "lstm",
                 mogrification_funcs: Union[List[Union[str, Callable, nn.Module, DLModule]],
                 Tuple[Union[str, Callable, nn.Module, DLModule]],
                 Dict[str, Union[str, Callable, nn.Module, DLModule]],
                 str, nn.Module, Callable, DLModule] = "sigmoid",
                 proj_size: int = 0,
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
        cell_kwargs = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "non_linearity": non_linearity,
            "funcs": funcs,
            "proj_size": proj_size,
            **kwargs,
            **lin_kwargs
        }
        if num_rounds % 2 == 0:
            r_odd = num_rounds // 2
            r_even = r_odd
        else:
            r_even = num_rounds // 2
            r_odd = r_even + 1
        self.Q = nn.ModuleList([
            nn.Linear(
                in_features=hidden_size,
                out_features=input_size,
                **lin_kwargs
            ) for _ in range(r_odd)
        ])
        self.R = nn.ModuleList([
            nn.Linear(
                in_features=input_size,
                out_features=hidden_size,
                **lin_kwargs
            ) for _ in range(r_even)
        ])
        if isinstance(mogrification_funcs, (str, type(None))) or callable(mogrification_funcs) or isinstance(mogrification_funcs, (nn.Module, DLModule)):
            mogrification_funcs = [mogrification_funcs] * num_rounds
        self.mogrification_funcs = self._resolve_funcs(mogrification_funcs, *args, **kwargs)
        if len(self.mogrification_funcs) > num_rounds:
            self.mogrification_funcs = self.mogrification_funcs[:num_rounds]
        elif len(self.mogrification_funcs) < num_rounds:
            rem = num_rounds - len(self.mogrification_funcs)
            self.mogrification_funcs = nn.ModuleList([*self.mogrification_funcs, *self.mogrification_funcs[:rem]])
        self.cell_type = cell_type.lower()
        if self.cell_type == "rnn":
            self.cell = RNNCell(**cell_kwargs)
        elif self.cell_type == "lstm":
            self.cell = LSTMCell(**cell_kwargs)
        elif self.cell_type == "gru":
            self.cell = GRUCell(**cell_kwargs)
        else:
            self.cell = LSTMCell(**cell_kwargs)
        self.Wo = nn.Linear(
            in_features=input_size,
            out_features=proj_size,
            bias=bias,
            **self.factory_kwargs
        ) if proj_size is not None else None
        self.proj_size = proj_size
        self.num_rounds = num_rounds
        self.hidden_size = hidden_size

    def mogrification_process(self, x: torch.Tensor, h_prev: torch.Tensor):
        curr_x = x
        curr_h = h_prev
        q_idx = 0
        r_idx = 0
        num_rounds = self.num_rounds
        for r in range(1, num_rounds + 1):
            if r % 2 != 0:
                gate = 2 * self.mogrification_funcs[r - 1](self.Q[q_idx](curr_h))
                curr_x = gate * curr_x
                q_idx += 1
            else:
                gate = 2 * self.mogrification_funcs[r - 1](self.R[r_idx](curr_x))
                curr_h = gate * curr_h
                r_idx += 1
        return curr_x, curr_h

    def forward(self, x: torch.Tensor,
                h_prev: Optional[torch.Tensor] = None,
                c_prev: Optional[torch.Tensor] = None):
        if h_prev is None:
            h_prev = torch.zeros((x.size(0), self.hidden_size), **self.factory_kwargs)
        if c_prev is None:
            c_prev = torch.zeros((x.size(0), self.hidden_size), **self.factory_kwargs)
        x, h_prev = self.mogrification_process(x, h_prev)
        if self.cell_type == "lstm":
            h_t, c_t = self.cell(x, h_prev, c_prev)
            return h_t, c_t
        else:
            h_t = self.cell(x, h_prev)
            if self.proj_size is not None:
                h_t = self.Wo(h_t)
            return h_t
