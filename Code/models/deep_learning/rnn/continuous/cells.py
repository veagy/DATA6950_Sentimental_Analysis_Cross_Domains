"""Continuous RNN cells: NeuralODECell, LTCCell, CfCCell."""
import warnings
import torch
import torch.nn as nn
from typing import Optional, Union, Callable, List, Tuple, Dict
from .....models.utils import DLModule

from ..base import LSTMCell, GRUCell


class NeuralODECell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 nn_module: Union[nn.Module, DLModule, dict],
                 funcs: Union[List[Tuple[str, Callable, nn.Module]],
                           Tuple[Union[str, Callable, nn.Module]],
                           Dict[str, Union[str, Callable, nn.Module]]] = None,
                 update: Union[str, nn.Module, DLModule] = "auto",
                 solve_method: str = "rk_4",
                 bias: bool = True,
                 proj_size: int = 0,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        g_kwargs = {
            "args": args,
            "kwargs": kwargs
        }
        if isinstance(nn_module, str) and nn_module.lower() == "auto":
            # Create default MLP
            self.f_theta = nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=bias, **self.factory_kwargs),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size, bias=bias, **self.factory_kwargs)
            )
        elif isinstance(nn_module, (nn.Module, DLModule)):
            self.f_theta = nn_module

        elif isinstance(nn_module, dict):
            from ...ffnn.nn_models import FeedForwardNeuralNetwork, FeedForwardNeuralNetworkOp
            nn_kwargs = {**nn_module, **self.factory_kwargs, **g_kwargs}
            try:
                self.f_theta = FeedForwardNeuralNetwork(**nn_kwargs)
            except Exception:
                try:
                    self.f_theta = FeedForwardNeuralNetworkOp(**nn_kwargs)
                except Exception:
                    from ...models import DLModelLayers
                    try:
                        self.f_theta = DLModelLayers(**nn_kwargs)
                    except Exception:
                        warnings.warn(f"The models are not fitting for given config."
                                      f"Writing a default nn module", UserWarning)
                        self.f_theta = nn.Sequential(
                            nn.Linear(
                                in_features=input_size,
                                out_features=hidden_size,
                                bias=bias,
                                **self.factory_kwargs
                            ),
                            nn.Sigmoid(),
                            nn.Linear(
                                in_features=hidden_size,
                                out_features=hidden_size,
                                bias=bias,
                                **self.factory_kwargs
                            )
                        )
        input_kwargs = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "funcs": funcs,
            "bias": bias,
            "proj_size": proj_size,
            **self.factory_kwargs,
            **g_kwargs
        }
        if isinstance(update, str):
            update = update.lower()
            if update == "auto":
                self.W = nn.Linear(
                    in_features=input_size,
                    out_features=hidden_size,
                    bias=bias,
                    **self.factory_kwargs
                )
                self.U = nn.Linear(
                    in_features=hidden_size,
                    out_features=hidden_size,
                    bias=bias,
                    **self.factory_kwargs
                )
                self.bias = nn.Parameter(
                    torch.zeros((hidden_size,), **self.factory_kwargs)
                )
                self.update = update

            elif update == "lstm":
                self.update = LSTMCell(**input_kwargs)
            elif update == "gru":
                self.update = GRUCell(**input_kwargs)
        elif isinstance(update, (nn.Module, DLModule)):
            self.update = update

        # Projection (Optional)
        if proj_size > 0:
            self.Wo = nn.Linear(
                in_features=hidden_size,
                out_features=proj_size,
                bias=bias,
                **self.factory_kwargs
            )
        else:
            self.Wo = None
        self.proj_size = proj_size

        if funcs is None:
            # Default: sigmoid for gates, tanh for cell/hidden
            if update == "auto":
                funcs = ["sigmoid"]
            else:
                funcs = ["sigmoid", "sigmoid", "tanh"]
        self.funcs = self._resolve_funcs(funcs, *args, **kwargs)
        self.solve_method = solve_method

    def ode_solver(self, h, t_span, method="rk_4"):
        if method == 'euler':
            return self._euler(h, t_span)
        elif method == 'rk4':
            return self._rk4(h, t_span)
        elif method == 'dormand_prince':
            return self._dormand_prince(h, t_span)
        else:
            warnings.warn(f"The given method {method} is not supported."
                          f"Changing to default 'rk_4' method.")
            return self._rk4(h, t_span)

    def _euler(self, h, t_span):
        """Method 1: Euler Integration"""
        for i in range(len(t_span) - 1):
            dt = t_span[i + 1] - t_span[i]
            h = h + dt * self.f_theta(h)
        return h

    def _rk4(self, h, t_span):
        """Method 2: Runge-Kutta 4th Order"""
        for i in range(len(t_span) - 1):
            dt = t_span[i + 1] - t_span[i]

            k1 = self.f_theta(h)
            k2 = self.f_theta(h + dt * k1 / 2)
            k3 = self.f_theta(h + dt * k2 / 2)
            k4 = self.f_theta(h + dt * k3)

            h = h + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return h

    def _dormand_prince(self, h, t_span):
        """
        Method 3: Dormand-Prince (RK45)
        Note: This implementation uses a fixed-step version of the
        Dormand-Prince coefficients for consistency with the others.
        """
        # Simplified Dormand-Prince step logic
        for i in range(len(t_span) - 1):
            dt = t_span[i + 1] - t_span[i]

            k1 = self.f_theta(h)
            k2 = self.f_theta(h + dt * (1 / 5) * k1)
            k3 = self.f_theta(h + dt * (3 / 40 * k1 + 9 / 40 * k2))
            k4 = self.f_theta(h + dt * (44 / 45 * k1 - 56 / 15 * k2 + 32 / 9 * k3))
            k5 = self.f_theta(h + dt * (19372 / 6561 * k1 - 25360 / 2187 * k2 + 64448 / 6561 * k3 - 212 / 729 * k4))
            k6 = self.f_theta(
                h + dt * (9017 / 3168 * k1 - 355 / 33 * k2 + 46732 / 5247 * k3 + 49 / 176 * k4 - 5103 / 18656 * k5))

            # 5th order update
            h = h + dt * (35 / 384 * k1 + 500 / 1113 * k3 + 125 / 192 * k4 - 2187 / 6784 * k5 + 11 / 84 * k6)
        return h

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor], t_span: Optional[torch.Tensor]):
        h_bar = self.ode_solver(h_prev, t_span, method=self.solve_method)

        if isinstance(self.update, str) and self.update == "auto":
            # self.funcs is ModuleList. Use first element.
            h = self.funcs[0](self.W(x) + self.U(h_bar) + self.bias)
        else:
            h = self.update(h_bar, x)

        if self.proj_size is not None and self.proj_size > 0:
            h = self.Wo(h)
        return h


class LTCCell(DLModule):
    def __init__(self, input_size: int,
                 hidden_size: int,
                 seq_len: int,
                 funcs: Union[List[Union[str, Callable, nn.Module, DLModule]],
                 Tuple[Union[str, Callable, nn.Module, DLModule]],
                 Dict[str, Union[str, Callable, nn.Module, DLModule]],
                 str, Callable, nn.Module, DLModule] = "sigmoid",
                 bias: bool = True,
                 time_delta: float = 0.1,
                 solver_type: str = "euler",
                 proj_size: int = None,
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
        self.dt = time_delta
        self.solver_type = solver_type.lower()
        self.W = nn.ParameterList([
            nn.Parameter(
                torch.randn((hidden_size, input_size), **self.factory_kwargs)
            ) for _ in range(seq_len)
        ])
        self.mu = nn.ParameterList([
            nn.Parameter(
                torch.randn((hidden_size, input_size), **self.factory_kwargs)
            ) for _ in range(seq_len)
        ])
        self.gamma = nn.ParameterList([
            nn.Parameter(
                torch.randn((hidden_size, input_size), **self.factory_kwargs)
            ) for _ in range(seq_len)
        ])
        self.E = nn.ParameterList([
            nn.Parameter(
                torch.randn((hidden_size, input_size), **self.factory_kwargs)
            ) for _ in range(seq_len)
        ])
        if isinstance(funcs, (str, nn.Module, Callable, DLModule)):
            funcs = [funcs] * seq_len
        self.funcs = self._resolve_funcs(funcs, *args, **kwargs)
        if len(self.funcs) < seq_len:
            rem = seq_len - len(self.funcs)
            self.funcs = nn.ModuleList([*self.funcs, *self.funcs[:rem]])
        elif len(self.funcs) > seq_len:
            self.funcs = self.funcs[:seq_len]
        self.seq_len = seq_len
        self.G_leak = nn.Parameter(torch.randn((hidden_size, 1), **self.factory_kwargs))
        self.E_leak = nn.Parameter(torch.randn((hidden_size, 1), **self.factory_kwargs))
        self.Wo = nn.Linear(
            in_features=input_size,
            out_features=proj_size,
            bias=bias,
            **self.factory_kwargs
        ) if proj_size is not None else None
        self.proj_size = proj_size
        self.hidden_size = hidden_size

    def _standardize(self, In: torch.Tensor, mu: torch.Tensor, gamma: torch.Tensor):
        return gamma * (In - mu)

    def _calc_weighted_func(self, In: List[torch.Tensor]):
        if len(In) < self.seq_len:
            rem = self.seq_len - len(In)
            temp = [torch.zeros_like(In[0], device=In[0].device, dtype=In[0].dtype) for _ in range(rem)]
            In = [*In, *temp]
        elif len(In) > self.seq_len:
            In = torch.zeros_like(In[0], device=In[0].device, dtype=In[0].dtype)
        out = torch.zeros_like(In[0])
        for i in range(self.seq_len):
            I_standard = self._standardize(In[i], self.mu[i], self.gamma[i])
            out += self.W[i] * self.funcs[i](I_standard)
        return out

    def _calc_weighted_pot_func(self, In: List[torch.Tensor]):
        if len(In) < self.seq_len:
            rem = self.seq_len - len(In)
            temp = [torch.zeros_like(In[0], device=In[0].device, dtype=In[0].dtype) for _ in range(rem)]
            In = [*In, *temp]
        elif len(In) > self.seq_len:
            In = torch.zeros_like(In[0], device=In[0].device, dtype=In[0].dtype)
        out = torch.zeros_like(In[0])
        for i in range(self.seq_len):
            I_standard = self._standardize(In[i], self.mu[i], self.gamma[i])
            out += self.W[i] * self.funcs[i](I_standard) * self.E[i]
        return out

    def _calc_f(self, x: torch.Tensor, In: List[torch.Tensor]):
        x = x.unsqueeze(-2).expand(..., self.hidden_size, -1)
        out = self.G_leak * self.E_leak + self._calc_weighted_pot_func(In)
        out = out - ((self.G_leak + self._calc_weighted_func(In)) * x)
        return out

    def _calc_tau_sys(self, In: List[torch.Tensor]):
        if len(In) < self.seq_len:
            rem = self.seq_len - len(In)
            temp = [torch.zeros_like(In[0], device=In[0].device, dtype=In[0].dtype) for _ in range(rem)]
            In = [*In, *temp]
        elif len(In) > self.seq_len:
            In = torch.zeros_like(In[0], device=In[0].device, dtype=In[0].dtype)
        out = self.G_leak
        for i in range(self.seq_len):
            out += self.W[i] * self.funcs[i](In[i])
        out += torch.tensor(float(1e-8), **self.factory_kwargs)
        out = 1.0 / out
        return out

    def _ode_solver(self, x: torch.Tensor, In: List[torch.Tensor]):
        if self.solver_type == "euler":
            x_t = x + self._calc_f(x, In) * (self.dt / self._calc_tau_sys(In))
        elif self.solver_type == "implicit":
            dt = self.dt / self._calc_tau_sys(In)
            num = x + dt * (self.G_leak * self.E_leak + self._calc_weighted_pot_func(In))
            den = 1 + dt * (self.G_leak + self._calc_weighted_func(In)) + 1e-6
            x_t = num / den
        elif self.solver_type == "rk4":
            dt = self.dt / self._calc_tau_sys(In)
            k1 = self._calc_f(x, In)
            k2 = self._calc_f(x + (dt / 2) * k1, In)
            k3 = self._calc_f(x + (dt / 2) * k2, In)
            k4 = self._calc_f(x + dt * k3, In)
            x_t = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        elif self.solver_type == "cfc":
            num = self.G_leak * self.E_leak + self._calc_weighted_pot_func(In)
            den = self.G_leak + self._calc_weighted_func(In) + 1e-6
            target = num / den
            x_t = target
            x_t += (x - target) * torch.exp(-(self.G_leak + self._calc_weighted_func(In)))
        else:
            x_t = x + self._calc_f(x, In) * (self.dt / self._calc_tau_sys(In))
        if self.proj_size is not None:
            x_t = self.Wo(x_t)
        return x_t

    def forward(self, x: torch.Tensor, In: List[torch.Tensor]):
        return self._ode_solver(x, In)


class CfCCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 funcs: Union[List[Union[str, Callable, nn.Module, DLModule]],
                 Tuple[Union[str, Callable, nn.Module, DLModule]],
                 Dict[str, Union[str, Callable, nn.Module, DLModule]],
                 str, Callable, nn.Module, DLModule] = "sigmoid",
                 bias: bool = True,
                 time_delta: float = 0.1,
                 proj_size: int = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        self.lin_kwargs = {
            "bias": bias,
            **self.factory_kwargs
        }
        self.input_size = input_size
        self.hidden_size = hidden_size
        if isinstance(funcs, (str, Callable, nn.Module, DLModule)):
            # Fix: Don't multiply string content! Create list of duplicates.
            single_func = funcs
            funcs = [[single_func for _ in range(10)] for _ in range(3)]
        if isinstance(funcs, dict):
            funcs = [[v for v in func.values()] for func in funcs.values()]

        # Ensure funcs is list of lists
        if not isinstance(funcs, list):
            funcs = [[funcs]]

        if len(funcs) > 3:
            funcs = funcs[:3]
        elif len(funcs) < 3:
            rem = 3 - len(funcs)
            extension = funcs[:rem]
            funcs.extend(extension)
        self.funcs = nn.ModuleList([])
        for func in funcs:
            self.funcs.append(self._resolve_funcs(func, *args, **kwargs))
        self.Wo = nn.Linear(
            in_features=input_size,
            out_features=proj_size,
            bias=bias,
            **self.factory_kwargs
        ) if proj_size is not None else None
        self.proj_size = proj_size
        self.dt = time_delta
        self.ff_layers = self._get_ff_layers()

    def _get_ff_layers(self):
        ff_layers = []
        for i in range(3):
            func = self.funcs[i]
            layers = [
                nn.Linear(
                    in_features=self.input_size + self.hidden_size,
                    out_features=self.hidden_size,
                    **self.lin_kwargs
                )
            ]
            for f in func:
                layers.append(f)
                layers.append(
                    nn.Linear(
                        in_features=self.hidden_size,
                        out_features=self.hidden_size,
                        **self.lin_kwargs
                    )
                )
            ff_layers.append(nn.Sequential(*layers))
        return nn.ModuleList(ff_layers)

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None):
        if h_prev is None:
            h_prev = torch.zeros((x.size(0), self.hidden_size), **self.factory_kwargs)
        x_comb = torch.cat([x, h_prev], dim=-1)
        ff_layers = []
        for ff in self.ff_layers:
            ff_layers.append(ff(x_comb))
        ff1, ff2, ff3 = ff_layers
        h_t = ff1 * torch.exp(-ff2 * self.dt) + ff3
        if self.proj_size is not None:
            h_t = self.Wo(h_t)
        return h_t
