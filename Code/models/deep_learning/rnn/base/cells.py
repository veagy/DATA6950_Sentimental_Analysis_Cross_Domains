"""Base RNN cells: RNNCell, LSTMCell, GRUCell."""
import torch
import torch.nn as nn
from typing import Optional, Union, Callable, List, Tuple, Dict
from .....models.utils import DLModule


class RNNCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 non_linearity: Union[str, Callable, nn.Module] = 'tanh',
                 funcs: Union[List, Tuple, Dict] = None,
                 bias: bool = True,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }

        self.Wih = nn.Linear(
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

        # Resolve Non-Linearity from funcs if provided
        if funcs is not None:
            if isinstance(funcs, (list, tuple)) and len(funcs) > 0:
                non_linearity = funcs[0]
            elif isinstance(funcs, dict) and len(funcs) > 0:
                non_linearity = list(funcs.values())[0]
        self.func = self._resolve_funcs(non_linearity, *args, **kwargs)

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        if h_prev is None:
            h_prev = torch.zeros((x.size(0), self.hidden_size), **self.factory_kwargs)
        return self.func(self.Wih(x) + self.Whh(h_prev))


class LSTMCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 funcs: Union[List[Tuple[str, Callable, nn.Module]],
                 Tuple[Union[str, Callable, nn.Module]],
                 Dict[str, Union[str, Callable, nn.Module]]] = None,
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
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj_size = proj_size

        # Determine recurrent input size (hidden_size or proj_size)
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

        # Input Gates
        self.Wii = nn.Linear(**in_lin_kwargs)
        self.Wif = nn.Linear(**in_lin_kwargs)
        self.Wig = nn.Linear(**in_lin_kwargs)
        self.Wio = nn.Linear(**in_lin_kwargs)

        # Recurrent Gates
        self.Whi = nn.Linear(**h_lin_kwargs)
        self.Whf = nn.Linear(**h_lin_kwargs)
        self.Whg = nn.Linear(**h_lin_kwargs)
        self.Who = nn.Linear(**h_lin_kwargs)

        # Projection (Optional)
        if proj_size > 0:
            self.Wo = nn.Linear(**out_lin_kwargs)
        else:
            self.Wo = None

        # Activation Functions
        if funcs is None:
            # Default: sigmoid for gates, tanh for cell/hidden
            funcs = ["sigmoid", "sigmoid", "tanh", "sigmoid", "tanh"]

        # Normalize funcs to list of callables/modules
        self.funcs = self._resolve_funcs(funcs, *args, **kwargs)

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None, c_prev: Optional[torch.Tensor] = None) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        if h_prev is None:
            h_prev = torch.zeros((self.hidden_size, 1), **self.factory_kwargs)
        if c_prev is None:
            c_prev = torch.zeros((self.hidden_size, 1), **self.factory_kwargs)

        f_sig_i, f_sig_f, f_tanh_g, f_sig_o, f_tanh_h = self.funcs

        i_t = f_sig_i(self.Wii(x) + self.Whi(h_prev))
        f_t = f_sig_f(self.Wif(x) + self.Whf(h_prev))
        g_t = f_tanh_g(self.Wig(x) + self.Whg(h_prev))
        o_t = f_sig_o(self.Wio(x) + self.Who(h_prev))

        c_t = (f_t * c_prev) + (i_t * g_t)
        h_t = o_t * f_tanh_h(c_t)

        if self.Wo is not None:
            h_t = self.Wo(h_t)

        return h_t, c_t


class GRUCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 funcs: Union[List[Tuple[str, Callable, nn.Module]],
                 Tuple[Union[str, Callable, nn.Module]],
                 Dict[str, Union[str, Callable, nn.Module]]],
                 bias: bool = True,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        in_kwargs = {
            "in_features": input_size,
            "out_features": hidden_size,
            "bias": bias,
            "device": device,
            "dtype": dtype
        }
        h_kwargs = {
            "in_features": hidden_size,
            "out_features": hidden_size,
            "bias": bias,
            "device": device,
            "dtype": dtype
        }
        self.Wir = nn.Linear(**in_kwargs)
        self.Wiz = nn.Linear(**in_kwargs)
        self.Win = nn.Linear(**in_kwargs)

        self.Whr = nn.Linear(**h_kwargs)
        self.Whz = nn.Linear(**h_kwargs)
        self.Whn = nn.Linear(**h_kwargs)

        if funcs is None:
            funcs = ["sigmoid", "sigmoid", "tanh"]

        self.funcs = self._resolve_funcs(funcs, *args, **kwargs)

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        if h_prev is None:
            h_prev = torch.zeros((x.size(0), self.hidden_size), device=x.device, dtype=x.dtype)
        r_sig, z_sig, n_tanh = self.funcs
        r = r_sig(self.Wir(x) + self.Whr(h_prev))
        z = z_sig(self.Wiz(x) + self.Whz(h_prev))
        n = n_tanh(self.Win(x) + (r * self.Whn(h_prev)))
        h_t = ((1 - z) * n) + (z * h_prev)
        return h_t
