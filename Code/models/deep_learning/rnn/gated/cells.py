"""Gated RNN cells: SRUCell, QRNNCell, MGUCell, GORUCell, JANETCell."""
import math
import warnings
import torch
import torch.nn as nn
from typing import Optional, Union, Callable, List, Tuple, Dict
from .....models.utils import DLModule


class SRUCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 bias: bool = True,
                 funcs: Union[List[Tuple[str, Callable, nn.Module]],
                 Tuple[Union[str, Callable, nn.Module]],
                 Dict[str, Union[str, Callable, nn.Module]]] = None,
                 proj_size: int = None,
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
            **self.factory_kwargs
        }
        self.W = nn.Linear(bias=False, **in_kwargs)
        self.Wf = nn.Linear(bias=bias, **in_kwargs)
        self.Wr = nn.Linear(bias=bias, **in_kwargs)

        self.Wo = nn.Linear(
            in_features=hidden_size,
            out_features=proj_size,
            bias=bias,
            **self.factory_kwargs
        ) if proj_size is not None else None
        self.proj = False if proj_size is None else True

        if funcs is None:
            funcs = ["sigmoid", "sigmoid", "tanh"]
        self.funcs = self._resolve_funcs(funcs, *args, **kwargs)

        self.output_size = hidden_size
        self.use_skip = input_size != hidden_size
        if self.use_skip:
            self.Wx_skip = nn.Linear(input_size, hidden_size, bias=False, **self.factory_kwargs)
        else:
            self.Wx_skip = None

    def forward(self, x: torch.Tensor, c_prev: Optional[torch.Tensor]):
        if c_prev is None:
            c_prev = torch.zeros((x.size(0), self.hidden_size), **self.factory_kwargs)

        x_bar = self.W(x)
        sig_f, sig_r, tanh_c = self.funcs
        f_t = sig_f(self.Wf(x))
        r_t = sig_r(self.Wr(x))

        c_t = (f_t * c_prev) + ((1 - f_t) * x_bar)

        x_skip = self.Wx_skip(x) if self.use_skip else x

        h_t = (r_t * tanh_c(c_t)) + ((1 - r_t) * x_skip)

        if self.proj:
            h_t = self.Wo(h_t)
        return h_t, c_t


class QRNNCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 kernel_size: int,
                 dimensionality: Union[int, float] = 1,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 funcs: Union[List[Tuple[str, Callable, nn.Module]],
                 Tuple[Union[str, Callable, nn.Module]],
                 Dict[str, Union[str, Callable, nn.Module]]] = None,
                 groups: int = 1,
                 bias: bool = True,
                 ifo_pooling: bool = False,
                 padding_mode: str = 'zeros',
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        if isinstance(dimensionality, float):
            dimensionality = int(dimensionality)
        dimensionality = int(math.fabs(dimensionality))
        if dimensionality > 3:
            warnings.warn(f"Given dimensionality {dimensionality} is beyond range.\n"
                          f"setting to default value.")
            dimensionality = 1
        self.dimensionality = dimensionality
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        self.kernel_size = kernel_size
        self.ifo_pooling = ifo_pooling

        in_kwargs = {
            "in_channels": input_size,
            "out_channels": hidden_size,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "bias": bias,
            "padding_mode": padding_mode,
            **self.factory_kwargs
        }

        match self.dimensionality:
            case 1:
                self.Wz = nn.Conv1d(**in_kwargs)
                self.Wf = nn.Conv1d(**in_kwargs)
                self.Wo = nn.Conv1d(**in_kwargs)
                if self.ifo_pooling:
                    self.Wi = nn.Conv1d(**in_kwargs)
            case 2:
                self.Wz = nn.Conv2d(**in_kwargs)
                self.Wf = nn.Conv2d(**in_kwargs)
                self.Wo = nn.Conv2d(**in_kwargs)
                if self.ifo_pooling:
                    self.Wi = nn.Conv2d(**in_kwargs)
            case 3:
                self.Wz = nn.Conv3d(**in_kwargs)
                self.Wf = nn.Conv3d(**in_kwargs)
                self.Wo = nn.Conv3d(**in_kwargs)
                if self.ifo_pooling:
                    self.Wi = nn.Conv3d(**in_kwargs)

        if funcs is None:
            funcs = ["sigmoid", "sigmoid", "tanh"]
        self.funcs = self._resolve_funcs(funcs, *args, **kwargs)

    def process_gates(self, x: torch.Tensor):
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(2)
        else:
            if x.size(1) == self.factory_kwargs.get("in_channels") or x.size(1) == self.Wz.in_channels:
                pass
            elif x.size(2) == self.Wz.in_channels:
                x = x.transpose(1, 2)

        tanh_z, sig_f, sig_o = self.funcs
        z = tanh_z(self.Wz(x))
        f = sig_f(self.Wf(x))
        o = sig_o(self.Wo(x))

        if self.ifo_pooling:
            i = self.Wi(x)
        else:
            i = None

        if is_2d:
            z = z.squeeze(2)
            f = f.squeeze(2)
            o = o.squeeze(2)
            if i is not None:
                i = i.squeeze(2)
        else:
            target_len = x.size(2) if not is_2d else 1
            if z.size(2) > target_len:
                z = z[:, :, :target_len]
                f = f[:, :, :target_len]
                o = o[:, :, :target_len]
                if i is not None:
                    i = i[:, :, :target_len]

        return z, f, o, i

    def pool(self, c_prev, z, f, o, i=None):
        if not self.ifo_pooling:
            c_t = (f * c_prev) + ((1 - f) * z)
            h_t = o * c_t
            return h_t, c_t
        else:
            c_t = (f * c_prev) + (i * z)
            h_t = o * c_t
            return h_t, c_t

    def forward(self, x: torch.Tensor, c_prev: Optional[torch.Tensor]):
        if c_prev is None:
            c_prev = torch.zeros((x.size(0), self.factory_kwargs.get('out_channels', self.Wz.out_channels)),
                                 **self.factory_kwargs)

        z, f, o, i = self.process_gates(x)
        return self.pool(c_prev, z, f, o, i)


class MGUCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 funcs: Union[List[Tuple[str, Callable, nn.Module]],
                 Tuple[Union[str, Callable, nn.Module]],
                 Dict[str, Union[str, Callable, nn.Module]]] = None,
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
        lin_kwargs = {
            "bias": bias,
            **self.factory_kwargs
        }
        self.Wf = nn.Linear(
            in_features=input_size + hidden_size,
            out_features=hidden_size,
            **lin_kwargs
        )
        self.Wh = nn.Linear(
            in_features=input_size + hidden_size,
            out_features=hidden_size,
            **lin_kwargs
        )
        if funcs is None:
            funcs = ["sigmoid", "tanh"]
        funcs = funcs[:2]
        self.funcs = self._resolve_funcs(funcs, *args, **kwargs)
        self.Wo = nn.Linear(
            in_features=hidden_size,
            out_features=proj_size,
            bias=bias,
            **self.factory_kwargs
        ) if proj_size is not None else None
        self.proj_size = proj_size
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None):
        if h_prev is None:
            h_prev = torch.zeros((x.size(0), self.hidden_size), **self.factory_kwargs)
        x_comb = torch.cat([x, h_prev], dim=-1)
        sig, tanh = self.funcs
        f_t = sig(self.Wf(x_comb))
        f_bar = f_t * h_prev
        h_comb = torch.cat([x, f_bar], dim=-1)
        h_bar = tanh(self.Wh(h_comb))
        h_t = ((1 - f_t) * h_prev) + (f_t * h_bar)
        if self.proj_size is not None:
            h_t = self.Wo(h_t)
        return h_t


class GORUCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_reflections: int,
                 funcs: Union[List[Tuple[str, Callable, nn.Module]],
                 Tuple[Union[str, Callable, nn.Module]],
                 Dict[str, Union[str, Callable, nn.Module]]] = None,
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
        self.Wr = nn.Linear(**in_kwargs)
        self.Wz = nn.Linear(**in_kwargs)
        self.Wh = nn.Linear(**in_kwargs)

        h_kwargs = {
            "in_features": hidden_size,
            "out_features": hidden_size,
            "bias": bias,
            **self.factory_kwargs
        }
        self.Ur = nn.Linear(**h_kwargs)
        self.Uz = nn.Linear(**h_kwargs)

        b_kwargs = {
            "size": (hidden_size,),
            **self.factory_kwargs
        }
        init_bias = torch.zeros(**b_kwargs)
        self.br = nn.Parameter(init_bias)
        self.bz = nn.Parameter(init_bias)
        self.bh = nn.Parameter(init_bias)

        self.v = nn.ParameterList([
            nn.Parameter(torch.ones(**b_kwargs)) for _ in range(num_reflections)
        ])

        self.dims = {
            "input_size": input_size,
            "hidden_size": hidden_size
        }
        self.num_reflections = num_reflections

        self.Wo = nn.Linear(
            in_features=hidden_size,
            out_features=proj_size,
            bias=bias,
            **self.factory_kwargs
        ) if proj_size is not None else None
        self.proj = False if proj_size is None else True

        if funcs is None:
            funcs = ["sigmoid", "sigmoid", "tanh"]
        self.funcs = self._resolve_funcs(funcs, *args, **kwargs)

    def get_ortho_mat(self):
        hidden_size = self.dims["hidden_size"]
        M = torch.eye(hidden_size)

        for v in self.v:
            v = v.view(-1, 1)
            v_sq_norm = torch.sum(v ** 2)
            H = torch.eye(hidden_size) - 2 * torch.mm(v, v.t()) / (v_sq_norm + 1e-8)
            M = torch.mm(H, M)
        return M

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None):
        if h_prev is None:
            h_prev = torch.zeros((self.dims["hidden_size"], 1), **self.factory_kwargs)
        M = self.get_ortho_mat()
        sig_r, sig_z, phi_h = self.funcs

        r_t = sig_r(self.Wr(x) + self.Ur(h_prev) + self.br)
        z_t = sig_z(self.Wz(x) + self.Uz(h_prev) + self.bz)
        h_bar_t = phi_h(self.Wh(x) + ((r_t * h_prev) @ M.t()) + self.bh)
        h_t = (z_t * h_prev) + ((1 - z_t) * h_bar_t)

        if self.proj:
            h_t = self.Wo(h_t)

        return h_t


class JANETCell(DLModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 funcs: Union[List[Tuple[str, Callable, nn.Module]],
                 Tuple[Union[str, Callable, nn.Module]],
                 Dict[str, Union[str, Callable, nn.Module]]] = None,
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
        lin_kwargs = {
            "bias": bias,
            **self.factory_kwargs
        }
        self.Wf = nn.Linear(
            in_features=input_size + hidden_size,
            out_features=hidden_size,
            **lin_kwargs
        )
        self.Wh = nn.Linear(
            in_features=input_size + hidden_size,
            out_features=hidden_size,
            **lin_kwargs
        )
        if funcs is None:
            funcs = ["sigmoid", "tanh"]
        funcs = funcs[:2]
        self.funcs = self._resolve_funcs(funcs, *args, **kwargs)
        self.Wo = nn.Linear(
            in_features=hidden_size,
            out_features=proj_size,
            bias=bias,
            **self.factory_kwargs
        ) if proj_size is not None else None
        self.proj_size = proj_size
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None):
        if h_prev is None:
            h_prev = torch.zeros((x.size(0), self.hidden_size), **self.factory_kwargs)
        x_comb = torch.cat([x, h_prev], dim=-1)
        sig, tanh = self.funcs
        f_t = sig(self.Wf(x_comb))
        c_bar = tanh(self.Wh(x_comb))
        h_t = (f_t * h_prev) + ((1 - f_t) * c_bar)
        return h_t
