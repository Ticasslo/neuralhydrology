"""Temporal cores of the ``SequentialForecast`` model.

Every core maps an embedded sequence of shape ``[batch size, sequence length, input size]`` to an output sequence of
shape ``[batch size, sequence length, core.output_size]``; output step k only depends on input steps <= k (causal).
``forward(x, n_hindcast)`` gets the number of hindcast steps at the start of the sequence; only the RNN cores use it.
The core is selected with the config argument ``sf_core``: 'lstm', 'gru', 'transformer', 's4d' or 'mamba'.
"""
import copy
import math
from typing import Callable

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba as MambaSSM
except ModuleNotFoundError:
    MambaSSM = None

from neuralhydrology.modelzoo.positional_encoding import PositionalEncoding
from neuralhydrology.utils.config import Config

SF_CORES = ['lstm', 'gru', 'transformer', 's4d', 'mamba']


def get_core(cfg: Config, input_size: int) -> nn.Module:
    """Get the temporal core specified by ``sf_core`` in the run configuration.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    input_size : int
        Number of features per time step of the embedded input sequence.

    Returns
    -------
    nn.Module
        Core with attribute ``output_size``.
    """
    core = cfg.sf_core.lower()
    if core in ['lstm', 'gru']:
        return RNNCore(cfg, input_size, cell=core)
    if core == 'transformer':
        return TransformerCore(cfg, input_size)
    if core == 's4d':
        return ResidualCore(cfg, input_size, lambda d: S4DBlock(d_model=d,
                                                                 d_state=cfg.sf_s4d_d_state,
                                                                 dt_min=cfg.sf_s4d_dt_min,
                                                                 dt_max=cfg.sf_s4d_dt_max,
                                                                 lr=cfg.sf_s4d_lr,
                                                                 dropout=cfg.sf_dropout))
    if core == 'mamba':
        if MambaSSM is None:
            raise ModuleNotFoundError('sf_core mamba requires mamba_ssm (pip install mamba_ssm causal-conv1d).')
        return ResidualCore(cfg, input_size, lambda d: MambaSSM(d_model=d,
                                                                 d_state=cfg.sf_mamba_d_state,
                                                                 d_conv=cfg.sf_mamba_d_conv,
                                                                 expand=cfg.sf_mamba_expand))
    raise ValueError(f'Unknown sf_core {cfg.sf_core}. Choose one of {SF_CORES}.')


class RNNCore(nn.Module):
    """LSTM or GRU core (``torch.nn``) with ``sf_layers`` stacked layers of size ``hidden_size``.

    The LSTM uses the forget gate bias ``initial_forget_bias`` in every layer, as the single-layer LSTMs of the library.
    As in ``SequentialForecastLSTM``, the RNN runs over the hindcast steps and then over the forecast steps, starting
    from the final hindcast state, with the sequence dimension first. This gives the same outputs as one run over the
    whole sequence and also the same gradients bit for bit, so a single-layer LSTM core trains exactly like
    ``SequentialForecastLSTM`` (one run gives gradients that differ in the order of the floating point sums).

    Parameters
    ----------
    cfg : Config
        The run configuration.
    input_size : int
        Number of input features per time step.
    cell : str
        'lstm' or 'gru'.
    """

    def __init__(self, cfg: Config, input_size: int, cell: str):
        super(RNNCore, self).__init__()
        rnn_class = nn.LSTM if cell == 'lstm' else nn.GRU
        self.rnn = rnn_class(input_size=input_size, hidden_size=cfg.hidden_size, num_layers=cfg.sf_layers)
        self.output_size = cfg.hidden_size
        if cell == 'lstm' and cfg.initial_forget_bias is not None:
            for layer in range(cfg.sf_layers):
                bias = getattr(self.rnn, f'bias_hh_l{layer}')
                bias.data[cfg.hidden_size:2 * cfg.hidden_size] = cfg.initial_forget_bias

    def forward(self, x: torch.Tensor, n_hindcast: int = None) -> torch.Tensor:
        x = x.transpose(0, 1)
        if n_hindcast is None:
            output, _ = self.rnn(x)
            return output.transpose(0, 1).contiguous()
        output_hindcast, state = self.rnn(x[:n_hindcast])
        output_forecast, _ = self.rnn(x[n_hindcast:], state)
        # concatenated as in SequentialForecastLSTM: a contiguous [batch size, sequence length, hidden size] tensor, so
        # that the output dropout of the model draws the same mask
        return torch.cat([output_hindcast.transpose(0, 1), output_forecast.transpose(0, 1)], dim=1)


class TransformerCore(nn.Module):
    """Causal pre-LayerNorm Transformer encoder core.

    Linear projection to ``hidden_size``, sinusoidal positional encoding of the library (added), ``sf_layers`` encoder
    layers (``norm_first=True``) with a causal attention mask and a final LayerNorm. The feed-forward size is
    ``sf_transformer_dim_feedforward`` (default ``4 * hidden_size``, as in Vaswani et al. 2017); it allows a fine match
    of the number of parameters, since ``hidden_size`` must be a multiple of ``sf_transformer_nheads``.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    input_size : int
        Number of input features per time step.
    """

    def __init__(self, cfg: Config, input_size: int):
        super(TransformerCore, self).__init__()
        d_model = cfg.hidden_size
        if d_model % cfg.sf_transformer_nheads != 0:
            raise ValueError('hidden_size must be divisible by sf_transformer_nheads.')
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(embedding_dim=d_model, position_type='sum',
                                                      dropout=cfg.sf_dropout, max_len=cfg.seq_length)
        dim_feedforward = cfg.sf_transformer_dim_feedforward or 4 * d_model
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=cfg.sf_transformer_nheads,
                                           dim_feedforward=dim_feedforward,
                                           dropout=cfg.sf_dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.sf_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(d_model)
        self.output_size = d_model

    def forward(self, x: torch.Tensor, n_hindcast: int = None) -> torch.Tensor:
        x = self.input_projection(x)
        # the positional encoding of the library expects [sequence length, batch size, features]
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device, dtype=x.dtype)
        x = self.encoder(x, mask=mask, is_causal=True)
        return self.final_norm(x)


class ResidualCore(nn.Module):
    """Pre-LayerNorm residual stack of sequence blocks (S4D, Mamba).

    Linear projection to ``hidden_size``, then ``sf_layers`` times ``x = x + Dropout(block(LayerNorm(x)))``, then a
    final LayerNorm. Each block maps ``[batch size, sequence length, hidden_size]`` to the same shape.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    input_size : int
        Number of input features per time step.
    make_block : Callable[[int], nn.Module]
        Function that returns one block for a given model dimension.
    """

    def __init__(self, cfg: Config, input_size: int, make_block: Callable[[int], nn.Module]):
        super(ResidualCore, self).__init__()
        d_model = cfg.hidden_size
        self.input_projection = nn.Linear(input_size, d_model)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(cfg.sf_layers)])
        self.blocks = nn.ModuleList([make_block(d_model) for _ in range(cfg.sf_layers)])
        self.dropout = nn.Dropout(p=cfg.sf_dropout)
        self.final_norm = nn.LayerNorm(d_model)
        self.output_size = d_model

    def forward(self, x: torch.Tensor, n_hindcast: int = None) -> torch.Tensor:
        x = self.input_projection(x)
        for norm, block in zip(self.norms, self.blocks):
            x = x + self.dropout(block(norm(x)))
        return self.final_norm(x)


class S4DBlock(nn.Module):
    """S4D layer (diagonal state space model) adapted from the minimal reference implementation of the S4 authors.

    Source: https://github.com/state-spaces/s4, ``models/s4/s4d.py`` (Apache License 2.0), Gu et al., "On the
    Parameterization and Initialization of Diagonal State Space Models", NeurIPS 2022. Changes: input and output are
    ``[batch size, sequence length, d_model]``; ``einops`` and ``DropoutNd`` replaced by ``torch`` and ``nn.Dropout``;
    the structure, initialization and parameter registration are unchanged. The SSM parameters ``log_dt``,
    ``log_A_real`` and ``A_imag`` carry an ``_optim`` attribute with their own learning rate (``lr``, None = global
    learning rate) and no weight decay, which ``get_optimizer`` turns into a separate parameter group.

    The convolution with the SSM kernel is computed with an FFT of length 2L and cut to the first L steps, so it is
    causal.

    Parameters
    ----------
    d_model : int
        Number of channels.
    d_state : int
        State size N (N // 2 complex conjugate pairs).
    dt_min, dt_max : float
        Range of the initial step sizes.
    lr : float, optional
        Learning rate of the SSM parameters (0 = fixed buffers, None = global learning rate).
    dropout : float, optional
        Dropout after the activation.
    """

    def __init__(self, d_model: int, d_state: int = 64, dt_min: float = 0.001, dt_max: float = 0.1,
                 lr: float = None, dropout: float = 0.0):
        super(S4DBlock, self).__init__()
        self.h = d_model
        self.n = d_state
        self.D = nn.Parameter(torch.randn(self.h))

        # SSM kernel parameters
        log_dt = torch.rand(self.h) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        C = torch.randn(self.h, self.n // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self._register('log_dt', log_dt, lr)
        self._register('log_A_real', torch.log(0.5 * torch.ones(self.h, self.n // 2)), lr)
        self._register('A_imag', math.pi * torch.arange(self.n // 2).repeat(self.h, 1).float(), lr)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(nn.Conv1d(self.h, 2 * self.h, kernel_size=1), nn.GLU(dim=-2))

    def _register(self, name: str, tensor: torch.Tensor, lr: float = None):
        # register a tensor with a configurable learning rate and 0 weight decay (as in the reference implementation)
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {'weight_decay': 0.0}
            if lr is not None:
                optim['lr'] = lr
            setattr(getattr(self, name), '_optim', optim)

    def kernel(self, length: int) -> torch.Tensor:
        # SSM convolution kernel of shape [d_model, length] (Vandermonde product)
        dt = torch.exp(self.log_dt)
        C = torch.view_as_complex(self.C)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        dtA = A * dt.unsqueeze(-1)
        K = dtA.unsqueeze(-1) * torch.arange(length, device=A.device)
        C = C * (torch.exp(dtA) - 1.) / A
        return 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        u = u.transpose(-1, -2)  # [batch size, d_model, length]
        length = u.size(-1)
        k_f = torch.fft.rfft(self.kernel(length), n=2 * length)
        u_f = torch.fft.rfft(u, n=2 * length)
        y = torch.fft.irfft(u_f * k_f, n=2 * length)[..., :length]
        # D term of the state space equation (skip connection)
        y = y + u * self.D.unsqueeze(-1)
        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        return y.transpose(-1, -2)


def count_parameters(module: nn.Module) -> int:
    """Number of trainable parameters of a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def mamba_block_parameters(d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> int:
    """Number of parameters of one ``mamba_ssm.Mamba`` block (default ``dt_rank``, ``bias=False``, ``conv_bias=True``).

    Follows the layer definitions in ``mamba_ssm/modules/mamba_simple.py`` (in_proj, conv1d, x_proj, dt_proj, A_log, D,
    out_proj), so the size can be computed on machines without the CUDA package; check against the real module on GPU.
    """
    d_inner = expand * d_model
    dt_rank = math.ceil(d_model / 16)
    return (d_model * 2 * d_inner  # in_proj
            + d_inner * d_conv + d_inner  # conv1d (depthwise, with bias)
            + d_inner * (dt_rank + 2 * d_state)  # x_proj
            + dt_rank * d_inner + d_inner  # dt_proj
            + d_inner * d_state  # A_log
            + d_inner  # D
            + d_inner * d_model)  # out_proj


def count_model_parameters(cfg: Config, update: dict = None) -> int:
    """Number of trainable parameters of ``SequentialForecast`` for the run configuration, optionally updated.

    The model is built on the CPU. For ``sf_core: mamba`` without ``mamba_ssm`` installed, the residual stack is built
    without blocks and each layer adds ``mamba_block_parameters`` plus its LayerNorm.

    Parameters
    ----------
    cfg : Config
        The run configuration (not changed).
    update : dict, optional
        Config arguments to change before building the model, e.g. ``{'hidden_size': 64}``.
    """
    # imported here because sequential_forecast imports this module
    from neuralhydrology.modelzoo.sequential_forecast import SequentialForecast

    cfg = copy.deepcopy(cfg)
    cfg.update_config({**(update or {}), 'device': 'cpu'})
    n_mamba_layers = 0
    if cfg.sf_core.lower() == 'mamba' and MambaSSM is None:
        n_mamba_layers = cfg.sf_layers
        cfg.update_config({'sf_core': 's4d', 'sf_layers': 0})
    n_params = count_parameters(SequentialForecast(cfg))
    if n_mamba_layers:
        d = cfg.hidden_size
        block = mamba_block_parameters(d, cfg.sf_mamba_d_state, cfg.sf_mamba_d_conv, cfg.sf_mamba_expand)
        n_params += n_mamba_layers * (block + 2 * d)
    return n_params


def _closest(values: list, key: Callable[[int], dict], cfg: Config, target: int) -> int:
    # value whose model is closest to target parameters; the number of parameters grows with the value (binary search)
    low, high = 0, len(values) - 1
    while low < high:
        mid = (low + high) // 2
        if count_model_parameters(cfg, key(values[mid])) < target:
            low = mid + 1
        else:
            high = mid
    return min(values[max(low - 1, 0):low + 1], key=lambda v: abs(count_model_parameters(cfg, key(v)) - target))


def match_parameters(cfg: Config, target: int, max_hidden_size: int = 1024, tolerance: float = 0.01) -> dict:
    """Find the config arguments that give ``SequentialForecast`` the number of parameters closest to ``target``.

    Searches ``hidden_size`` (for the Transformer core only multiples of ``sf_transformer_nheads``, with the default
    feed-forward size). For the Transformer core, ``sf_transformer_dim_feedforward`` is then searched for the chosen
    ``hidden_size``, because steps of ``sf_transformer_nheads`` in ``hidden_size`` change the size by several percent.
    All other ``sf_*`` arguments are kept.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    target : int
        Target number of trainable parameters, e.g. that of the LSTM reference model.
    max_hidden_size : int, optional
        Largest candidate ``hidden_size``.
    tolerance : float, optional
        Largest accepted relative difference to ``target``.

    Returns
    -------
    dict
        Config arguments to update, ``hidden_size`` and for the Transformer core ``sf_transformer_dim_feedforward``.

    Raises
    ------
    ValueError
        If the closest model differs from ``target`` by more than ``tolerance``.
    """
    transformer = cfg.sf_core.lower() == 'transformer'
    step = cfg.sf_transformer_nheads if transformer else 1
    sizes = list(range(step, max_hidden_size + 1, step))
    hidden_size = _closest(sizes, lambda h: {'hidden_size': h, 'sf_transformer_dim_feedforward': None}, cfg, target)
    update = {'hidden_size': hidden_size}
    if transformer:
        update['sf_transformer_dim_feedforward'] = _closest(
            list(range(1, 16 * hidden_size + 1)),
            lambda f: {'hidden_size': hidden_size, 'sf_transformer_dim_feedforward': f}, cfg, target)
    difference = abs(count_model_parameters(cfg, update) - target) / target
    if difference > tolerance:
        raise ValueError(f'No hidden_size of sf_core {cfg.sf_core} matches {target} parameters within {tolerance:.1%} '
                         f'(closest: {update}, difference {difference:.2%}).')
    return update
