from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.sf_cores import get_core
from neuralhydrology.utils.config import Config


class SequentialForecast(BaseModel):
    """Sequential forecast model with an exchangeable temporal core.

    Same embeddings and head as ``SequentialForecastLSTM``: the hindcast and forecast inputs pass through separate
    embedding networks of equal output size, the two embedded sequences are concatenated along time and passed to the
    core, together with the number of hindcast steps. The head is applied to every time step. With ``sf_core: lstm`` and
    ``sf_layers: 1``, the model equals ``SequentialForecastLSTM`` bit for bit in outputs and gradients (the RNN core
    rolls out over hindcast, then forecast steps, see ``RNNCore``); the parameters have the same names except for the
    prefix ``lstm.`` => ``core.rnn.``.

    The core is chosen with ``sf_core`` ('lstm', 'gru', 'transformer', 's4d', 'mamba'), see ``sf_cores.py``.
    Do not use this model with ``forecast_overlap`` > 0.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    Raises
    ------
    ValueError if forecast_overlap > 0
    ValueError if forecast and hindcast embedding nets have different output sizes.
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['hindcast_embedding_net', 'forecast_embedding_net', 'core', 'head']

    def __init__(self, cfg: Config):
        super(SequentialForecast, self).__init__(cfg=cfg)

        if cfg.forecast_overlap:
            raise ValueError('Forecast overlap cannot be set for a sequential forecasting model. '
                             'Please set to None or remove from config file.')

        self.forecast_embedding_net = InputLayer(cfg, embedding_type='forecast')
        self.hindcast_embedding_net = InputLayer(cfg, embedding_type='hindcast')
        if self.forecast_embedding_net.output_size != self.hindcast_embedding_net.output_size:
            raise ValueError('Forecast and hindcast embedding nets must have the same output size when using a '
                             'sequential forecast model.')

        self.core = get_core(cfg, input_size=self.hindcast_embedding_net.output_size)
        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(cfg=cfg, n_in=self.core.output_size, n_out=self.output_size)

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the SequentialForecast model.

        Parameters
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs as a dictionary.
                - y_hat: Predictions over the whole sequence from the head layer, [batch size, seq, n_targets].
                - core_output: Output sequence of the core, [batch size, seq, core output size].
        """
        # embeddings are [seq, batch, features]; the cores work on [batch, seq, features]
        x_h = self.hindcast_embedding_net(data)
        x_f = self.forecast_embedding_net(data)
        x = torch.cat([x_h, x_f], dim=0).transpose(0, 1)

        core_output = self.core(x, n_hindcast=x_h.shape[0])
        pred = self.head(self.dropout(core_output))
        pred['core_output'] = core_output
        return pred
