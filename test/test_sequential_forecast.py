"""Tests of the SequentialForecast model with exchangeable cores (sf_cores.py) and of the optimizer parameter groups"""
import itertools
import math
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest
import torch

from neuralhydrology.evaluation.evaluate import start_evaluation
from neuralhydrology.modelzoo import get_model
from neuralhydrology.modelzoo.sequential_forecast import SequentialForecast
from neuralhydrology.modelzoo.sequential_forecast_lstm import SequentialForecastLSTM
from neuralhydrology.modelzoo.sf_cores import count_model_parameters, count_parameters, get_core, match_parameters
from neuralhydrology.training import get_lr_scheduler, get_optimizer
from neuralhydrology.training.train import start_training
from neuralhydrology.utils.config import Config
from test import Fixture

HINDCAST = ['prcp(mm/day)', 'tmax(C)']
FORECAST = ['prcp(mm/day)']
CPU_CORES = ['lstm', 'gru', 'transformer', 's4d']
# trainable parameters of SequentialForecastLSTM in the B1b configuration (33 hindcast, 5 forecast, 33 static inputs)
B1B_PARAMETERS = 84801


def _config(get_config: Callable[[str], Config], tmp_path: Path, **extra) -> Config:
    # daily_regression on the CAMELS-US test basins with different hindcast and forecast inputs; fc embeddings of 16 for
    # dynamic and static inputs give both phases 32 features, as in the BiasCast configurations
    config = get_config('daily_regression')
    update = {'dataset': 'camels_us', 'data_dir': config.data_dir / 'camels_us', 'forcings': 'daymet',
              'dynamic_inputs': HINDCAST, 'hindcast_inputs': HINDCAST, 'forecast_inputs': FORECAST,
              'seq_length': config.seq_length, 'forecast_seq_length': 2, 'model': 'sequential_forecast',
              'sf_core': 'lstm', 'dynamics_embedding': {'type': 'fc', 'hiddens': [16], 'activation': 'linear'},
              'statics_embedding': {'type': 'fc', 'hiddens': [16], 'activation': 'linear'}, 'log_n_figures': 0,
              'log_tensorboard': False}
    update.update(extra)
    config.update_config(update)
    config.run_dir = tmp_path
    return config


def _b1b_config(get_config: Callable[[str], Config], tmp_path: Path, **extra) -> Config:
    # model part of the B1b configuration; dummy names, the model only uses the numbers of inputs
    return _config(get_config, tmp_path, hindcast_inputs=[f'h{i}' for i in range(33)],
                   forecast_inputs=[f'f{i}' for i in range(5)], static_attributes=[f's{i}' for i in range(33)],
                   hidden_size=128, seq_length=365, forecast_seq_length=1, **extra)


def _random_data(config: Config, batch_size: int = 3, seed: int = 0) -> dict:
    generator = torch.Generator().manual_seed(seed)
    length = config.seq_length - config.forecast_seq_length

    def phase(names: list, steps: int) -> dict:
        return {name: torch.randn(batch_size, steps, 1, generator=generator) for name in names}

    return {'x_d_hindcast': phase(config.hindcast_inputs, length),
            'x_d_forecast': phase(config.forecast_inputs, config.forecast_seq_length),
            'x_s': torch.randn(batch_size, len(config.static_attributes), generator=generator)}


def test_lstm_core_equals_sequential_forecast_lstm(get_config: Fixture[Callable[[str], Config]], tmp_path: Path):
    config = _config(get_config, tmp_path)
    reference = SequentialForecastLSTM(config).eval()
    model = SequentialForecast(config).eval()

    # same parameter names up to the prefix of the LSTM; random weights instead of the initialization
    state = {name: torch.randn_like(value) for name, value in reference.state_dict().items()}
    model.load_state_dict({name.replace('lstm.', 'core.rnn.', 1): value for name, value in state.items()})
    reference.load_state_dict(state)

    data = _random_data(config)
    with torch.no_grad():
        y_reference = reference(data)['y_hat']
        y_model = model(data)['y_hat']
    assert y_model.shape == y_reference.shape == (3, config.seq_length, 1)
    assert torch.equal(y_model, y_reference)


def test_lstm_core_trains_like_sequential_forecast_lstm(get_config: Fixture[Callable[[str], Config]], tmp_path: Path):
    # same initialization, batches, dropout masks and optimizer steps => same weights bit for bit
    config = _config(get_config, tmp_path, initial_forget_bias=3, output_dropout=0.4)
    models = []
    for model_class in (SequentialForecastLSTM, SequentialForecast):
        torch.manual_seed(111)
        model = model_class(config)
        optimizer = get_optimizer(model, config)
        generator = torch.Generator().manual_seed(0)
        for step in range(20):
            data = _random_data(config, batch_size=8, seed=step)
            target = torch.randn(8, generator=generator)
            optimizer.zero_grad()
            ((model(data)['y_hat'][:, -1, 0] - target)**2).mean().backward()
            optimizer.step()
        models.append(model.state_dict())
    reference, model = models
    assert all(torch.equal(value, model[name.replace('lstm.', 'core.rnn.', 1)]) for name, value in reference.items())


def test_lstm_core_same_initialization(get_config: Fixture[Callable[[str], Config]], tmp_path: Path):
    # submodules are created in the same order, so the same seed gives the same initial weights
    config = _config(get_config, tmp_path, initial_forget_bias=3)
    torch.manual_seed(111)
    reference = SequentialForecastLSTM(config).state_dict()
    torch.manual_seed(111)
    model = SequentialForecast(config).state_dict()
    assert [name.replace('lstm.', 'core.rnn.', 1) for name in reference] == list(model)
    assert all(torch.equal(value, model[name.replace('lstm.', 'core.rnn.', 1)]) for name, value in reference.items())


@pytest.mark.parametrize('core', CPU_CORES)
@pytest.mark.parametrize('step', [0, 17, 29])
def test_core_is_causal(get_config: Fixture[Callable[[str], Config]], tmp_path: Path, core: str, step: int):
    config = _config(get_config, tmp_path, sf_core=core, sf_layers=2, hidden_size=16)
    module = get_core(config, input_size=32).eval()
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(2, config.seq_length, 32, generator=generator)
    x_changed = x.clone()
    x_changed[:, step] += 10 * torch.randn(2, 32, generator=generator)
    with torch.no_grad():
        out, out_changed = module(x), module(x_changed)
    assert out.shape == (2, config.seq_length, module.output_size)
    # outputs before the changed step stay the same, the output at the changed step differs
    if step > 0:
        assert (out[:, :step] - out_changed[:, :step]).abs().max() <= 1e-6
    assert (out[:, step] - out_changed[:, step]).abs().max() > 1e-3


@pytest.mark.parametrize('core', CPU_CORES)
def test_model_output_shape(get_config: Fixture[Callable[[str], Config]], tmp_path: Path, core: str):
    config = _config(get_config, tmp_path, sf_core=core, hidden_size=16)
    model = get_model(config).eval()
    with torch.no_grad():
        pred = model(_random_data(config))
    assert pred['y_hat'].shape == (3, config.seq_length, 1)
    assert pred['core_output'].shape == (3, config.seq_length, 16)
    assert torch.isfinite(pred['y_hat']).all()


def test_b1b_lstm_parameters(get_config: Fixture[Callable[[str], Config]], tmp_path: Path):
    config = _b1b_config(get_config, tmp_path)
    assert count_parameters(SequentialForecastLSTM(config)) == B1B_PARAMETERS
    assert count_model_parameters(config) == B1B_PARAMETERS
    assert count_model_parameters(config, {'hidden_size': 64}) < B1B_PARAMETERS


@pytest.mark.parametrize('core, extra', [('lstm', {}), ('gru', {}), ('transformer', {'sf_layers': 2}),
                                         ('transformer', {'sf_layers': 4}), ('s4d', {'sf_layers': 2}),
                                         ('mamba', {'sf_layers': 2})])
def test_match_parameters(get_config: Fixture[Callable[[str], Config]], tmp_path: Path, core: str, extra: dict):
    config = _b1b_config(get_config, tmp_path, sf_core=core, **extra)
    update = match_parameters(config, B1B_PARAMETERS)
    assert abs(count_model_parameters(config, update) - B1B_PARAMETERS) <= 0.01 * B1B_PARAMETERS
    if core == 'lstm':
        assert update == {'hidden_size': 128}
    if core == 'transformer':
        assert update['hidden_size'] % config.sf_transformer_nheads == 0
        # the feed-forward size gives a much finer match than hidden_size alone
        assert abs(count_model_parameters(config, update) - B1B_PARAMETERS) <= 0.002 * B1B_PARAMETERS


def test_match_parameters_tolerance(get_config: Fixture[Callable[[str], Config]], tmp_path: Path):
    # hidden_size 128 and 129 give 84,801 and 85,966 parameters
    config = _b1b_config(get_config, tmp_path, sf_core='lstm')
    with pytest.raises(ValueError, match='No hidden_size'):
        match_parameters(config, B1B_PARAMETERS + 300, tolerance=0.001)


def test_optimizer_single_group_without_ssm(get_config: Fixture[Callable[[str], Config]], tmp_path: Path):
    config = _config(get_config, tmp_path, sf_core='lstm')
    model = get_model(config)
    optimizer = get_optimizer(model, config)
    assert len(optimizer.param_groups) == 1
    assert [id(p) for p in optimizer.param_groups[0]['params']] == [id(p) for p in model.parameters()]


@pytest.mark.parametrize('optimizer_name', ['Adam', 'AdamW'])
def test_optimizer_ssm_group(get_config: Fixture[Callable[[str], Config]], tmp_path: Path, optimizer_name: str):
    config = _config(get_config, tmp_path, sf_core='s4d', sf_layers=2, hidden_size=16, sf_s4d_lr=1e-4,
                     optimizer=optimizer_name, lr_scheduler={'type': 'CosineAnnealingLR', 'T_max': 30})
    model = get_model(config)
    optimizer = get_optimizer(model, config)
    assert len(optimizer.param_groups) == 2
    main_group, ssm_group = optimizer.param_groups
    # 3 SSM parameters (log_dt, log_A_real, A_imag) per layer
    assert len(ssm_group['params']) == 6
    assert ssm_group['lr'] == 1e-4 and ssm_group['weight_decay'] == 0.0
    assert main_group['lr'] == config.learning_rate
    # every parameter in exactly one group
    grouped = [id(p) for p in itertools.chain(main_group['params'], ssm_group['params'])]
    assert sorted(grouped) == sorted(id(p) for p in model.parameters())

    scheduler = get_lr_scheduler(optimizer, config)
    optimizer.step()
    scheduler.step()
    factor = (1 + math.cos(math.pi / 30)) / 2
    assert main_group['lr'] == pytest.approx(config.learning_rate * factor)
    assert ssm_group['lr'] == pytest.approx(1e-4 * factor)


def test_s4d_fixed_ssm_parameters(get_config: Fixture[Callable[[str], Config]], tmp_path: Path):
    # sf_s4d_lr 0 registers the SSM parameters as buffers
    config = _config(get_config, tmp_path, sf_core='s4d', hidden_size=16, sf_s4d_lr=0.0)
    model = get_model(config)
    assert not any(hasattr(p, '_optim') for p in model.parameters())
    assert 'core.blocks.0.log_dt' in model.state_dict()
    assert len(get_optimizer(model, config).param_groups) == 1


def test_config_keys(get_config: Fixture[Callable[[str], Config]], tmp_path: Path):
    config = _config(get_config, tmp_path, sf_core='mamba', sf_mamba_d_state=32, sf_mamba_d_conv=3,
                     sf_mamba_expand=1)
    assert (config.sf_mamba_d_state, config.sf_mamba_d_conv, config.sf_mamba_expand) == (32, 3, 1)
    assert config.sf_layers == 1 and config.sf_dropout == 0.0 and config.sf_s4d_lr is None

    config = _config(get_config, tmp_path)
    del config.as_dict()['sf_core']
    with pytest.raises(ValueError, match='sf_core'):
        SequentialForecast(config)

    config = _config(get_config, tmp_path, sf_core='rnn')
    with pytest.raises(ValueError, match='Unknown sf_core'):
        SequentialForecast(config)


@pytest.mark.parametrize('core', CPU_CORES)
@pytest.mark.parametrize('data_on_gpu', [False, True])
def test_training_and_evaluation_run(get_config: Fixture[Callable[[str], Config]], tmp_path: Path, core: str,
                                     data_on_gpu: bool):
    # one epoch with validation, best model and the test period; data_on_gpu uses the device of the config (cpu)
    config = _config(get_config, tmp_path, sf_core=core, sf_layers=2, hidden_size=16, sf_s4d_lr=1e-4,
                     lr_scheduler={'type': 'CosineAnnealingLR', 'T_max': 1}, loss='NSE', early_stopping_patience=1,
                     data_on_gpu=data_on_gpu)
    start_training(config)
    weights = torch.load(config.run_dir / 'model_epoch001.pt', weights_only=True)
    assert all(torch.isfinite(value).all() for value in weights.values())
    assert (config.run_dir / 'best_model.pt').is_file()
    start_evaluation(cfg=config, run_dir=config.run_dir, epoch=1, period='test')
    metrics = pd.read_csv(config.run_dir / 'test' / 'model_epoch001' / 'test_metrics.csv', dtype={'basin': str})
    assert len(metrics) == 4 and metrics['NSE'].notna().all()
