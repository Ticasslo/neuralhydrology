"""Tests of resume_training_state and max_training_hours: a run continued from its training state must match an
uninterrupted run bit for bit (weights, optimizer, best model, validation metrics), on the CPU"""
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest
import torch

from neuralhydrology.nh_run import continue_run
from neuralhydrology.training.train import start_training
from neuralhydrology.utils.config import Config
from test import Fixture

HINDCAST = ['prcp(mm/day)', 'tmax(C)']
FORECAST = ['prcp(mm/day)']
EPOCHS = 4


def _config(get_config: Callable[[str], Config], run_dir: Path, core: str, epochs: int, **extra) -> Config:
    # sequential_forecast on the CAMELS-US test basins with dropout (random number generators matter), a cosine
    # learning rate schedule and early stopping that never stops, as in the 30-epoch protocol of the project
    config = get_config('daily_regression')
    update = {'dataset': 'camels_us', 'data_dir': config.data_dir / 'camels_us', 'forcings': 'daymet',
              'dynamic_inputs': HINDCAST, 'hindcast_inputs': HINDCAST, 'forecast_inputs': FORECAST,
              'forecast_seq_length': 2, 'model': 'sequential_forecast', 'sf_core': core, 'sf_dropout': 0.1,
              'dynamics_embedding': {'type': 'fc', 'hiddens': [16], 'activation': 'tanh'},
              'statics_embedding': {'type': 'fc', 'hiddens': [16], 'activation': 'tanh'}, 'log_n_figures': 0,
              'log_tensorboard': False, 'learning_rate': 1e-3,
              'lr_scheduler': {'type': 'CosineAnnealingLR', 'T_max': EPOCHS, 'eta_min': 1e-5},
              'early_stopping_patience': EPOCHS, 'early_stopping_min_delta': 0.0, 'early_stopping_metric': 'NSE',
              'early_stopping_mode': 'max', 'save_validation_results': True, 'epochs': epochs}
    update.update(extra)
    config.update_config(update)
    config.run_dir = run_dir
    return config


def _single_run_dir(parent: Path) -> Path:
    runs = [p for p in parent.iterdir() if p.is_dir()]
    assert len(runs) == 1, runs
    return runs[0]


def _load(path: Path) -> dict:
    return torch.load(str(path), map_location='cpu', weights_only=True)


def _equal(a, b) -> bool:
    # nested comparison of state dicts (tensors bit for bit)
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


@pytest.mark.parametrize('core', ['lstm', 'transformer'])
def test_resumed_run_equals_uninterrupted_run(get_config: Fixture[Callable[[str], Config]], tmp_path: Path, core: str):
    start_training(_config(get_config, tmp_path / 'straight', core, EPOCHS))
    straight = _single_run_dir(tmp_path / 'straight')

    start_training(_config(get_config, tmp_path / 'resumed', core, EPOCHS // 2))
    resumed = _single_run_dir(tmp_path / 'resumed')
    original_config = (resumed / 'config.yml').read_text()
    continue_run(resumed, config_file={'resume_training_state': True, 'epochs': EPOCHS})

    # continued inside the run directory, original configuration kept
    assert [p.name for p in resumed.iterdir() if p.is_dir() and p.name.startswith('continue_training')] == []
    assert (resumed / 'config.yml').read_text() == original_config
    assert (resumed / f'config_continue_from_epoch{EPOCHS // 2:03d}.yml').is_file()
    assert f'Training state of epoch {EPOCHS // 2} restored' in (resumed / 'output.log').read_text()

    for epoch in range(1, EPOCHS + 1):
        assert _equal(_load(straight / f'model_epoch{epoch:03d}.pt'), _load(resumed / f'model_epoch{epoch:03d}.pt'))
        straight_metrics, resumed_metrics = [
            pd.read_csv(run / 'validation' / f'model_epoch{epoch:03d}' / 'validation_metrics.csv')
            for run in (straight, resumed)
        ]
        pd.testing.assert_frame_equal(straight_metrics, resumed_metrics, check_exact=True)
    assert not (resumed / f'model_epoch{EPOCHS + 1:03d}.pt').exists()
    assert _equal(_load(straight / f'optimizer_state_epoch{EPOCHS:03d}.pt'),
                  _load(resumed / f'optimizer_state_epoch{EPOCHS:03d}.pt'))
    assert _equal(_load(straight / 'best_model.pt'), _load(resumed / 'best_model.pt'))
    assert _equal(_load(straight / f'training_state_epoch{EPOCHS:03d}.pt'),
                  _load(resumed / f'training_state_epoch{EPOCHS:03d}.pt'))


def test_max_training_hours_pauses_after_epoch(get_config: Fixture[Callable[[str], Config]], tmp_path: Path):
    start_training(_config(get_config, tmp_path, 'lstm', EPOCHS, max_training_hours=1e-9))
    run_dir = _single_run_dir(tmp_path)
    assert 'Training paused at epoch 1' in (run_dir / 'output.log').read_text()
    assert 'stopped early' not in (run_dir / 'output.log').read_text()
    assert all((run_dir / f'{name}_epoch001.pt').is_file() for name in ['model', 'optimizer_state', 'training_state'])
    assert not (run_dir / 'model_epoch002.pt').exists()

    # the paused run continues to the last epoch
    continue_run(run_dir, config_file={'resume_training_state': True, 'max_training_hours': None})
    assert (run_dir / f'model_epoch{EPOCHS:03d}.pt').is_file()
    assert not (run_dir / f'model_epoch{EPOCHS + 1:03d}.pt').exists()


def test_resume_without_training_state_raises(get_config: Fixture[Callable[[str], Config]], tmp_path: Path):
    start_training(_config(get_config, tmp_path, 'lstm', 1))
    run_dir = _single_run_dir(tmp_path)
    for path in run_dir.glob('training_state_epoch*.pt'):
        path.unlink()
    with pytest.raises(FileNotFoundError, match='training_state'):
        continue_run(run_dir, config_file={'resume_training_state': True, 'epochs': 2})


def test_continue_without_resume_keeps_library_behavior(get_config: Fixture[Callable[[str], Config]], tmp_path: Path):
    # without resume_training_state: new subdirectory, 'epochs' more epochs, as before
    start_training(_config(get_config, tmp_path, 'lstm', 1))
    run_dir = _single_run_dir(tmp_path)
    continue_run(run_dir, config_file={'epochs': 1})
    continued = run_dir / 'continue_training_from_epoch001'
    assert (continued / 'model_epoch002.pt').is_file()
    assert (run_dir / 'config.yml').is_file() and (continued / 'config.yml').is_file()
