"""Test high_flow_oversample: weighted draws of high-flow training samples in GpuBatchLoader (data_on_gpu)"""
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datautils.gpu_loader import GpuBatchLoader
from neuralhydrology.training.train import start_training
from neuralhydrology.utils.config import Config
from test import Fixture

FORCINGS = ['prcp(mm/day)', 'tmax(C)']
OPTS = {'quantile': 0.9, 'factor': 5}


def _config(get_config: Callable[[str], Config], tmp_path: Path, **extra) -> Config:
    # daily_regression on the CAMELS-US test basins, as test_gpu_loader.py
    config = get_config('daily_regression')
    update = {'dataset': 'camels_us', 'data_dir': config.data_dir / 'camels_us', 'forcings': 'daymet',
              'dynamic_inputs': FORCINGS, 'log_n_figures': 0, 'log_tensorboard': False}
    update.update(extra)
    config.update_config(update)
    config.run_dir = tmp_path
    config.train_dir = tmp_path / 'train_data'
    config.train_dir.mkdir(parents=True, exist_ok=True)
    return config


def _order(loader: GpuBatchLoader, seed: int) -> torch.Tensor:
    # sample order of one epoch: the row of each drawn sample, read back from the yielded batches is not needed
    torch.manual_seed(seed)
    if loader._weights is None:
        return torch.cat(list(DataLoader(range(loader.n_samples), batch_size=loader.batch_size, shuffle=True)))
    return torch.multinomial(loader._weights, loader.n_samples, replacement=True)


def test_without_key_order_unchanged(get_config: Fixture[Callable[[str], Config]], tmp_path: Path):
    # no key => no weights, the epoch yields the same batches as the uniform shuffle of the unchanged loader
    config = _config(get_config, tmp_path)
    assert config.high_flow_oversample == {}
    ds = get_dataset(cfg=config, is_train=True, period='train', scaler={})
    loader = GpuBatchLoader(ds, batch_size=config.batch_size, device='cpu', shuffle=True)
    assert loader._weights is None
    torch.manual_seed(3)
    first = [b['y'] for b in loader]
    torch.manual_seed(3)
    second = [b['y'] for b in loader]
    assert all(torch.equal(a.nan_to_num(-9e9), b.nan_to_num(-9e9)) for a, b in zip(first, second))


def test_thresholds_and_share(get_config: Fixture[Callable[[str], Config]], tmp_path: Path):
    config = _config(get_config, tmp_path, high_flow_oversample=OPTS)
    ds = get_dataset(cfg=config, is_train=True, period='train', scaler={})
    loader = GpuBatchLoader(ds, batch_size=config.batch_size, device='cpu', shuffle=True)
    target = loader._y[loader._row_of, 0].numpy().astype(np.float64)
    basin = loader._basin_of.numpy()
    # thresholds are the per-basin quantile of the training targets only
    for b, threshold in loader.oversample_thresholds.items():
        values = target[basin == b]
        assert threshold == pytest.approx(np.nanquantile(values, OPTS['quantile']))
        assert np.array_equal(loader.oversample_high[basin == b], values > threshold)
    high = loader.oversample_high
    assert 0.05 < high.mean() < 0.15
    # expected share of high samples per epoch, and the drawn share over several epochs
    expected = OPTS['factor'] * high.sum() / (OPTS['factor'] * high.sum() + (~high).sum())
    drawn = np.concatenate([high[_order(loader, s).numpy()] for s in range(20)])
    assert drawn.mean() == pytest.approx(expected, abs=0.02)
    # the epoch keeps the same number of batches, and the same seed gives the same order
    assert len(loader) == int(np.ceil(len(ds) / config.batch_size))
    assert torch.equal(_order(loader, 7), _order(loader, 7))
    assert not torch.equal(_order(loader, 7), _order(loader, 8))


def test_batches_follow_weighted_order(get_config: Fixture[Callable[[str], Config]], tmp_path: Path):
    # the yielded batches are the samples of the weighted order, in that order
    config = _config(get_config, tmp_path, high_flow_oversample=OPTS)
    ds = get_dataset(cfg=config, is_train=True, period='train', scaler={})
    loader = GpuBatchLoader(ds, batch_size=config.batch_size, device='cpu', shuffle=True)
    order = _order(loader, 5)
    torch.manual_seed(5)
    y_last = torch.cat([b['y'][:, -1, 0] for b in loader])
    expected = loader._y[loader._row_of[order], 0]
    assert torch.equal(y_last.nan_to_num(-9e9), expected.nan_to_num(-9e9))


def test_evaluation_not_weighted(get_config: Fixture[Callable[[str], Config]], tmp_path: Path):
    # validation and test loaders (shuffle False, not is_train) ignore the key
    config = _config(get_config, tmp_path, high_flow_oversample=OPTS)
    ds = get_dataset(cfg=config, is_train=True, period='train', scaler={})
    assert GpuBatchLoader(ds, batch_size=config.batch_size, device='cpu', shuffle=False)._weights is None


@pytest.mark.parametrize('value', [{'quantile': 1.0, 'factor': 5}, {'quantile': 0.9}, {'quantile': 0.9, 'factor': 0}])
def test_bad_options(get_config: Fixture[Callable[[str], Config]], tmp_path: Path, value: dict):
    config = _config(get_config, tmp_path, high_flow_oversample=value)
    with pytest.raises(ValueError):
        config.high_flow_oversample


def test_requires_data_on_gpu(get_config: Fixture[Callable[[str], Config]], tmp_path: Path):
    config = _config(get_config, tmp_path, high_flow_oversample=OPTS, data_on_gpu=False, epochs=1, device='cpu')
    with pytest.raises(NotImplementedError):
        start_training(config)


def test_training_runs(get_config: Fixture[Callable[[str], Config]], tmp_path: Path):
    # a short training with the key finishes and writes the model of each epoch
    config = _config(get_config, tmp_path, high_flow_oversample=OPTS, data_on_gpu=True, epochs=2, device='cpu')
    start_training(config)
    assert len(list(config.run_dir.glob('model_epoch*.pt'))) == 2
