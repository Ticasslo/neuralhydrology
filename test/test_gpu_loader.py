"""Test that GpuBatchLoader (data_on_gpu) yields the same batches, in the same order, as the default DataLoader"""
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datautils.gpu_loader import GpuBatchLoader
from neuralhydrology.evaluation.evaluate import start_evaluation
from neuralhydrology.training.train import start_training
from neuralhydrology.utils.config import Config
from test import Fixture

FORCINGS = ['prcp(mm/day)', 'tmax(C)']


def _config(get_config: Callable[[str], Config], tmp_path: Path, forecast: bool, **extra) -> Config:
    # daily_regression on the CAMELS-US test basins; forecast=True feeds both forcings to the hindcast and forecast phases
    # (the sequential forecast LSTM needs embeddings of equal size)
    config = get_config('daily_regression')
    update = {'dataset': 'camels_us', 'data_dir': config.data_dir / 'camels_us', 'forcings': 'daymet',
              'dynamic_inputs': FORCINGS, 'log_n_figures': 0, 'log_tensorboard': False}
    if forecast:
        update.update({'model': 'sequential_forecast_lstm', 'hindcast_inputs': FORCINGS,
                       'forecast_inputs': FORCINGS, 'forecast_seq_length': 2})
    update.update(extra)
    config.update_config(update)
    config.run_dir = tmp_path
    config.train_dir = tmp_path / 'train_data'
    config.train_dir.mkdir(parents=True, exist_ok=True)
    return config


def _assert_same_batch(new: dict, old: dict, with_dates: bool):
    # same keys, shapes and values (NaN equal to NaN); training batches of the new loader carry no date
    old_keys = {k for k in old if with_dates or not k.startswith('date')}
    assert set(new) == old_keys
    for key in old_keys:
        if key.startswith('date'):
            assert np.array_equal(new[key], old[key])
        elif isinstance(old[key], dict):
            assert set(new[key]) == set(old[key])
            for k in old[key]:
                assert torch.equal(new[key][k].nan_to_num(-9e9), old[key][k].nan_to_num(-9e9))
        else:
            assert torch.equal(new[key].nan_to_num(-9e9), old[key].nan_to_num(-9e9))


@pytest.mark.parametrize('forecast', [False, True])
@pytest.mark.parametrize('num_workers', [0, 2])
def test_training_batches(get_config: Fixture[Callable[[str], Config]], tmp_path: Path, forecast: bool,
                          num_workers: int):
    config = _config(get_config, tmp_path, forecast)
    ds = get_dataset(cfg=config, is_train=True, period='train', scaler={})
    for seed in (0, 1):
        torch.manual_seed(seed)
        old = list(DataLoader(ds, batch_size=config.batch_size, shuffle=True, num_workers=num_workers,
                              collate_fn=ds.collate_fn))
        torch.manual_seed(seed)
        loader = GpuBatchLoader(ds, batch_size=config.batch_size, device='cpu', shuffle=True)
        new = list(loader)
        assert len(new) == len(old) == len(loader) == math.ceil(len(ds) / config.batch_size)
        for b_new, b_old in zip(new, old):
            _assert_same_batch(b_new, b_old, with_dates=False)
        # the replaced loader and the new one leave the global random number generator in the same state
        state_new = torch.get_rng_state()
        torch.manual_seed(seed)
        list(DataLoader(ds, batch_size=config.batch_size, shuffle=True, num_workers=num_workers,
                        collate_fn=ds.collate_fn))
        assert torch.equal(state_new, torch.get_rng_state())


@pytest.mark.parametrize('forecast', [False, True])
def test_evaluation_batches(get_config: Fixture[Callable[[str], Config]], tmp_path: Path, forecast: bool):
    config = _config(get_config, tmp_path, forecast)
    train_ds = get_dataset(cfg=config, is_train=True, period='train', scaler={})
    basin = Path(config.validation_basin_file).read_text().split()[0]
    ds = get_dataset(cfg=config, is_train=False, period='validation', basin=basin, scaler=train_ds.scaler)
    old = list(DataLoader(ds, batch_size=config.batch_size, num_workers=0, collate_fn=ds.collate_fn))
    new = list(GpuBatchLoader(ds, batch_size=config.batch_size, device='cpu', shuffle=False, with_dates=True))
    assert len(new) == len(old)
    for b_new, b_old in zip(new, old):
        _assert_same_batch(b_new, b_old, with_dates=True)


def test_timestep_counter(get_config: Fixture[Callable[[str], Config]], tmp_path: Path):
    config = _config(get_config, tmp_path, forecast=True, timestep_counter=True)
    ds = get_dataset(cfg=config, is_train=True, period='train', scaler={})
    torch.manual_seed(0)
    old = list(DataLoader(ds, batch_size=config.batch_size, shuffle=True, collate_fn=ds.collate_fn))
    torch.manual_seed(0)
    new = list(GpuBatchLoader(ds, batch_size=config.batch_size, device='cpu', shuffle=True))
    assert 'hindcast_counter' in new[0]['x_d'] and 'forecast_counter' in new[0]['x_d']
    for b_new, b_old in zip(new, old):
        _assert_same_batch(b_new, b_old, with_dates=False)


def test_nan_streaks(get_config: Fixture[Callable[[str], Config]], tmp_path: Path):
    # one feature per group; sequences and steps are dropped with the configured probabilities, never all groups
    p_seq, p_step = 0.3, 0.2
    config = _config(get_config, tmp_path, forecast=True, hindcast_inputs=[[FORCINGS[0]], [FORCINGS[1]]],
                     forecast_inputs=[[FORCINGS[0]]], nan_sequence_probability=p_seq, nan_step_probability=p_step)
    ds = get_dataset(cfg=config, is_train=True, period='train', scaler={})
    torch.manual_seed(0)
    hindcast = torch.cat([torch.cat([b['x_d_hindcast'][k] for k in FORCINGS], dim=-1)
                          for b in GpuBatchLoader(ds, batch_size=config.batch_size, device='cpu', shuffle=True)])
    nan = hindcast.isnan()                       # [samples, steps, groups]; the test forcings contain no NaN
    n_samples, n_groups = nan.shape[0], nan.shape[2]
    seq_dropped = nan.all(dim=1)                 # a fully dropped sequence (all steps dropped by chance is negligible)
    assert not seq_dropped.all(dim=1).any()
    # P(group dropped) = p_seq - p_seq**2 / 2 for two groups (when both are drawn, one of them is restored)
    expected_seq = p_seq - p_seq**2 / n_groups
    rate_seq = seq_dropped.float().mean().item()
    assert abs(rate_seq - expected_seq) < 3 * math.sqrt(expected_seq * (1 - expected_seq) / (n_samples * n_groups))
    steps_kept = nan.permute(0, 2, 1)[~seq_dropped]
    rate_step = steps_kept.float().mean().item()
    assert abs(rate_step - p_step) < 3 * math.sqrt(p_step * (1 - p_step) / steps_kept.numel())
    # evaluation batches are never masked
    val = get_dataset(cfg=config, is_train=False, period='validation', basin=ds.basins[0], scaler=ds.scaler)
    for b in GpuBatchLoader(val, batch_size=config.batch_size, device='cpu', shuffle=False, with_dates=True):
        assert not any(v.isnan().any() for v in b['x_d_hindcast'].values())


def test_unsupported(get_config: Fixture[Callable[[str], Config]], tmp_path: Path):
    config = _config(get_config, tmp_path, forecast=False, use_basin_id_encoding=True)
    ds = get_dataset(cfg=config, is_train=True, period='train', scaler={})
    with pytest.raises(NotImplementedError, match='use_basin_id_encoding'):
        GpuBatchLoader(ds, batch_size=config.batch_size, device='cpu', shuffle=True)


@pytest.mark.parametrize('forecast', [False, True])
@pytest.mark.parametrize('num_workers', [0, 2])
def test_training_and_evaluation_identical(get_config: Fixture[Callable[[str], Config]], tmp_path: Path, forecast: bool,
                                           num_workers: int):
    # two epochs of training + validation, then test evaluation, with and without data_on_gpu: bit-identical results
    out = {}
    for data_on_gpu in (False, True):
        run_dir = tmp_path / f'data_on_gpu_{data_on_gpu}'
        config = _config(get_config, run_dir, forecast, epochs=2, data_on_gpu=data_on_gpu, num_workers=num_workers)
        start_training(config)
        start_evaluation(cfg=config, run_dir=config.run_dir, epoch=2, period='test')
        weights = torch.load(config.run_dir / 'model_epoch002.pt', weights_only=True)
        metrics = pd.read_csv(config.run_dir / 'test' / 'model_epoch002' / 'test_metrics.csv', dtype={'basin': str})
        out[data_on_gpu] = (weights, metrics)
    (w_old, m_old), (w_new, m_new) = out[False], out[True]
    assert set(w_old) == set(w_new)
    assert all(torch.equal(w_old[k], w_new[k]) for k in w_old)
    pd.testing.assert_frame_equal(m_old, m_new, check_exact=True)
