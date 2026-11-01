import copy
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from neuralhydrology.datasetzoo.lamah import LamaH
from neuralhydrology.datautils.gpu_loader import GpuBatchLoader
from neuralhydrology.modelzoo.sequential_forecast_lstm import SequentialForecastLSTM
from test.test_b2_source_inputs import b2_config


@pytest.mark.parametrize('p_step,p_sequence', [(0.1, 0.12), (0.05, 0.05)])
@pytest.mark.parametrize('vectorized', [False, True])
def test_b3_group_masking_preserves_storage_targets_and_eval(b2_config, p_step, p_sequence, vectorized):
    cfg = b2_config(nan_step_probability=p_step, nan_sequence_probability=p_sequence)
    ds = LamaH(cfg, is_train=True, period='train', scaler={})
    before = copy.deepcopy(ds._x_d)
    targets = copy.deepcopy(ds._y)
    np.random.seed(43)
    torch.manual_seed(43)
    loader = (GpuBatchLoader(ds, batch_size=8, device='cpu', shuffle=False, with_dates=True)
              if vectorized else DataLoader(ds, batch_size=8, collate_fn=ds.collate_fn))
    batch = next(iter(loader))
    # At each hindcast step, the entire two-variable Q source shares its added mask.
    plain_cfg = b2_config(nan_step_probability=0., nan_sequence_probability=0.)
    plain = LamaH(plain_cfg, is_train=True, period='train', scaler={})
    ref = next(iter(DataLoader(plain, batch_size=8, collate_fn=plain.collate_fn)))
    added = {k: batch['x_d_hindcast'][k].isnan() & ref['x_d_hindcast'][k].isfinite()
             for k in ('qmean_mmd', 'qmax_mmd')}
    valid = ref['x_d_hindcast']['qmean_mmd'].isfinite() & ref['x_d_hindcast']['qmax_mmd'].isfinite()
    assert torch.equal(added['qmean_mmd'][valid], added['qmax_mmd'][valid])
    torch.testing.assert_close(batch['y'], ref['y'], rtol=0, atol=0, equal_nan=True)
    model = SequentialForecastLSTM(cfg)
    pred = model(batch)['y_hat']
    assert pred.isfinite().all()
    pred.square().mean().backward()
    assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
    for basin in before:
        for freq in before[basin]:
            for key in before[basin][freq]:
                torch.testing.assert_close(ds._x_d[basin][freq][key], before[basin][freq][key],
                                           rtol=0, atol=0, equal_nan=True)
            torch.testing.assert_close(ds._y[basin][freq], targets[basin][freq], rtol=0, atol=0, equal_nan=True)
    val = LamaH(cfg, is_train=False, period='validation', basin='1', scaler=ds.scaler)
    ordinary = next(iter(DataLoader(val, batch_size=8, collate_fn=val.collate_fn)))
    fast = next(iter(GpuBatchLoader(val, batch_size=8, device='cpu', shuffle=False, with_dates=True)))
    for phase in ('x_d_hindcast', 'x_d_forecast'):
        for key in ordinary[phase]:
            torch.testing.assert_close(ordinary[phase][key], fast[phase][key], rtol=0, atol=0, equal_nan=True)
            assert ordinary[phase][key].isfinite().all()


@pytest.mark.parametrize('vectorized', [False, True])
def test_b3_all_groups_missing_at_one_step_is_supported(b2_config, vectorized):
    cfg = b2_config(nan_step_probability=1., nan_sequence_probability=0.)
    ds = LamaH(cfg, is_train=True, period='train', scaler={})
    loader = (GpuBatchLoader(ds, batch_size=4, device='cpu', shuffle=False)
              if vectorized else DataLoader(ds, batch_size=4, collate_fn=ds.collate_fn))
    batch = next(iter(loader))
    assert all(v.isnan().all() for phase in ('x_d_hindcast', 'x_d_forecast') for v in batch[phase].values())
    assert SequentialForecastLSTM(cfg)(batch)['y_hat'].isfinite().all()


@pytest.mark.parametrize('vectorized', [False, True])
def test_b3_forced_whole_sequence_draw_restores_one_source(b2_config, vectorized):
    cfg = b2_config(nan_step_probability=0., nan_sequence_probability=1.)
    ds = LamaH(cfg, is_train=True, period='train', scaler={})
    batch = next(iter(GpuBatchLoader(ds, batch_size=4, device='cpu', shuffle=False))) if vectorized else ds.collate_fn([ds[i] for i in range(4)])
    # With one ECMWF source, a whole-sequence draw is restored; step masking is separate.
    assert batch['x_d_forecast']['ECMWF_a'].isfinite().all()
    # This guard restores a source, not observations naturally missing in the raw input.
    kept = torch.stack([v.isfinite().any(dim=1).squeeze(-1) for v in batch['x_d_hindcast'].values()])
    assert kept.any(dim=0).all()
