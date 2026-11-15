# Tests of the gefs variant of the 7-day forecast (step H): configure_lamah_h(mode='gefs'), the lead-wise masking in
# both loaders, causality (forecast day k sees only the GEFS run issued on the first forecast day), missing GEFS cells,
# the masked-mean embedding per forecast day, the parameter count and a short training + evaluation.
# Synthetic LamaH files of test_h_lead_time with GEFS_<variable>_lead<k> columns added.
import copy

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from neuralhydrology.datasetzoo.lamah import LamaH
from neuralhydrology.datautils.forecast_inputs import prepare_forecast_inputs
from neuralhydrology.datautils.gpu_loader import GpuBatchLoader
from neuralhydrology.datautils.source_groups import LAMAH_GEFS_VARIABLES, configure_lamah_h
from neuralhydrology.evaluation.evaluate import start_evaluation
from neuralhydrology.evaluation.lead_metrics import lead_time_metric_names
from neuralhydrology.modelzoo import get_model
from neuralhydrology.modelzoo.sf_cores import count_parameters
from neuralhydrology.training.train import start_training
from test.test_h_lead_time import (CPU_CORES, ECMWF, HINDCAST_GROUPS, HINDCAST_STEPS, HORIZON, TRAINING, _predict,
                                   _predict_changed, _window, h_config)

GEFS = [[f'GEFS_{name}_lead{k}' for name in LAMAH_GEFS_VARIABLES] for k in range(1, HORIZON + 1)]
GEFS_NAMES = [name for group in GEFS for name in group]
# Missing cells of basin 1 as in the real data: one message (ssrd, lead 3, Jan 15) and one missing file (tp of the run
# issued on Jan 20 => lead k on Jan 20 + k - 1).
GAP_CELL = ('2003-01-15', 'GEFS_ssrd_lead3')
GAP_RUN = pd.Timestamp('2003-01-20')


@pytest.fixture
def g_config(h_config, tmp_path):
    folder = tmp_path / 'A_basins_total_upstrm/2_timeseries/daily'
    for basin in (1, 2):
        path = folder / f'ID_{basin}.csv'
        frame = pd.read_csv(path, sep=';')
        dates = pd.to_datetime(dict(year=frame['YYYY'], month=frame['MM'], day=frame['DD']))
        day = np.arange(len(frame), dtype=float)
        for k in range(1, HORIZON + 1):
            for j, name in enumerate(LAMAH_GEFS_VARIABLES):
                frame[f'GEFS_{name}_lead{k}'] = np.sin(day / (6 + j)) + basin + .5 * k
        if basin == 1:
            frame.loc[dates == GAP_CELL[0], GAP_CELL[1]] = np.nan
            for k in range(1, HORIZON + 1):
                frame.loc[dates == GAP_RUN + pd.Timedelta(days=k - 1), f'GEFS_tp_lead{k}'] = np.nan
        frame.to_csv(path, sep=';', index=False)

    def make(**extra):
        return h_config('gefs', **extra)
    return make


def test_configure_gefs(h_config):
    base = h_config()
    before = copy.deepcopy(base.as_dict())
    cfg = configure_lamah_h(base, 'gefs', horizon=HORIZON)
    assert base.as_dict() == before
    assert (cfg.seq_length, cfg.forecast_seq_length, cfg.predict_last_n) == (HINDCAST_STEPS + HORIZON, HORIZON,
                                                                             HORIZON)
    assert cfg.timestep_counter and cfg.forecast_input_mode == 'gefs'
    assert cfg.hindcast_inputs == HINDCAST_GROUPS
    assert cfg.forecast_inputs == [ECMWF] + GEFS
    assert cfg.dynamic_inputs == HINDCAST_GROUPS + [ECMWF] + GEFS
    allowed = {'seq_length', 'forecast_seq_length', 'predict_last_n', 'timestep_counter', 'forecast_input_mode',
               'forecast_inputs', 'dynamic_inputs'}
    for key, value in before.items():
        if key not in allowed:
            assert cfg.as_dict()[key] == value
    short = configure_lamah_h(base, 'gefs', horizon=3)
    assert short.forecast_inputs == [ECMWF] + GEFS[:3] and short.seq_length == HINDCAST_STEPS + 3
    for variables in ([], ['t2m', 't2m']):
        with pytest.raises(ValueError, match='distinct GEFS'):
            configure_lamah_h(base, 'gefs', gefs_variables=variables)
    with pytest.raises(ValueError, match='grouped one-day'):
        configure_lamah_h(cfg, 'gefs')
    bad = copy.deepcopy(base)
    bad.update_config({'forecast_input_mode': 'gefs7'})
    with pytest.raises(ValueError, match='forecast_input_mode'):
        bad.forecast_input_mode


def test_gefs_groups_are_checked(h_config):
    cfg = h_config('gefs')
    data = {'x_d': {'hindcast_counter': torch.zeros(HINDCAST_STEPS, 1), 'forecast_counter': torch.zeros(HORIZON, 1)},
            'x_d_hindcast': {}, 'x_d_forecast': {}}
    mixed = [ECMWF[:4] + [GEFS[0][0]]] + GEFS
    renamed = [ECMWF] + GEFS[:3] + [[name.replace('ssrd', 'str') for name in GEFS[3]]] + GEFS[4:]
    for forecast in ([ECMWF] + GEFS[:-1], [ECMWF] + GEFS[1:2] + GEFS[:1] + GEFS[2:], GEFS + [ECMWF], mixed, renamed,
                     [ECMWF] + [group[:4] for group in GEFS[:-1]] + [GEFS[-1]], [ECMWF] + GEFS + [GEFS[0]]):
        bad = copy.deepcopy(cfg)
        bad.update_config({'forecast_inputs': forecast})
        with pytest.raises(ValueError, match='gefs requires'):
            prepare_forecast_inputs(copy.deepcopy(data), bad)
    # The HRES-only variants never accept GEFS columns.
    hybrid = h_config('hybrid')
    hybrid.update_config({'forecast_inputs': [ECMWF] + GEFS})
    with pytest.raises(ValueError, match='only ECMWF HRES'):
        prepare_forecast_inputs(copy.deepcopy(data), hybrid)


def test_gefs_samples_and_loaders(g_config):
    cfg = g_config()
    ds = LamaH(cfg, is_train=True, period='train', scaler={})
    # Missing GEFS cells never remove a sample (masked mean, no input check in the sample filter).
    ends = sorted(ds[i]['date'][-1] for i in range(len(ds)))
    assert ends == sorted(list(pd.date_range('2003-01-07', '2003-01-31').values) * 2)

    cpu = list(DataLoader(ds, batch_size=4, collate_fn=ds.collate_fn))
    gpu = list(GpuBatchLoader(ds, batch_size=4, device='cpu', shuffle=False, with_dates=True))
    assert len(cpu) == len(gpu) == 13
    for regular, vectorized in zip(cpu, gpu):
        assert regular.keys() == vectorized.keys()
        for key in regular:
            if key == 'date':
                np.testing.assert_array_equal(regular[key], vectorized[key])
            elif isinstance(regular[key], dict):
                assert regular[key].keys() == vectorized[key].keys()
                for feature in regular[key]:
                    torch.testing.assert_close(regular[key][feature], vectorized[key][feature], rtol=0, atol=0,
                                               equal_nan=True)
            else:
                torch.testing.assert_close(regular[key], vectorized[key], rtol=0, atol=0, equal_nan=True)

    n_gap_steps = 0
    for b, batch in enumerate(gpu):
        forecast = batch['x_d_forecast']
        assert set(forecast) == set(ECMWF) | set(GEFS_NAMES) | {'forecast_counter'}
        for row in range(len(batch['y'])):
            basin, first, last = _window(ds, 4 * b + row)
            stored = ds._x_d[basin]['1D']
            # HRES of the first forecast day only; GEFS lead k on forecast day k only, with its stored value.
            for name in ECMWF:
                assert torch.isfinite(forecast[name][row, 0]).all() and torch.isnan(forecast[name][row, 1:]).all()
            for k, group in enumerate(GEFS, start=1):
                for name in group:
                    values = forecast[name][row, :, 0]
                    others = torch.cat([values[:k - 1], values[k:]])
                    assert torch.isnan(others).all()
                    torch.testing.assert_close(values[k - 1], stored[name][first + k - 1, 0], rtol=0, atol=0,
                                               equal_nan=True)
                    n_gap_steps += int(torch.isnan(values[k - 1]))
    # Gaps of basin 1 reach exactly: ssrd lead 3 on Jan 15 => forecast day 3 of the sample starting on Jan 13; the tp
    # run of Jan 20 => every forecast day of the sample starting on Jan 20 (and of no other sample).
    assert n_gap_steps == 1 + HORIZON


def test_gefs_only_issued_runs_reach_the_forecast(g_config):
    cfg = g_config()
    ds = LamaH(cfg, is_train=True, period='train', scaler={})
    torch.manual_seed(0)
    model = get_model(cfg).eval()
    basin, first, last = _window(ds)
    reference = _predict(model, ds)
    assert reference.shape == (HINDCAST_STEPS + HORIZON,) and torch.isfinite(reference).all()
    start = first - HINDCAST_STEPS
    for k, group in enumerate(GEFS, start=1):
        issued = first + k - 1
        # Rows of other issuances (later runs, or earlier runs of the same lead) never change the sample.
        assert torch.equal(reference, _predict_changed(model, ds, group, start, issued))
        assert torch.equal(reference, _predict_changed(model, ds, group, issued + 1, last + 1))
        # The run issued on the first forecast day changes forecast day k and later, nothing before it.
        changed = _predict_changed(model, ds, group, issued, issued + 1)
        assert torch.equal(reference[:HINDCAST_STEPS + k - 1], changed[:HINDCAST_STEPS + k - 1])
        assert (reference[HINDCAST_STEPS + k - 1] - changed[HINDCAST_STEPS + k - 1]).abs() > 1e-6
    # HRES: first forecast day only, as hybrid.
    assert torch.equal(reference, _predict_changed(model, ds, ECMWF, first + 1, last + 1))
    changed = _predict_changed(model, ds, ['ECMWF_tp'], first, first + 1)
    assert torch.equal(reference[:HINDCAST_STEPS], changed[:HINDCAST_STEPS])
    assert (reference[HINDCAST_STEPS] - changed[HINDCAST_STEPS]).abs() > 1e-6


def test_gefs_with_training_masks(g_config):
    # The input masks of B3 (P10S12) drop whole groups or steps before the lead-wise masking: they can hide a GEFS
    # value, never reveal one of another forecast day, in both loaders.
    cfg = g_config(nan_sequence_probability=.5, nan_step_probability=.5)
    ds = LamaH(cfg, is_train=True, period='train', scaler={})
    torch.manual_seed(0)
    np.random.seed(0)
    samples = [ds[i] for i in range(len(ds))]
    batches = list(GpuBatchLoader(ds, batch_size=8, device='cpu', shuffle=False))
    forecasts = [s['x_d_forecast'] for s in samples] + [{k: v[r] for k, v in b['x_d_forecast'].items()}
                                                        for b in batches for r in range(len(b['y']))]
    hidden = 0
    for forecast in forecasts:
        for name in ECMWF:
            assert torch.isnan(forecast[name][1:]).all()
        for k, group in enumerate(GEFS, start=1):
            for name in group:
                values = forecast[name][:, 0]
                assert torch.isnan(torch.cat([values[:k - 1], values[k:]])).all()
                hidden += int(torch.isnan(values[k - 1]))
    # The masks are active (far more hidden GEFS values than the 8 raw gaps of each loader).
    assert hidden > 100


@pytest.mark.parametrize('core', CPU_CORES)
def test_gefs_forward_backward_and_embedding(g_config, core):
    cfg = g_config(sf_core=core)
    ds = LamaH(cfg, is_train=True, period='train', scaler={})
    torch.manual_seed(0)
    model = get_model(cfg)
    net = model.forecast_embedding_net
    assert net._pos_enc is None and len(net.dynamics_embeddings) == 1 + HORIZON
    for batch in GpuBatchLoader(ds, batch_size=8, device='cpu', shuffle=False):
        pred = model(batch)['y_hat']
        assert torch.isfinite(pred).all()
    batch = next(iter(GpuBatchLoader(ds, batch_size=4, device='cpu', shuffle=False)))
    model.zero_grad()
    pred = model(batch)['y_hat']
    assert pred.shape == (4, HINDCAST_STEPS + HORIZON, 1)
    pred[:, -HORIZON:].square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    # Every forecast group receives a gradient: HRES on day 1, GEFS lead k on day k.
    for group in net.dynamics_embeddings:
        assert any(p.grad is not None and (p.grad != 0).any() for p in group.parameters())

    # Forecast embedding = [masked mean of the available groups (4), step counter (1), statics (2)]: day 1 averages
    # HRES and GEFS lead 1, day k > 1 is the GEFS lead-k group alone.
    with torch.no_grad():
        embedded = net(batch)
        x = batch['x_d_forecast']
        assert embedded.shape == (HORIZON, 4, 7)
        torch.testing.assert_close(embedded[..., 4:5], x['forecast_counter'].transpose(0, 1))
        hres = net.dynamics_embeddings[0](torch.cat([x[name][:, 0] for name in ECMWF], dim=-1))
        for k, group in enumerate(GEFS, start=1):
            own = net.dynamics_embeddings[k](torch.cat([x[name][:, k - 1] for name in group], dim=-1))
            expected = (hres + own) / 2 if k == 1 else own
            torch.testing.assert_close(embedded[k - 1, :, :4], expected)


def test_gefs_parameters(g_config, h_config):
    # gefs adds one embedding per lead time and nothing else.
    gefs = get_model(g_config())
    per_group = sum(p.numel() for p in gefs.forecast_embedding_net.dynamics_embeddings[1].parameters())
    assert per_group == len(LAMAH_GEFS_VARIABLES) * 4 + 4
    assert count_parameters(gefs) - count_parameters(get_model(h_config('hybrid'))) == HORIZON * per_group


def test_gefs_training_and_evaluation(g_config):
    cfg = g_config(**TRAINING)
    start_training(cfg)
    run_dir = cfg.run_dir
    lead_names = lead_time_metric_names(['NSE', 'KGE'], HORIZON)
    for epoch in (1, 2):
        frame = pd.read_csv(run_dir / 'validation' / f'model_epoch00{epoch}' / 'validation_metrics.csv',
                            dtype={'basin': str})
        assert set(lead_names) <= set(frame.columns) and frame['NSE_lead_mean'].notna().all()
    assert (run_dir / 'best_model.pt').exists()
    start_evaluation(cfg=cfg, run_dir=run_dir, epoch=2, period='test')
    metrics = pd.read_csv(run_dir / 'test' / 'model_epoch002' / 'test_metrics.csv', dtype={'basin': str})
    assert list(metrics.columns) == ['basin', 'NSE', 'KGE'] + lead_names and len(metrics) == 2
    assert np.isfinite(metrics[[name for name in lead_names if name.startswith('NSE')]].to_numpy()).all()
