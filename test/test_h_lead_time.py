# Tests of the 7-day forecast (step H): configure_lamah_h, the three forecast-input variants (past, hybrid, perfect) in
# both loaders, causality, the model with every CPU core, and training + evaluation with per-lead-time metrics.
# Synthetic LamaH files with the real variable names of the forecast phase; Mamba is checked on the GPU only.
import copy
import inspect
import pickle

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datasetzoo.lamah import LamaH
from neuralhydrology.datautils.gpu_loader import GpuBatchLoader
from neuralhydrology.datautils.source_groups import LAMAH_H_MODES, LAMAH_PERFECT_FORECAST, configure_lamah_h
from neuralhydrology.evaluation import get_tester
from neuralhydrology.evaluation.evaluate import start_evaluation
from neuralhydrology.evaluation.lead_metrics import lead_time_metric_names
from neuralhydrology.modelzoo import get_model
from neuralhydrology.modelzoo.sf_cores import count_parameters
from neuralhydrology.training.train import start_training
from neuralhydrology.utils.config import Config

HORIZON, HINDCAST_STEPS = 7, 4
ECMWF = ['ECMWF_t2m', 'ECMWF_d2m', 'ECMWF_ssrd', 'ECMWF_tp', 'ECMWF_e']
REANALYSIS = list(LAMAH_PERFECT_FORECAST)
HINDCAST_GROUPS = [['ERA5L_2m_temp_mean', 'ERA5L_2m_dp_temp_mean'], ['EOBS_qq'], ['MSWEP_RR'], ['GLEAM_ETA'],
                   ['qmean_mmd', 'qmax_mmd']]
MODES = list(LAMAH_H_MODES)
CPU_CORES = ['lstm', 'gru', 'transformer', 's4d']
TRAINING = dict(lead_time_metrics=True, early_stopping_metric='NSE_lead_mean', early_stopping_mode='max',
                early_stopping_patience=2, validate_every=1, validate_n_random_basins=2, metrics=['NSE', 'KGE'],
                clip_targets_to_zero=['qmax'], log_tensorboard=False, log_n_figures=0, save_validation_results=True,
                data_on_gpu=True, epochs=2)


@pytest.fixture
def h_config(tmp_path):
    # One-day grouped configuration (as after configure_lamah_b2, shortened); h_config(mode) extends it to 7 days.
    dates = pd.date_range('2002-12-01', '2003-04-30')
    for directory in ('A_basins_total_upstrm/1_attributes', 'D_gauges/1_attributes',
                      'A_basins_total_upstrm/2_timeseries/daily', 'D_gauges/2_timeseries/daily'):
        (tmp_path / directory).mkdir(parents=True)
    pd.DataFrame({'ID': [1, 2], 'elev_mean': [100., 300.]}).to_csv(
        tmp_path / 'A_basins_total_upstrm/1_attributes/Catchment_attributes.csv', sep=';', index=False)
    pd.DataFrame({'ID': [1, 2], 'area_gov': [43.2, 86.4]}).to_csv(
        tmp_path / 'D_gauges/1_attributes/Gauge_attributes.csv', sep=';', index=False)
    (tmp_path / 'basins.txt').write_text('1\n2\n')
    for basin, area in ((1, 43.2), (2, 86.4)):
        day = np.arange(len(dates), dtype=float)
        mean = 2 + basin + day * .03 + .4 * np.sin(day / 5)
        maximum = mean + .5 + .1 * np.cos(day)
        # A missing Q input (train), a missing label in the train period and one in the test period.
        mean[dates == '2003-01-03'] = np.nan
        maximum[dates == '2003-01-07'] = np.nan
        maximum[dates == '2003-03-10'] = np.nan
        calendar = dict(YYYY=dates.year, MM=dates.month, DD=dates.day)
        forcings = {name: np.sin(day / (3 + j)) + basin + j for j, name in enumerate(REANALYSIS)}
        forcings.update({name: np.cos(day / (4 + j)) + basin - j for j, name in enumerate(ECMWF)})
        pd.DataFrame(dict(**calendar, **forcings, qmean_mmd=mean * 86.4 / area,
                          qmax_mmd=maximum * 86.4 / area)).to_csv(
            tmp_path / f'A_basins_total_upstrm/2_timeseries/daily/ID_{basin}.csv', sep=';', index=False)
        pd.DataFrame(dict(**calendar, qmean=mean, qmax=maximum)).to_csv(
            tmp_path / f'D_gauges/2_timeseries/daily/ID_{basin}.csv', sep=';', index=False)
    data = dict(experiment_name='synthetic_H', run_dir=tmp_path / 'runs', train_dir=tmp_path / 'train_data',
                data_dir=tmp_path, dataset='lamah_a', train_basin_file=tmp_path / 'basins.txt',
                validation_basin_file=tmp_path / 'basins.txt', test_basin_file=tmp_path / 'basins.txt',
                train_start_date='01/01/2003', train_end_date='31/01/2003',
                validation_start_date='01/02/2003', validation_end_date='28/02/2003',
                test_start_date='01/03/2003', test_end_date='31/03/2003',
                dynamic_inputs=HINDCAST_GROUPS + [ECMWF], hindcast_inputs=HINDCAST_GROUPS, forecast_inputs=[ECMWF],
                target_variables=['qmax'], static_attributes=['elev_mean'], seq_length=HINDCAST_STEPS + 1,
                predict_last_n=1, forecast_seq_length=1, forecast_overlap=0, model='sequential_forecast',
                sf_core='lstm', sf_layers=2, sf_s4d_lr=1e-4, hidden_size=16, head='regression',
                output_activation='linear', initial_forget_bias=3., output_dropout=0.,
                dynamics_embedding=dict(type='fc', hiddens=[4], activation='linear', dropout=0.),
                statics_embedding=dict(type='fc', hiddens=[2], activation='linear', dropout=0.),
                nan_handling_method='masked_mean', timestep_counter=False, loss='NSE', batch_size=4,
                num_workers=0, verbose=0, save_train_data=False, device='cpu', epochs=1,
                learning_rate=.001, optimizer='Adam', seed=111)

    def make(mode=None, **extra):
        cfg = Config(dict(copy.deepcopy(data), **extra))
        return cfg if mode is None else configure_lamah_h(cfg, mode, horizon=HORIZON)
    return make


def _window(ds, index=0):
    # Stored rows of the first and the last forecast day of a sample.
    basin, indices = ds.lookup_table[index]
    last = int(indices[0])
    return basin, last - HORIZON + 1, last


def _predict(model, ds, index=0):
    with torch.no_grad():
        return model(ds.collate_fn([ds[index]]))['y_hat'][0, :, 0]


def _predict_changed(model, ds, names, start, stop, index=0):
    # Prediction of a sample after overwriting stored rows [start, stop) of the given variables; the store is restored.
    basin = ds.lookup_table[index][0]
    stored = ds._x_d[basin]['1D']
    backup = {name: stored[name][start:stop].clone() for name in names}
    for name in names:
        stored[name][start:stop] = 1e3
    changed = _predict(model, ds, index)
    for name in names:
        stored[name][start:stop] = backup[name]
    return changed


def test_configure_lamah_h(h_config):
    base = h_config()
    before = copy.deepcopy(base.as_dict())
    for mode, forecast_mode in LAMAH_H_MODES.items():
        cfg = configure_lamah_h(base, mode)
        assert base.as_dict() == before
        assert (cfg.seq_length, cfg.forecast_seq_length, cfg.predict_last_n) == (HINDCAST_STEPS + HORIZON, HORIZON,
                                                                                 HORIZON)
        assert cfg.timestep_counter and cfg.forecast_input_mode == forecast_mode
        assert cfg.hindcast_inputs == HINDCAST_GROUPS
        allowed = {'seq_length', 'forecast_seq_length', 'predict_last_n', 'timestep_counter', 'forecast_input_mode'}
        if mode == 'perfect':
            # Reanalysis of the forecast days replaces HRES; the columns are listed once for the loader.
            allowed |= {'forecast_inputs', 'dynamic_inputs'}
            assert cfg.forecast_inputs == [REANALYSIS] and cfg.dynamic_inputs == HINDCAST_GROUPS
        else:
            assert cfg.forecast_inputs == [ECMWF] and cfg.dynamic_inputs == HINDCAST_GROUPS + [ECMWF]
        for key, value in before.items():
            if key not in allowed:
                assert cfg.as_dict()[key] == value
    assert configure_lamah_h(base, 'hybrid', horizon=3).seq_length == HINDCAST_STEPS + 3


def test_configure_lamah_h_rejects(h_config):
    base = h_config()
    with pytest.raises(ValueError, match='H mode'):
        configure_lamah_h(base, 'future')
    for horizon in (1, 0, 7.0):
        with pytest.raises(ValueError, match='horizon'):
            configure_lamah_h(base, 'past', horizon=horizon)
    # Only a one-day grouped masked-mean configuration is extended, never an H configuration again.
    for cfg in (configure_lamah_h(base, 'hybrid'), h_config(nan_handling_method='input_replacing'),
                h_config(forecast_inputs=[['qmax_mmd']]), h_config(forecast_inputs=[ECMWF[:2], ECMWF[2:]]),
                h_config(forecast_overlap=1)):
        with pytest.raises(ValueError, match='grouped one-day'):
            configure_lamah_h(cfg, 'past')
    for perfect in (REANALYSIS[:4], REANALYSIS[:4] + REANALYSIS[:1]):
        with pytest.raises(ValueError, match='one distinct'):
            configure_lamah_h(base, 'perfect', perfect_inputs=perfect)
    with pytest.raises(ValueError, match='not hindcast inputs'):
        configure_lamah_h(base, 'perfect', perfect_inputs=['ERA5L_total_et'] + REANALYSIS[1:])


@pytest.mark.parametrize('mode', MODES)
def test_samples_and_loaders(h_config, mode):
    cfg = h_config(mode)
    ds = LamaH(cfg, is_train=True, period='train', scaler={})
    # Every sample holds 7 target days: the first one ends 6 days after the start of the period. A missing label
    # (Jan 7) no longer removes a sample, the other targets of the sample remain.
    ends = sorted(ds[i]['date'][-1] for i in range(len(ds)))
    assert ends == sorted(list(pd.date_range('2003-01-07', '2003-01-31').values) * 2)
    sample = ds[0]
    assert sample['y'].shape == (HINDCAST_STEPS + HORIZON, 1)
    assert sample['x_d_hindcast']['qmax_mmd'].shape == (HINDCAST_STEPS, 1)

    original = copy.deepcopy(ds._x_d)
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
    for basin, data in original.items():
        for feature, values in data['1D'].items():
            torch.testing.assert_close(values, ds._x_d[basin]['1D'][feature], rtol=0, atol=0, equal_nan=True)

    forecast = gpu[0]['x_d_forecast']
    names = REANALYSIS if mode == 'perfect' else ECMWF
    assert set(forecast) == set(names) | {'forecast_counter'}
    assert set(gpu[0]['x_d_hindcast']) == {name for group in HINDCAST_GROUPS for name in group} | {'hindcast_counter'}
    assert all(forecast[name].shape == (4, HORIZON, 1) for name in names)
    if mode == 'past':
        assert all(torch.isnan(forecast[name]).all() for name in names)
    elif mode == 'hybrid':
        assert all(torch.isfinite(forecast[name][:, 0]).all() for name in names)
        assert all(torch.isnan(forecast[name][:, 1:]).all() for name in names)
    else:
        # Reanalysis of the 7 forecast days themselves, and no HRES column is loaded at all.
        assert not any(name.startswith('ECMWF_') for data in ds._x_d.values() for name in data['1D'])
        for row in range(4):
            basin, first, last = _window(ds, row)
            for name in names:
                torch.testing.assert_close(forecast[name][row], ds._x_d[basin]['1D'][name][first:last + 1],
                                           rtol=0, atol=0)
                torch.testing.assert_close(gpu[0]['x_d_hindcast'][name][row],
                                           ds._x_d[basin]['1D'][name][first - HINDCAST_STEPS:first], rtol=0, atol=0)


@pytest.mark.parametrize('mode', MODES)
def test_only_released_inputs_reach_the_forecast(h_config, mode):
    cfg = h_config(mode)
    ds = LamaH(cfg, is_train=True, period='train', scaler={})
    torch.manual_seed(0)
    model = get_model(cfg).eval()
    basin, first, last = _window(ds)
    reference = _predict(model, ds)
    assert reference.shape == (HINDCAST_STEPS + HORIZON,) and torch.isfinite(reference).all()

    # Discharge of the forecast days is never an input.
    assert torch.equal(reference, _predict_changed(model, ds, ['qmean_mmd', 'qmax_mmd'], first, last + 1))
    # Positive control: the last hindcast day changes its own step and every forecast step, nothing before it.
    changed = _predict_changed(model, ds, ['MSWEP_RR'], first - 1, first)
    assert torch.equal(reference[:HINDCAST_STEPS - 1], changed[:HINDCAST_STEPS - 1])
    assert ((reference - changed).abs()[HINDCAST_STEPS - 1:] > 1e-6).all()

    if mode == 'past':
        # No HRES value of any forecast day reaches the model.
        assert torch.equal(reference, _predict_changed(model, ds, ECMWF, first, last + 1))
    elif mode == 'hybrid':
        # Only the HRES issued for the first forecast day is available; later issuances cannot change the sample.
        assert torch.equal(reference, _predict_changed(model, ds, ECMWF, first + 1, last + 1))
        changed = _predict_changed(model, ds, ['ECMWF_tp'], first, first + 1)
        assert torch.equal(reference[:HINDCAST_STEPS], changed[:HINDCAST_STEPS])
        assert ((reference - changed).abs()[HINDCAST_STEPS:] > 1e-6).all()
    else:
        # Reanalysis of a forecast day is used from that day on (upper bound), never earlier.
        changed = _predict_changed(model, ds, ['MSWEP_RR'], last, last + 1)
        assert torch.equal(reference[:-1], changed[:-1])
        assert (reference[-1] - changed[-1]).abs() > 1e-6
        changed = _predict_changed(model, ds, ['MSWEP_RR'], first, first + 1)
        assert torch.equal(reference[:HINDCAST_STEPS], changed[:HINDCAST_STEPS])
        assert ((reference - changed).abs()[HINDCAST_STEPS:] > 1e-6).all()


@pytest.mark.parametrize('core', CPU_CORES)
@pytest.mark.parametrize('mode', MODES)
def test_forward_backward(h_config, mode, core):
    cfg = h_config(mode, sf_core=core)
    ds = LamaH(cfg, is_train=True, period='train', scaler={})
    batch = next(iter(GpuBatchLoader(ds, batch_size=4, device='cpu', shuffle=False)))
    torch.manual_seed(0)
    model = get_model(cfg)
    pred = model(batch)['y_hat']
    assert pred.shape == (4, HINDCAST_STEPS + HORIZON, 1) and torch.isfinite(pred).all()
    pred[:, -HORIZON:].square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

    # Forecast embedding = [masked mean of the forecast group (4), step counter (1), statics (2)].
    embedded = model.forecast_embedding_net(batch)
    assert embedded.shape == (HORIZON, 4, 7)
    torch.testing.assert_close(embedded[..., 4:5], batch['x_d_forecast']['forecast_counter'].transpose(0, 1))
    group = model.forecast_embedding_net.dynamics_embeddings
    gradients = [p.grad for p in group.parameters() if p.grad is not None]
    if mode == 'past':
        # The hidden group contributes zeros and receives no gradient: the forecast steps see counter and statics.
        assert (embedded[..., :4] == 0).all()
        assert all((gradient == 0).all() for gradient in gradients)
    else:
        assert (embedded[0, :, :4] != 0).any()
        assert any((gradient != 0).any() for gradient in gradients)
        hidden = embedded[1:, :, :4]
        assert (hidden == 0).all() if mode == 'hybrid' else (hidden != 0).any()


def test_same_parameters_in_all_modes(h_config):
    # The three variants differ in the forecast inputs only, never in the size of the model.
    counts = {mode: count_parameters(get_model(h_config(mode))) for mode in MODES}
    assert len(set(counts.values())) == 1, counts


def _hand_metrics(xr):
    # Tester layout: row j = sample ending on date j, column i = lead i + 1. Target day t (row index of the sample that
    # ends on it) at lead k is row t + HORIZON - k, column k - 1; the days t = 0 .. n - HORIZON hold every lead.
    obs = xr['qmax_obs'].values.astype(np.float64)
    sim = xr['qmax_sim'].values.astype(np.float64)
    sim = np.where(sim < 0, 0, sim)
    rows = np.arange(obs.shape[0] - HORIZON + 1)
    lead_obs = np.stack([obs[rows + HORIZON - k, k - 1] for k in range(1, HORIZON + 1)], axis=1)
    lead_sim = np.stack([sim[rows + HORIZON - k, k - 1] for k in range(1, HORIZON + 1)], axis=1)
    valid = ~np.isnan(lead_obs).any(axis=1) & ~np.isnan(lead_sim).any(axis=1)
    assert np.array_equal(lead_obs[valid], np.repeat(lead_obs[valid, :1], HORIZON, axis=1))

    def nse(o, s):
        return 1 - ((s - o)**2).sum() / ((o - o.mean())**2).sum()

    values = {f'NSE_lead{k}': nse(lead_obs[valid, k - 1], lead_sim[valid, k - 1]) for k in range(1, HORIZON + 1)}
    values['NSE_lead_mean'] = np.mean(list(values.values()))
    # Default metric of the tester: last time step of every sample, with its own mask.
    last = ~np.isnan(obs[:, -1]) & ~np.isnan(sim[:, -1])
    values['NSE'] = nse(obs[last, -1], sim[last, -1])
    return values, int(valid.sum())


@pytest.mark.parametrize('mode', MODES)
def test_training_selects_on_lead_mean_and_evaluates_per_lead(h_config, mode):
    cfg = h_config(mode, **TRAINING)
    start_training(cfg)
    run_dir = cfg.run_dir
    lead_names = lead_time_metric_names(['NSE', 'KGE'], HORIZON)

    # best_model.pt = epoch with the highest median over basins of the NSE averaged over the 7 lead times.
    medians = []
    for epoch in (1, 2):
        frame = pd.read_csv(run_dir / 'validation' / f'model_epoch00{epoch}' / 'validation_metrics.csv',
                            dtype={'basin': str})
        assert set(['NSE', 'KGE'] + lead_names) <= set(frame.columns) and len(frame) == 2
        assert frame['NSE_lead_mean'].notna().all()
        medians.append(float(np.nanmedian(frame['NSE_lead_mean'])))
    best = int(np.argmax(medians)) + 1
    best_weights = torch.load(run_dir / 'best_model.pt', weights_only=True)
    epoch_weights = torch.load(run_dir / f'model_epoch00{best}.pt', weights_only=True)
    assert best_weights.keys() == epoch_weights.keys()
    assert all(torch.equal(best_weights[name], epoch_weights[name]) for name in best_weights)
    if medians[0] != medians[1]:
        other = torch.load(run_dir / f'model_epoch00{3 - best}.pt', weights_only=True)
        assert any(not torch.equal(best_weights[name], other[name]) for name in best_weights)

    start_evaluation(cfg=cfg, run_dir=run_dir, epoch=2, period='test')
    folder = run_dir / 'test' / 'model_epoch002'
    metrics = pd.read_csv(folder / 'test_metrics.csv', dtype={'basin': str}).set_index('basin')
    assert list(metrics.columns) == ['NSE', 'KGE'] + lead_names and len(metrics) == 2
    # The pickle was written a few lines above by this very test run (tester output in tmp_path), not external data.
    with (folder / 'test_results.p').open('rb') as fp:
        results = pickle.load(fp)
    for basin in ('1', '2'):
        xr = results[basin]['1D']['xr']
        # Samples end on Mar 7 .. Mar 31; Mar 7 .. Mar 25 are forecast at every lead, Mar 10 has no observation.
        assert list(xr['time_step'].values) == list(range(-(HORIZON - 1), 1))
        assert pd.DatetimeIndex(xr['date'].values).equals(pd.date_range('2003-03-07', '2003-03-31'))
        hand, n_days = _hand_metrics(xr)
        assert n_days == 18
        for name in [f'NSE_lead{k}' for k in range(1, HORIZON + 1)] + ['NSE_lead_mean']:
            assert metrics.loc[basin, name] == pytest.approx(hand[name], rel=1e-6, abs=1e-8)
        # float32 computation in the tester.
        assert metrics.loc[basin, 'NSE'] == pytest.approx(hand['NSE'], rel=1e-4, abs=1e-5)

    # Without the flag the evaluation is the one of the library: same default metrics, no lead columns.
    cfg.update_config({'lead_time_metrics': False})
    start_evaluation(cfg=cfg, run_dir=run_dir, epoch=2, period='test')
    plain = pd.read_csv(folder / 'test_metrics.csv', dtype={'basin': str}).set_index('basin')
    assert list(plain.columns) == ['NSE', 'KGE']
    pd.testing.assert_frame_equal(plain, metrics[['NSE', 'KGE']])


def test_unknown_selection_metric_stops_training(h_config):
    # A lead-time run never falls back to the validation loss when the selection metric is misspelled.
    cfg = h_config('hybrid', **dict(TRAINING, early_stopping_metric='NSE_lead_men', epochs=1))
    with pytest.raises(ValueError, match='not found in validation'):
        start_training(cfg)


def test_additional_features_do_not_leak(h_config, tmp_path):
    # A data set built with additional_feature_files must not leave its features in the shared default list of
    # get_dataset: the next data set without them (e.g. the next test, or the next run in a process) would get them.
    dates = pd.date_range('2002-12-01', '2003-04-30', freq='1D', name='date')
    extra = {basin: pd.DataFrame({'extra': np.arange(len(dates), dtype=float)}, index=dates) for basin in ('1', '2')}
    path = tmp_path / 'extra.p'
    with path.open('wb') as fp:
        pickle.dump(extra, fp)
    default = inspect.signature(get_dataset).parameters['additional_features'].default
    with_extra = get_dataset(h_config(additional_feature_files=[path]), is_train=True, period='train', scaler={})
    assert len(with_extra.additional_features) == 1
    assert default == [] and inspect.signature(get_dataset).parameters['additional_features'].default == []
    without = get_dataset(h_config(), is_train=True, period='train', scaler={})
    assert without.additional_features == []


def test_lead_metrics_need_several_targets(h_config, tmp_path):
    cfg = h_config(lead_time_metrics=True, metrics=['NSE'])
    with pytest.raises(ValueError, match='predict_last_n'):
        get_tester(cfg=cfg, run_dir=tmp_path, period='validation', init_model=False)
    cfg = h_config('past', lead_time_metrics=True)
    with pytest.raises(ValueError, match='requires metrics'):
        get_tester(cfg=cfg, run_dir=tmp_path, period='validation', init_model=False)
