import copy

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from neuralhydrology.datasetzoo.lamah import LamaH
from neuralhydrology.datautils.gpu_loader import GpuBatchLoader
from neuralhydrology.datautils.source_groups import configure_lamah_b2, lamah_source_groups
from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.sequential_forecast_lstm import SequentialForecastLSTM
from neuralhydrology.utils.config import Config


@pytest.fixture
def b2_config(tmp_path):
    dates = pd.date_range('2002-12-20', '2003-03-01')
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
        mean = 2 + basin + day * .03
        maximum = mean + .5 + .1 * np.cos(day)
        mean[dates == '2003-01-03'] = np.nan
        maximum[dates == '2003-01-07'] = np.nan
        calendar = dict(YYYY=dates.year, MM=dates.month, DD=dates.day)
        pd.DataFrame(dict(**calendar, ERA5L_a=day + basin, ECMWF_a=np.sin(day / 4) + basin,
                          qmean_mmd=mean * 86.4 / area, qmax_mmd=maximum * 86.4 / area)).to_csv(
            tmp_path / f'A_basins_total_upstrm/2_timeseries/daily/ID_{basin}.csv', sep=';', index=False)
        pd.DataFrame(dict(**calendar, qmean=mean, qmax=maximum)).to_csv(
            tmp_path / f'D_gauges/2_timeseries/daily/ID_{basin}.csv', sep=';', index=False)
    data = dict(experiment_name='synthetic_B2', run_dir=tmp_path, train_dir=tmp_path / 'train_data',
                data_dir=tmp_path, dataset='lamah_a', train_basin_file=tmp_path / 'basins.txt',
                validation_basin_file=tmp_path / 'basins.txt', test_basin_file=tmp_path / 'basins.txt',
                train_start_date='01/01/2003', train_end_date='12/01/2003',
                validation_start_date='13/01/2003', validation_end_date='20/01/2003',
                test_start_date='21/01/2003', test_end_date='28/01/2003',
                dynamic_inputs=[['ERA5L_a'], ['qmean_mmd', 'qmax_mmd'], ['ECMWF_a']],
                hindcast_inputs=[['ERA5L_a'], ['qmean_mmd', 'qmax_mmd']], forecast_inputs=[['ECMWF_a']],
                target_variables=['qmax'], static_attributes=['elev_mean'], seq_length=5,
                predict_last_n=1, forecast_seq_length=1, forecast_overlap=0, model='sequential_forecast_lstm',
                hidden_size=8, head='regression', output_activation='linear', initial_forget_bias=3.,
                output_dropout=0., dynamics_embedding=dict(type='fc', hiddens=[4], activation='linear', dropout=0.),
                statics_embedding=dict(type='fc', hiddens=[2], activation='linear', dropout=0.),
                nan_handling_method='masked_mean', timestep_counter=False, loss='NSE', batch_size=4,
                num_workers=0, verbose=0, save_train_data=False, device='cpu', epochs=1,
                learning_rate=.001, optimizer='Adam', seed=111)

    def make(**extra):
        return Config(dict(copy.deepcopy(data), **extra))
    return make


def test_grouped_lamah_and_one_day_causality(b2_config):
    cfg = b2_config()
    ds = LamaH(cfg, is_train=True, period='train', scaler={})
    assert set(LamaH._get_list_of_all_variables(cfg)) == {
        'qmax', 'ERA5L_a', 'ECMWF_a', 'qmean_mmd', 'qmax_mmd'}
    first = ds[0]
    assert first['date'][-1] == np.datetime64('2003-01-01')
    assert first['x_d_hindcast']['qmax_mmd'].shape == (4, 1)
    basin, indices = ds.lookup_table[0]
    last = indices[0]
    expected = ds._x_d[basin]['1D']['qmax_mmd'][last - 4:last]
    torch.testing.assert_close(first['x_d_hindcast']['qmax_mmd'], expected)
    # qmax is already converted in B1b forcing files; no second area conversion.
    raw = ds._load_basin_data('1')
    np.testing.assert_allclose(raw.qmax, raw.qmax_mmd, equal_nan=True)


def test_lookup_independent_calendar_and_labels(b2_config):
    cfg = b2_config()
    ds = LamaH(cfg, is_train=True, period='train', scaler={})
    dates = [ds[i]['date'][-1] for i in range(len(ds))]
    expected = list(pd.date_range('2003-01-01', '2003-01-12').difference(pd.DatetimeIndex(['2003-01-07'])))
    assert len(ds) == 22
    assert sorted(dates) == sorted(expected * 2)
    # Missing qmean input on Jan 3 is retained; a missing Jan 7 label is rejected.
    assert any(torch.isnan(ds[i]['x_d_hindcast']['qmean_mmd']).any() for i in range(len(ds)))
    legacy = LamaH(b2_config(nan_handling_method=None), is_train=True, period='train', scaler={})
    assert len(legacy) < len(ds)
    for i in range(len(legacy)):
        assert all(torch.isfinite(x).all() for x in legacy[i]['x_d_hindcast'].values())


def test_masked_mean_preserves_b1b_normalization(b2_config):
    baseline = LamaH(b2_config(nan_handling_method=None), is_train=True, period='train', scaler={})
    grouped = LamaH(b2_config(), is_train=True, period='train', scaler={})
    assert baseline.scaler.keys() == grouped.scaler.keys()
    # B2 changes the eligible windows, while fitting the same daily data as B1b.
    for key in ('xarray_feature_center', 'xarray_feature_scale'):
        assert baseline.scaler[key].identical(grouped.scaler[key])
    for key in ('attribute_means', 'attribute_stds'):
        pd.testing.assert_series_equal(baseline.scaler[key], grouped.scaler[key])
    assert baseline._per_basin_target_stds.keys() == grouped._per_basin_target_stds.keys()
    for basin in baseline._per_basin_target_stds:
        torch.testing.assert_close(baseline._per_basin_target_stds[basin],
                                   grouped._per_basin_target_stds[basin], rtol=0, atol=0, equal_nan=True)


def test_missing_static_excludes_training_basin(b2_config, monkeypatch):
    original = LamaH._load_combined_attributes

    def load(self):
        original(self)
        self._attributes['1'][:] = torch.nan

    monkeypatch.setattr(LamaH, '_load_combined_attributes', load)
    ds = LamaH(b2_config(), is_train=True, period='train', scaler={})
    assert len(ds) == 11 and {basin for basin, _ in ds.lookup_table.values()} == {'2'}


def test_evaluation_retains_missing_label(b2_config):
    ds = LamaH(b2_config(), is_train=True, period='train', scaler={})
    evaluation = LamaH(b2_config(), is_train=False, period='train', basin='1', scaler=ds.scaler)
    assert len(evaluation) == 12
    sample = next(evaluation[i] for i in range(len(evaluation)) if evaluation[i]['date'][-1] == np.datetime64('2003-01-07'))
    assert torch.isnan(sample['y'][-1]).all()


@pytest.mark.parametrize('horizon,counter', [(1, False), (1, True), (7, True)])
def test_vectorized_batches_match_cpu_and_do_not_mutate(b2_config, horizon, counter):
    cfg = b2_config(seq_length=4 + horizon, predict_last_n=horizon, forecast_seq_length=horizon,
                    timestep_counter=counter, forecast_input_mode='first_day' if horizon > 1 else 'all')
    ds = LamaH(cfg, is_train=True, period='train', scaler={})
    original = copy.deepcopy(ds._x_d)
    cpu = list(DataLoader(ds, batch_size=4, collate_fn=ds.collate_fn))
    gpu = list(GpuBatchLoader(ds, batch_size=4, device='cpu', shuffle=False, with_dates=True))
    assert len(cpu) == len(gpu)
    for regular, vectorized in zip(cpu, gpu):
        assert regular.keys() == vectorized.keys()
        for key in regular:
            if key == 'date':
                np.testing.assert_array_equal(regular[key], vectorized[key])
            elif isinstance(regular[key], dict):
                assert regular[key].keys() == vectorized[key].keys()
                for feature in regular[key]:
                    torch.testing.assert_close(regular[key][feature], vectorized[key][feature], equal_nan=True)
            else:
                torch.testing.assert_close(regular[key], vectorized[key], equal_nan=True)
    for basin, data in original.items():
        for feature, values in data['1D'].items():
            torch.testing.assert_close(values, ds._x_d[basin]['1D'][feature], equal_nan=True)
    if horizon > 1:
        assert torch.isnan(gpu[0]['x_d_forecast']['ECMWF_a'][:, 1:]).all()
        assert torch.isfinite(gpu[0]['x_d_forecast']['ECMWF_a'][:, 0]).all()


def test_masked_mean_arithmetic_and_all_missing(b2_config):
    layer = InputLayer(b2_config(), 'hindcast')
    for name, parameter in layer.named_parameters():
        parameter.data.fill_(0 if 'bias' in name else 1)
    # Independent expected means: (2 + (3+5))/2=5, group Q absent => 2, all absent => 0.
    data = dict(x_d_hindcast=dict(ERA5L_a=torch.tensor([[[2.], [2.], [float('nan')]]]),
                                 qmean_mmd=torch.tensor([[[3.], [float('nan')], [float('nan')]]]),
                                 qmax_mmd=torch.tensor([[[5.], [5.], [float('nan')]]])))
    embedding, _ = layer(data, concatenate_output=False)
    torch.testing.assert_close(embedding[:, 0], torch.tensor([[5.] * 4, [2.] * 4, [0.] * 4]))


@pytest.mark.parametrize('horizon', [1, 7])
def test_forward_backward_and_future_release_invariance(b2_config, horizon):
    cfg = b2_config(seq_length=4 + horizon, predict_last_n=horizon, forecast_seq_length=horizon,
                    timestep_counter=True, forecast_input_mode='first_day')
    ds = LamaH(cfg, is_train=True, period='train', scaler={})
    batch = next(iter(GpuBatchLoader(ds, batch_size=4, device='cpu', shuffle=False)))
    model = SequentialForecastLSTM(cfg)
    pred = model(batch)['y_hat']
    assert pred.shape == (4, 4 + horizon, 1) and torch.isfinite(pred).all()
    pred.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    embedded = model.forecast_embedding_net(batch)
    torch.testing.assert_close(embedded[..., 4:5], batch['x_d_forecast']['forecast_counter'].transpose(0, 1))
    # Later HRES issuances are unavailable. Changing their stored values cannot change this sample.
    basin, indices = ds.lookup_table[0]
    first = indices[0] - horizon + 1
    reference = ds.collate_fn([ds[0]])
    ds._x_d[basin]['1D']['ECMWF_a'][first + 1:first + horizon] = 1e6
    changed = ds.collate_fn([ds[0]])
    torch.testing.assert_close(model(reference)['y_hat'], model(changed)['y_hat'])


def test_first_day_rejects_unsupported_contracts(b2_config):
    with pytest.raises(ValueError, match='forecast_input_mode'):
        _ = b2_config(forecast_input_mode='invalid').forecast_input_mode
    for extra in (dict(timestep_counter=False), dict(forecast_inputs=[['qmax_mmd']]), dict(forecast_overlap=1)):
        cfg = b2_config(forecast_input_mode='first_day', timestep_counter=True, **{
            k: v for k, v in extra.items() if k != 'timestep_counter'})
        if 'timestep_counter' in extra:
            cfg.update_config(extra)
        ds = LamaH(cfg, is_train=True, period='train', scaler={})
        with pytest.raises(ValueError):
            _ = ds[0]


def test_b1b_to_b2_groups_without_other_scientific_changes(b2_config):
    hindcast = [f'{source}_{i}' for source, count in [('ERA5L', 21), ('EOBS', 7), ('MSWEP', 1), ('GLEAM', 2)]
                for i in range(count)] + ['qmean_mmd', 'qmax_mmd']
    forecast = [f'ECMWF_{i}' for i in range(5)]
    original = b2_config(dynamic_inputs=hindcast + forecast, hindcast_inputs=hindcast, forecast_inputs=forecast,
                         seq_length=365, epochs=30, early_stopping_patience=30, nan_handling_method=None)
    before = copy.deepcopy(original.as_dict())
    improved = configure_lamah_b2(original)
    assert original.as_dict() == before
    assert list(map(len, improved.hindcast_inputs)) == [21, 7, 1, 2, 2]
    assert list(map(len, improved.forecast_inputs)) == [5]
    allowed = {'dynamic_inputs', 'hindcast_inputs', 'forecast_inputs', 'nan_handling_method',
               'nan_step_probability', 'nan_sequence_probability', 'timestep_counter', 'forecast_input_mode'}
    for key, value in before.items():
        if key not in allowed:
            assert improved.as_dict()[key] == value
    with pytest.raises(ValueError):
        lamah_source_groups(['qmean', 'qmax'])
    with pytest.raises(ValueError):
        lamah_source_groups(['ERA5L_a', 'ERA5L_a'])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_cuda_vectorized_model(b2_config):
    cfg = b2_config(seq_length=11, forecast_seq_length=7, predict_last_n=7,
                    timestep_counter=True, forecast_input_mode='first_day')
    ds = LamaH(cfg, is_train=True, period='train', scaler={})
    cpu = next(iter(DataLoader(ds, batch_size=4, collate_fn=ds.collate_fn)))
    cuda = next(iter(GpuBatchLoader(ds, batch_size=4, device='cuda', shuffle=False)))
    for key in ('x_d_hindcast', 'x_d_forecast'):
        for variable in cpu[key]:
            torch.testing.assert_close(cpu[key][variable], cuda[key][variable].cpu(), equal_nan=True)
    model = SequentialForecastLSTM(cfg).cuda()
    predicted = model(cuda)['y_hat']
    assert torch.isfinite(predicted).all()
    predicted.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
