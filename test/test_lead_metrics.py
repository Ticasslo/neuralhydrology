# Tests of the per-lead-time metrics (evaluation/lead_metrics.py) against hand computations on a synthetic forecast.
# No model and no data set: the xarray layout of the tester is rebuilt from a [target day, lead] forecast matrix.
import types

import numpy as np
import pandas as pd
import pytest
import xarray

from neuralhydrology.evaluation.lead_metrics import (calculate_lead_time_metrics, check_lead_time_config,
                                                     lead_time_metric_names, lead_time_series)
from neuralhydrology.evaluation.metrics import calculate_metrics

N_LEADS, N_DAYS = 7, 400
# Days of the period that every lead forecasts: the first sample ends on day N_LEADS - 1, the last one on the last day.
FIRST, LAST = N_LEADS - 1, N_DAYS - N_LEADS + 1


def _synthetic():
    # truth[d] = observation of day d; forecast[d, k] = forecast of day d issued k + 1 days ahead. The error grows with
    # the lead time; the series holds missing observations, one missing simulation and one negative simulation.
    rng = np.random.default_rng(0)
    days = pd.date_range('2017-10-01', periods=N_DAYS, freq='1D')
    truth = np.abs(rng.normal(5, 2, N_DAYS)).astype(np.float32)
    truth[[50, 51, 120]] = np.nan
    forecast = np.stack([truth + rng.normal(0, 0.3 * (k + 1), N_DAYS) - 0.2 * k for k in range(N_LEADS)],
                        axis=1).astype(np.float32)
    forecast[np.isnan(forecast)] = 1.0
    forecast[200, 3] = np.nan
    forecast[10, 0] = -4.0
    # Tester layout: sample j ends on days[j + N_LEADS - 1]; its time_step i - (N_LEADS - 1) is day j + i at lead i + 1.
    n_samples = N_DAYS - N_LEADS + 1
    obs = np.stack([truth[j:j + N_LEADS] for j in range(n_samples)])
    sim = np.stack([[forecast[j + i, i] for i in range(N_LEADS)] for j in range(n_samples)])
    xr = xarray.Dataset({'qmax_obs': (('date', 'time_step'), obs), 'qmax_sim': (('date', 'time_step'), sim)},
                        coords={'date': days[N_LEADS - 1:], 'time_step': np.arange(-(N_LEADS - 1), 1)})
    return days, truth, forecast, xr


def _nse(obs, sim):
    return 1 - ((sim - obs)**2).sum() / ((obs - obs.mean())**2).sum()


def _kge(obs, sim):
    r = np.corrcoef(obs, sim)[0, 1]
    return 1 - np.sqrt((r - 1)**2 + (sim.std() / obs.std() - 1)**2 + (sim.mean() / obs.mean() - 1)**2)


def _hand_series(truth, forecast, clip):
    # Days with an observation and a simulation at every lead, in float64 as the helper.
    valid = ~np.isnan(truth[FIRST:LAST]) & ~np.isnan(forecast[FIRST:LAST]).any(axis=1)
    obs = truth[FIRST:LAST].astype(np.float64)[valid]
    sim = forecast[FIRST:LAST].astype(np.float64)[valid]
    return valid, obs, np.where(sim < 0, 0, sim) if clip else sim


def test_target_days_and_joint_mask():
    days, truth, forecast, xr = _synthetic()
    dates, obs, sim = lead_time_series(xr, 'qmax', clip_to_zero=True)
    assert dates.equals(days[FIRST:LAST])
    assert obs.shape == sim.shape == (LAST - FIRST, N_LEADS) and obs.dtype == sim.dtype == np.float64
    valid, hand_obs, hand_sim = _hand_series(truth, forecast, clip=True)
    # Three missing observations and one missing simulation remove four days, each of them at every lead.
    assert valid.sum() == LAST - FIRST - 4
    assert np.isnan(obs[~valid]).all() and np.isnan(sim[~valid]).all()
    assert np.isnan(sim[list(dates).index(days[200])]).all()
    for lead in range(N_LEADS):
        np.testing.assert_array_equal(obs[valid, lead], hand_obs)
        np.testing.assert_array_equal(sim[valid, lead], hand_sim[:, lead])
    assert (sim[valid] >= 0).all()


def test_metrics_match_hand_computation():
    _, truth, forecast, xr = _synthetic()
    values = calculate_lead_time_metrics(xr, 'qmax', ['NSE', 'KGE'], clip_to_zero=True)
    assert list(values) == lead_time_metric_names(['NSE', 'KGE'], N_LEADS)
    _, obs, sim = _hand_series(truth, forecast, clip=True)
    for name, function in (('NSE', _nse), ('KGE', _kge)):
        hand = [function(obs, sim[:, lead]) for lead in range(N_LEADS)]
        for lead in range(N_LEADS):
            assert values[f'{name}_lead{lead + 1}'] == pytest.approx(hand[lead], rel=1e-9, abs=1e-9)
        assert values[f'{name}_lead_mean'] == pytest.approx(np.mean(hand), rel=1e-9, abs=1e-9)
    # The synthetic error grows with the lead time.
    assert all(values[f'NSE_lead{lead}'] > values[f'NSE_lead{lead + 1}'] for lead in range(1, N_LEADS))


def test_default_metric_differs_from_last_lead():
    # The default metric of the tester takes the last time step of every sample: the same forecasts as the last lead,
    # but on all sample dates and with its own mask => close to, not equal to, <metric>_lead<n>.
    _, truth, forecast, xr = _synthetic()
    obs, sim = xr['qmax_obs'].isel(time_step=-1), xr['qmax_sim'].isel(time_step=-1)
    default = calculate_metrics(obs, xarray.where(sim < 0, 0, sim), metrics=['NSE'], resolution='1D')['NSE']
    hand_obs = truth[N_LEADS - 1:].astype(np.float64)
    hand_sim = forecast[N_LEADS - 1:, -1].astype(np.float64)
    valid = ~np.isnan(hand_obs) & ~np.isnan(hand_sim)
    # The tester metric runs on the float32 values of the xarray => looser tolerance than the float64 helper.
    assert default == pytest.approx(_nse(hand_obs[valid], np.where(hand_sim[valid] < 0, 0, hand_sim[valid])), rel=1e-6)
    last_lead = calculate_lead_time_metrics(xr, 'qmax', ['NSE'], clip_to_zero=True)[f'NSE_lead{N_LEADS}']
    assert abs(default - last_lead) > 1e-6


def test_clipping_is_optional():
    _, truth, forecast, xr = _synthetic()
    clipped = calculate_lead_time_metrics(xr, 'qmax', ['NSE'], clip_to_zero=True)
    raw = calculate_lead_time_metrics(xr, 'qmax', ['NSE'])
    _, obs, sim = _hand_series(truth, forecast, clip=False)
    # Every lead holds negative simulations (the inserted -4 at lead 1, noise at the others): without clipping they
    # stay in the metric, with clipping they count as 0, closer to the non-negative observations.
    assert (sim < 0).any(axis=0).all()
    for lead in range(N_LEADS):
        assert raw[f'NSE_lead{lead + 1}'] == pytest.approx(_nse(obs, sim[:, lead]), rel=1e-9)
        assert raw[f'NSE_lead{lead + 1}'] < clipped[f'NSE_lead{lead + 1}']


def test_all_nan_and_short_series_give_nan_keys():
    _, _, _, xr = _synthetic()
    names = lead_time_metric_names(['NSE', 'KGE'], N_LEADS)
    empty = xr.copy(deep=True)
    empty['qmax_obs'][:] = np.nan
    values = calculate_lead_time_metrics(empty, 'qmax', ['NSE', 'KGE'])
    assert list(values) == names and all(np.isnan(value) for value in values.values())
    # Fewer samples than leads => no day is forecast at every lead.
    values = calculate_lead_time_metrics(xr.isel(date=slice(0, N_LEADS - 1)), 'qmax', ['NSE', 'KGE'])
    assert list(values) == names and all(np.isnan(value) for value in values.values())
    # Exactly as many samples as leads => one common day; the helper returns it.
    dates, obs, _ = lead_time_series(xr.isel(date=slice(0, N_LEADS)), 'qmax')
    assert len(dates) == 1 and obs.shape == (1, N_LEADS)


@pytest.mark.parametrize('layout,message', [('gap', 'gap-free'), ('time_step', 'time_step'),
                                            ('samples', 'point predictions')])
def test_wrong_layouts_are_rejected(layout, message):
    _, _, _, xr = _synthetic()
    bad = {'gap': lambda: xr.isel(date=[0, 1, 3, 4, 5, 6, 7, 8]),
           'time_step': lambda: xr.assign_coords(time_step=np.arange(N_LEADS)),
           'samples': lambda: xr.expand_dims(samples=2)}[layout]()
    with pytest.raises(ValueError, match=message):
        calculate_lead_time_metrics(bad, 'qmax', ['NSE'])


def test_misaligned_observations_are_detected():
    _, _, _, xr = _synthetic()
    broken = xr.copy(deep=True)
    broken['qmax_obs'][:, 0] = np.roll(broken['qmax_obs'].values[:, 0], 1)
    with pytest.raises(RuntimeError, match='Observations differ'):
        calculate_lead_time_metrics(broken, 'qmax', ['NSE'])


def test_check_lead_time_config():
    def config(**extra):
        return types.SimpleNamespace(**dict(dict(use_frequencies=['1D'], predict_last_n=N_LEADS, metrics=['NSE']),
                                            **extra))

    check_lead_time_config(config())
    check_lead_time_config(config(use_frequencies=[]))
    for extra, message in ((dict(use_frequencies=['1D', '1h']), 'single frequency'),
                           (dict(predict_last_n=1), 'predict_last_n'),
                           (dict(predict_last_n={'1D': N_LEADS}), 'predict_last_n'),
                           (dict(metrics=[]), 'requires metrics')):
        with pytest.raises(ValueError, match=message):
            check_lead_time_config(config(**extra))


def test_metric_names():
    assert lead_time_metric_names(['NSE', 'KGE'], 2) == ['NSE_lead1', 'NSE_lead2', 'NSE_lead_mean',
                                                         'KGE_lead1', 'KGE_lead2', 'KGE_lead_mean']
    assert lead_time_metric_names([], 7) == []
