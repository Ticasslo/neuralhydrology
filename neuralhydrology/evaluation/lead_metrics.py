# Per-lead-time metrics of multi-step forecasts (single frequency, predict_last_n > 1). No torch dependency.
import numpy as np
import pandas as pd
import xarray

from neuralhydrology.evaluation.metrics import calculate_metrics
from neuralhydrology.utils.errors import AllNaNError


def check_lead_time_config(cfg):
    # Without these the lead keys would be missing and model selection would fall back to the loss.
    if len(cfg.use_frequencies) > 1:
        raise ValueError('lead_time_metrics supports a single frequency only.')
    if not isinstance(cfg.predict_last_n, int) or cfg.predict_last_n < 2:
        raise ValueError('lead_time_metrics requires predict_last_n >= 2.')
    if not cfg.metrics:
        raise ValueError('lead_time_metrics requires metrics.')


def lead_time_metric_names(metrics, n_leads):
    # Column order of one target: <metric>_lead1 ... <metric>_lead<n>, <metric>_lead_mean.
    names = []
    for metric in metrics:
        names += [f'{metric}_lead{lead}' for lead in range(1, n_leads + 1)] + [f'{metric}_lead_mean']
    return names


def lead_time_series(xr, target_variable, resolution='1D', clip_to_zero=False):
    # xr is the dataset of one basin built by the tester: dims (date, time_step), date = last day of each sample on
    # a gap-free axis, time_step = -(n - 1) ... 0. Lead k (1 = first forecast day) is column k - 1 and the value of
    # the sample ending on date[j] belongs to target day date[j - (n - k)].
    # Returns the target days common to all leads and [day, lead] arrays in which a day is NaN for every lead as
    # soon as one lead lacks its observation or simulation => all leads are scored on the same days.
    obs, sim = xr[f'{target_variable}_obs'], xr[f'{target_variable}_sim']
    if set(sim.dims) != {'date', 'time_step'}:
        raise ValueError('lead_time_metrics supports point predictions with dims (date, time_step) only.')
    obs, sim = obs.transpose('date', 'time_step').values, sim.transpose('date', 'time_step').values
    n_dates, n_leads = obs.shape
    if not np.array_equal(xr['time_step'].values, np.arange(-(n_leads - 1), 1)):
        raise ValueError('lead_time_metrics expects time_step = -(n - 1) ... 0.')
    dates = pd.DatetimeIndex(xr['date'].values)
    if n_dates > 1 and not (np.diff(dates.values) == pd.Timedelta(resolution).to_timedelta64()).all():
        raise ValueError('lead_time_metrics expects a gap-free date axis.')
    n_days = n_dates - n_leads + 1
    if n_days < 1:
        return dates[:0], np.empty((0, n_leads)), np.empty((0, n_leads))
    # Column i comes from the samples that end n - 1 - i days after the target day.
    obs = np.stack([obs[n_leads - 1 - i:n_dates - i, i] for i in range(n_leads)], axis=1).astype(np.float64)
    sim = np.stack([sim[n_leads - 1 - i:n_dates - i, i] for i in range(n_leads)], axis=1).astype(np.float64)
    valid = ~(np.isnan(obs).any(axis=1) | np.isnan(sim).any(axis=1))
    # Every column of obs holds the observation of the same day; a mismatch means the alignment above is wrong.
    if not np.allclose(obs[valid], obs[valid][:, -1:], rtol=1e-5, atol=1e-8):
        raise RuntimeError('Observations differ between lead times of the same target day.')
    if clip_to_zero:
        sim = np.where(sim < 0, 0, sim)
    obs, sim = np.where(valid[:, None], obs, np.nan), np.where(valid[:, None], sim, np.nan)
    return dates[:n_days], obs, sim


def calculate_lead_time_metrics(xr, target_variable, metrics, resolution='1D', clip_to_zero=False):
    # <metric>_lead<k> for every lead and <metric>_lead_mean, the plain mean over leads (NaN if one lead is NaN).
    dates, obs, sim = lead_time_series(xr, target_variable, resolution=resolution, clip_to_zero=clip_to_zero)
    n_leads = obs.shape[1]
    per_lead = []
    for i in range(n_leads):
        obs_lead = xarray.DataArray(obs[:, i], coords={'date': dates}, dims=['date'])
        sim_lead = xarray.DataArray(sim[:, i], coords={'date': dates}, dims=['date'])
        try:
            per_lead.append(calculate_metrics(obs_lead, sim_lead, metrics=metrics, resolution=resolution))
        except AllNaNError:
            per_lead.append(None)
    names = next((list(values) for values in per_lead if values is not None), list(metrics))
    result = {}
    for name in names:
        by_lead = [np.nan if values is None else values[name] for values in per_lead]
        result.update({f'{name}_lead{i + 1}': value for i, value in enumerate(by_lead)})
        result[f'{name}_lead_mean'] = float(np.mean(by_lead))
    return result
