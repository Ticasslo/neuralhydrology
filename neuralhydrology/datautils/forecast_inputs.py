# Shared by the regular dataset and the vectorized batch loader.
import re
from functools import lru_cache

import torch

GEFS_FEATURE = re.compile(r'GEFS_(.+)_lead(\d+)$')


@lru_cache(maxsize=None)
def gefs_lead(name):
    # Lead time (1-based) of a GEFS forecast column GEFS_<variable>_lead<k>, None for any other column.
    match = GEFS_FEATURE.match(name)
    return int(match[2]) if match else None


def check_gefs_groups(cfg):
    # gefs: one HRES group, then one group per lead time 1..forecast_seq_length with the same GEFS variables each.
    groups = cfg.forecast_inputs
    if not groups or not isinstance(groups[0], list):
        raise ValueError('gefs requires the ECMWF HRES group first in the forecast inputs.')
    _check_gefs_groups(tuple(map(tuple, groups)), cfg.forecast_seq_length)


@lru_cache(maxsize=None)
def _check_gefs_groups(groups, horizon):
    # Cached: runs once per configuration, not once per sample.
    if (not groups[0] or any(not name.startswith('ECMWF_') for name in groups[0])):
        raise ValueError('gefs requires the ECMWF HRES group first in the forecast inputs.')
    leads = [{gefs_lead(name) for name in group} for group in groups[1:]]
    variables = [[GEFS_FEATURE.match(name)[1] if gefs_lead(name) else None for name in group] for group in groups[1:]]
    if (leads != [{k} for k in range(1, horizon + 1)] or not variables[0] or None in variables[0]
            or len(set(variables[0])) != len(variables[0]) or any(v != variables[0] for v in variables)):
        raise ValueError(f'gefs requires one GEFS group per lead time 1..{horizon} with the same variables.')


def prepare_forecast_inputs(data, cfg):
    if cfg.nan_handling_method == 'masked_mean' and cfg.timestep_counter:
        for phase in ('hindcast', 'forecast'):
            key, counter = f'x_d_{phase}', f'{phase}_counter'
            if key not in data or counter not in data.get('x_d', {}):
                raise ValueError('Masked-mean counters require single-frequency hindcast/forecast inputs.')
            data[key][counter] = data['x_d'][counter].clone()
    mode = cfg.forecast_input_mode
    if mode in ('first_day', 'none', 'gefs'):
        if (cfg.nan_handling_method != 'masked_mean' or not cfg.timestep_counter
                or cfg.forecast_overlap or not cfg.hindcast_inputs_flattened):
            raise ValueError(f'{mode} requires masked_mean, timestep_counter and zero-overlap forecast inputs.')
        features = cfg.forecast_inputs_flattened
        if mode == 'gefs':
            check_gefs_groups(cfg)
        elif not features or any(not name.startswith('ECMWF_') for name in features):
            raise ValueError(f'{mode} permits only ECMWF HRES forecast features.')
        data['x_d_forecast'] = dict(data['x_d_forecast'])
        for feature in features:
            # Both [time, feature] samples and [batch, time, feature] batches.
            values = data['x_d_forecast'][feature].clone()
            lead = gefs_lead(feature)
            if lead is None:
                # first_day and gefs keep the HRES issued for the first forecast day, none hides every forecast day.
                values[..., 1 if mode in ('first_day', 'gefs') else 0:, :] = torch.nan
            else:
                # Forecast day k keeps only the GEFS run of lead k, i.e. the run issued on the first forecast day.
                if values.shape[-2] != cfg.forecast_seq_length:
                    raise ValueError('gefs requires forecast inputs of forecast_seq_length steps.')
                values[..., :lead - 1, :] = torch.nan
                values[..., lead:, :] = torch.nan
            data['x_d_forecast'][feature] = values
    return data
