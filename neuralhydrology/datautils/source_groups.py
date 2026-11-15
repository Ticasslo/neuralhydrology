import copy

from neuralhydrology.utils.config import Config


def lamah_source_groups(variables):
    # Stable source order; keep the original variable order within each source.
    order = ('ERA5L', 'EOBS', 'MSWEP', 'GLEAM', 'Q', 'ECMWF')
    groups = {source: [] for source in order}
    if len(variables) != len(set(variables)):
        raise ValueError('Duplicate variables in a model phase.')
    for variable in variables:
        source = 'Q' if variable in ('qmean_mmd', 'qmax_mmd') else variable.split('_', 1)[0]
        if source not in groups:
            raise ValueError(f'Unknown LamaH source: {variable}')
        groups[source].append(variable)
    return [groups[source] for source in order if groups[source]]


def configure_lamah_b2(cfg: Config, *, preserve_masking: bool = False) -> Config:
    # B2 adds only grouping, masked mean and its sample filter to the B1b configuration.
    # C can reuse a selected B3 configuration by explicitly retaining its training masks.
    if (cfg.model not in ('sequential_forecast_lstm', 'sequential_forecast') or cfg.target_variables != ['qmax']
            or cfg.forecast_seq_length != 1 or cfg.forecast_overlap
            or cfg.seq_length != 365 or cfg.predict_last_n != 1
            or cfg.epochs != 30 or cfg.early_stopping_patience != 30):
        raise ValueError('B2 requires the one-day, 365-step, 30-epoch B1b protocol.')
    hindcast = lamah_source_groups(cfg.hindcast_inputs_flattened)
    forecast = lamah_source_groups(cfg.forecast_inputs_flattened)
    expected = (21, 7, 1, 2, 2)
    if (tuple(map(len, hindcast)) != expected or len(forecast) != 1 or len(forecast[0]) != 5
            or not all(name.startswith('ECMWF_') for name in forecast[0])
            or set(hindcast[-1]) != {'qmean_mmd', 'qmax_mmd'}):
        raise ValueError('B1b inputs must contain ERA5L/EOBS/MSWEP/GLEAM/Q and five ECMWF variables.')
    flattened = [variable for group in hindcast + forecast for variable in group]
    if set(flattened) != set(cfg.dynamic_inputs_flattened):
        raise ValueError('B1b dynamic inputs differ from its hindcast/forecast phases.')
    result = copy.deepcopy(cfg)
    result.update_config(dict(dynamic_inputs=hindcast + forecast, hindcast_inputs=hindcast,
                              forecast_inputs=forecast, nan_handling_method='masked_mean',
                              nan_step_probability=cfg.nan_step_probability if preserve_masking else 0.,
                              nan_sequence_probability=cfg.nan_sequence_probability if preserve_masking else 0.,
                              timestep_counter=False, forecast_input_mode='all'))
    return result


# Reanalysis counterparts of ECMWF_t2m, ECMWF_d2m, ECMWF_ssrd, ECMWF_tp, ECMWF_e, in that order.
LAMAH_PERFECT_FORECAST = ('ERA5L_2m_temp_mean', 'ERA5L_2m_dp_temp_mean', 'EOBS_qq', 'MSWEP_RR', 'GLEAM_ETA')
# Forecast-phase input of each H variant => forecast_input_mode.
LAMAH_H_MODES = dict(past='none', hybrid='first_day', perfect='all')
# GEFSv12 reforecast counterparts of the five HRES variables, as columns GEFS_<variable>_lead<k> (k = 1..horizon):
# the value on day d is the forecast for day d issued on day d - k + 1 (04_GEFS_Download).
LAMAH_GEFS_VARIABLES = ('t2m', 'd2m', 'ssrd', 'tp', 'e')


def configure_lamah_h(cfg: Config, mode: str, *, horizon: int = 7,
                      perfect_inputs=LAMAH_PERFECT_FORECAST, gefs_variables=LAMAH_GEFS_VARIABLES) -> Config:
    # H extends a grouped one-day configuration (configure_lamah_b2) to horizon days: the hindcast keeps its length,
    # the forecast phase grows from 1 to horizon steps and every step is a target.
    # past: HRES hidden on all days; hybrid: HRES on the first day only; perfect: reanalysis on all days;
    # gefs: HRES on the first day, plus one GEFS group per lead time, of which forecast day k sees lead k only.
    if mode not in LAMAH_H_MODES and mode != 'gefs':
        raise ValueError(f'H mode must be one of {tuple(LAMAH_H_MODES) + ("gefs",)}.')
    if not isinstance(horizon, int) or horizon < 2:
        raise ValueError('H requires a horizon of at least two days.')
    hindcast, forecast = cfg.hindcast_inputs, cfg.forecast_inputs
    if (cfg.model not in ('sequential_forecast_lstm', 'sequential_forecast') or cfg.target_variables != ['qmax']
            or cfg.forecast_seq_length != 1 or cfg.forecast_overlap or cfg.predict_last_n != 1
            or cfg.nan_handling_method != 'masked_mean' or cfg.forecast_input_mode != 'all'
            or not hindcast or not isinstance(hindcast[0], list)
            or len(forecast) != 1 or not isinstance(forecast[0], list)
            or not all(name.startswith('ECMWF_') for name in forecast[0])):
        raise ValueError('H requires a grouped one-day masked-mean configuration with one ECMWF forecast group.')
    update = dict(seq_length=cfg.seq_length + horizon - 1, forecast_seq_length=horizon, predict_last_n=horizon,
                  timestep_counter=True, forecast_input_mode=LAMAH_H_MODES.get(mode, mode))
    if mode == 'perfect':
        perfect = list(perfect_inputs)
        if len(perfect) != len(forecast[0]) or len(set(perfect)) != len(perfect):
            raise ValueError('perfect needs one distinct reanalysis variable per ECMWF forecast variable.')
        missing = [name for name in perfect if name not in cfg.hindcast_inputs_flattened]
        if missing:
            raise ValueError(f'perfect inputs are not hindcast inputs: {missing}')
        # The same columns feed both phases, so the loader must list them once.
        update.update(forecast_inputs=[perfect], dynamic_inputs=copy.deepcopy(hindcast))
    if mode == 'gefs':
        variables = list(gefs_variables)
        if not variables or len(set(variables)) != len(variables):
            raise ValueError('gefs needs distinct GEFS variables.')
        lead_groups = [[f'GEFS_{name}_lead{k}' for name in variables] for k in range(1, horizon + 1)]
        update.update(forecast_inputs=copy.deepcopy(forecast) + lead_groups,
                      dynamic_inputs=copy.deepcopy(hindcast) + copy.deepcopy(forecast) + lead_groups)
    result = copy.deepcopy(cfg)
    result.update_config(update)
    return result
