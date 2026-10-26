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


def configure_lamah_b2(cfg: Config) -> Config:
    # B2 adds only grouping, masked mean and its sample filter to the B1b configuration.
    if (cfg.model != 'sequential_forecast_lstm' or cfg.target_variables != ['qmax']
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
                              nan_step_probability=0., nan_sequence_probability=0.,
                              timestep_counter=False, forecast_input_mode='all'))
    return result
