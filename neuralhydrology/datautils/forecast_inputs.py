# Shared by the regular dataset and the vectorized batch loader.
import torch


def prepare_forecast_inputs(data, cfg):
    if cfg.nan_handling_method == 'masked_mean' and cfg.timestep_counter:
        for phase in ('hindcast', 'forecast'):
            key, counter = f'x_d_{phase}', f'{phase}_counter'
            if key not in data or counter not in data.get('x_d', {}):
                raise ValueError('Masked-mean counters require single-frequency hindcast/forecast inputs.')
            data[key][counter] = data['x_d'][counter].clone()
    if cfg.forecast_input_mode == 'first_day':
        if (cfg.nan_handling_method != 'masked_mean' or not cfg.timestep_counter
                or cfg.forecast_overlap or not cfg.hindcast_inputs_flattened):
            raise ValueError('first_day requires masked_mean, timestep_counter and zero-overlap forecast inputs.')
        features = cfg.forecast_inputs_flattened
        if not features or any(not name.startswith('ECMWF_') for name in features):
            raise ValueError('first_day permits only ECMWF HRES forecast features.')
        data['x_d_forecast'] = dict(data['x_d_forecast'])
        for feature in features:
            # Both [time, feature] samples and [batch, time, feature] batches.
            values = data['x_d_forecast'][feature].clone()
            values[..., 1:, :] = torch.nan
            data['x_d_forecast'][feature] = values
    return data
