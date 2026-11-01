import copy

import pytest
import torch

from neuralhydrology.datautils.source_groups import configure_lamah_b2
from neuralhydrology.modelzoo import get_model
from test.test_b2_source_inputs import b2_config


def _source_config(b2_config, **extra):
    hindcast = [f'{source}_{i}' for source, count in [('ERA5L', 21), ('EOBS', 7), ('MSWEP', 1), ('GLEAM', 2)]
                for i in range(count)] + ['qmean_mmd', 'qmax_mmd']
    forecast = [f'ECMWF_{i}' for i in range(5)]
    values = dict(model='sequential_forecast', sf_core='lstm', sf_layers=2,
                  sf_transformer_nheads=2, sf_s4d_lr=1e-4, sf_mamba_d_inner=16,
                  learning_rate=5e-4, dynamic_inputs=hindcast + forecast,
                  hindcast_inputs=hindcast, forecast_inputs=forecast,
                  seq_length=365, epochs=30, early_stopping_patience=30,
                  nan_handling_method=None)
    values.update(extra)
    return b2_config(**values)


def _missing_group_batch(cfg):
    generator = torch.Generator().manual_seed(7)
    length = cfg.seq_length - cfg.forecast_seq_length
    data = dict(
        x_d_hindcast={v: torch.randn(2, length, 1, generator=generator) for v in cfg.hindcast_inputs_flattened},
        x_d_forecast={v: torch.randn(2, 1, 1, generator=generator) for v in cfg.forecast_inputs_flattened},
        x_s=torch.randn(2, len(cfg.static_attributes), generator=generator))
    # Sample 0 loses all Q history and the forecast group. Sample 1 loses ERA5L's
    # last five days and every hindcast group at the last step (all-missing fallback).
    for v, values in data['x_d_hindcast'].items():
        if v in ('qmean_mmd', 'qmax_mmd'):
            values[0] = torch.nan
        if v.startswith('ERA5L_'):
            values[1, -5:] = torch.nan
        values[1, -1] = torch.nan
    for values in data['x_d_forecast'].values():
        values[0] = torch.nan
    return data


@pytest.mark.parametrize('core', ['lstm', 'gru', 'transformer', 's4d', 'mamba'])
def test_configures_generic_core_without_other_changes(b2_config, core):
    # Mamba is a config-only check here; constructing its real core requires mamba_ssm/CUDA.
    original = _source_config(b2_config, sf_core=core)
    before = copy.deepcopy(original.as_dict())
    configured = configure_lamah_b2(original)
    assert original.as_dict() == before
    assert configured is not original
    assert configured.model == 'sequential_forecast'
    assert configured.sf_core == core
    assert list(map(len, configured.hindcast_inputs)) == [21, 7, 1, 2, 2]
    assert list(map(len, configured.forecast_inputs)) == [5]
    allowed = {'dynamic_inputs', 'hindcast_inputs', 'forecast_inputs', 'nan_handling_method',
               'nan_step_probability', 'nan_sequence_probability', 'timestep_counter', 'forecast_input_mode'}
    for key, value in before.items():
        if key not in allowed:
            assert configured.as_dict()[key] == value


@pytest.mark.parametrize('model', ['sequential_forecast_lstm', 'sequential_forecast'])
@pytest.mark.parametrize('step_probability,sequence_probability', [(.1, .12), (.05, .05)])
def test_preserves_b3_masks_only_when_requested(b2_config, model, step_probability, sequence_probability):
    original = _source_config(b2_config, model=model, nan_handling_method='masked_mean',
                              nan_step_probability=step_probability,
                              nan_sequence_probability=sequence_probability)
    before = copy.deepcopy(original.as_dict())
    default = configure_lamah_b2(original)
    selected_b3 = configure_lamah_b2(original, preserve_masking=True)
    assert default.nan_step_probability == default.nan_sequence_probability == 0.
    assert (selected_b3.nan_step_probability, selected_b3.nan_sequence_probability) == (
        step_probability, sequence_probability)
    assert original.as_dict() == before


def test_rejects_model_outside_sequential_forecast_family(b2_config):
    with pytest.raises(ValueError, match='one-day'):
        configure_lamah_b2(_source_config(b2_config, model='cudalstm'))


@pytest.mark.parametrize('core', ['lstm', 'gru', 'transformer', 's4d'])
def test_generic_cores_learn_with_missing_source_groups(b2_config, core):
    cfg = configure_lamah_b2(_source_config(b2_config, sf_core=core,
                              nan_step_probability=.1, nan_sequence_probability=.12), preserve_masking=True)
    torch.manual_seed(11)
    model = get_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    prediction = model(_missing_group_batch(cfg))['y_hat']
    assert prediction.shape == (2, 365, 1)
    assert torch.isfinite(prediction).all()
    prediction[:, -1].square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any(not torch.equal(before[name], value) for name, value in model.named_parameters())


def test_grouped_generic_lstm_equals_original_with_missing_groups(b2_config):
    cfg = configure_lamah_b2(_source_config(b2_config, sf_layers=1,
                              nan_step_probability=.1, nan_sequence_probability=.12), preserve_masking=True)
    original_cfg = copy.deepcopy(cfg)
    original_cfg.update_config(dict(model='sequential_forecast_lstm'))
    original = get_model(original_cfg).eval()
    generic = get_model(cfg).eval()
    generic.load_state_dict({name.replace('lstm.', 'core.rnn.', 1): value
                             for name, value in original.state_dict().items()}, strict=True)
    batch = _missing_group_batch(cfg)
    with torch.no_grad():
        reference, current = original(batch)['y_hat'], generic(batch)['y_hat']
    assert torch.isfinite(reference).all()
    assert torch.equal(reference, current)
