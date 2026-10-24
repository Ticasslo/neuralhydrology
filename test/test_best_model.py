"""Test that best_model.pt holds the epoch with the best validation metric, independent of early stopping min_delta"""
import math
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest
import torch

from neuralhydrology.training.basetrainer import BaseTrainer
from neuralhydrology.training.early_stopping import EarlyStopping


def _make_trainer(run_dir: Path, mode: str) -> BaseTrainer:
    # bypass __init__ (needs data and a full config); set only what the best-model logic uses
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.cfg = SimpleNamespace(run_dir=run_dir, early_stopping_metric='NSE', early_stopping_mode=mode)
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    trainer.experiment_logger = None
    trainer.early_stopping = EarlyStopping(patience=5, min_delta=0.005, mode=mode)
    trainer._best_model_path = None
    trainer._best_metric_value = float('inf') if mode == 'min' else float('-inf')
    return trainer


def _run(trainer: BaseTrainer, values: List[float]) -> tuple:
    # same order as the epoch loop in BaseTrainer.train_and_validate
    stopped_at = None
    for epoch, value in enumerate(values, start=1):
        with torch.no_grad():
            trainer.model.weight.fill_(epoch)  # tag the weights with the epoch number
        if trainer._is_best_model(value):
            trainer._save_best_model(epoch, value)
        if trainer.early_stopping(value, epoch):
            stopped_at = epoch
            break
    best_epoch = None
    if trainer._best_model_path is not None:
        state = torch.load(trainer._best_model_path, weights_only=True)
        best_epoch = int(state['weight'].item())
    return best_epoch, stopped_at


@pytest.mark.parametrize('mode, values, best_epoch, stopped_at', [
    # improvements below min_delta after epoch 3 must not overwrite the best model with a worse epoch
    ('max', [0.60, 0.65, 0.680, 0.684, 0.682, 0.683, 0.681, 0.6835], 4, 8),
    ('min', [1.0, 0.95, 0.92, 0.916, 0.918, 0.917, 0.919, 0.9165], 4, 8),
    # ties keep the earlier epoch
    ('max', [0.60, 0.65, 0.65, 0.64], 2, None),
    # a NaN metric never becomes the best model and counts as no improvement
    ('max', [0.60, math.nan, 0.62, math.nan, math.nan, math.nan, math.nan, math.nan], 3, 8),
])
def test_best_model_is_best_epoch(tmp_path: Path, mode: str, values: List[float], best_epoch: int, stopped_at: int):
    trainer = _make_trainer(tmp_path, mode)
    assert _run(trainer, values) == (best_epoch, stopped_at)
    assert trainer._best_metric_value == values[best_epoch - 1]
