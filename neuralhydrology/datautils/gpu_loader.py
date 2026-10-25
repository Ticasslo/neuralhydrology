"""Batch loader that keeps a dataset in device memory and builds each batch with one index operation (``data_on_gpu``)."""
import math
from typing import Dict, Iterator, List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from neuralhydrology.datasetzoo.basedataset import BaseDataset

Batch = Dict[str, Union[torch.Tensor, np.ndarray, Dict[str, torch.Tensor]]]


class GpuBatchLoader:
    """Drop-in replacement for ``DataLoader(ds, batch_size, shuffle, collate_fn=ds.collate_fn)``.

    All basins of the dataset are stacked into a few tensors on ``device`` once. Each batch is then gathered with one
    index operation per block, so the per-sample ``__getitem__`` and the per-variable ``collate_fn`` are skipped. Keys,
    shapes and values are identical to ``collate_fn`` (except that training batches carry no ``date``). The sample order
    is drawn by iterating a ``DataLoader`` over the sample indices with the same ``batch_size`` and ``shuffle``, so the
    global random number generator is used exactly as by the ``DataLoader`` it replaces.

    Parameters
    ----------
    ds : BaseDataset
        Dataset with a single frequency.
    batch_size : int
        Number of samples per batch.
    device : torch.device
        Device that holds the data and the batches.
    shuffle : bool
        Shuffle the samples in every iteration (training) or keep the dataset order (evaluation).
    with_dates : bool, optional
        Add the ``date`` array (numpy, as ``collate_fn``) to each batch; needed by the tester.
    """

    def __init__(self, ds: BaseDataset, batch_size: int, device: torch.device, shuffle: bool, with_dates: bool = False):
        _check_supported(ds)
        cfg = ds.cfg
        self.cfg = cfg
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.shuffle = shuffle
        self.with_dates = with_dates
        self.n_samples = len(ds)
        freq = ds.frequencies[0]
        seq_len = ds.seq_len[0]

        # window offsets relative to the last time step of a sample, same slices as BaseDataset.__getitem__
        fsl = cfg.forecast_seq_length
        if cfg.hindcast_inputs_flattened:
            hindcast_end = -fsl + 1 + (cfg.forecast_overlap or 0)
            self._off = {"x_d": None, "x_d_hindcast": torch.arange(1 - seq_len, hindcast_end),
                         "x_d_forecast": torch.arange(1 - fsl, 1)}
        else:
            self._off = {"x_d": torch.arange(1 - seq_len, 1), "x_d_hindcast": None, "x_d_forecast": None}
        off_y = torch.arange(1 - seq_len, 1)

        # variable names per block, in the order of the dataset's dictionaries (as collate_fn returns them)
        first = next(iter(ds._x_d.values()))[freq]
        self._keys = {"x_d": [k for k in first] if not cfg.hindcast_inputs_flattened else [],
                      "x_d_hindcast": [k for k in first if k in cfg.hindcast_inputs_flattened],
                      "x_d_forecast": [k for k in first if k in cfg.forecast_inputs_flattened]}

        # stack all basins along the time axis; a sample is (basin number, row of its last time step)
        blocks = {name: [] for name in self._keys}
        y, dates, attributes, stds = [], [], [], []
        position, offset = {}, 0
        for basin, data in ds._x_d.items():
            position[basin] = (len(y), offset)
            for name, keys in self._keys.items():
                if keys:
                    blocks[name].append(torch.cat([data[freq][k] for k in keys], dim=1))
            y.append(ds._y[basin][freq])
            dates.append(ds._dates[basin][freq])
            if ds._attributes:
                attributes.append(ds._attributes[basin])
            if ds._per_basin_target_stds:
                stds.append(ds._per_basin_target_stds[basin])
            offset += len(y[-1])
        basin_of, row_of = np.empty(self.n_samples, dtype=np.int64), np.empty(self.n_samples, dtype=np.int64)
        for i in range(self.n_samples):
            basin, indices = ds.lookup_table[i]
            basin_of[i] = position[basin][0]
            row_of[i] = position[basin][1] + indices[0]

        to = dict(device=self.device)
        self._blocks = {name: torch.cat(b).to(**to) for name, b in blocks.items() if b}
        self._y = torch.cat(y).to(**to)
        self._attributes = torch.stack(attributes).to(**to) if attributes else None
        self._stds = torch.stack(stds).to(**to) if stds else None
        self._off = {name: o.to(**to) if o is not None else None for name, o in self._off.items()}
        self._off_y = off_y.to(**to)
        self._basin_of = torch.from_numpy(basin_of).to(**to)
        self._row_of = torch.from_numpy(row_of).to(**to)
        # dates stay in host memory (numpy datetime64), gathered only for evaluation batches
        self._dates = np.concatenate(dates) if with_dates else None
        self._row_of_np = row_of if with_dates else None
        self._off_y_np = off_y.numpy()

        self._counters = None
        if cfg.timestep_counter:
            self._counters = {"hindcast_counter": ds.hindcast_counter.to(**to),
                              "forecast_counter": ds.forecast_counter.to(**to)}
        self._nan_streaks = ds.is_train and bool(cfg.nan_step_probability or cfg.nan_sequence_probability)

    def __len__(self) -> int:
        return math.ceil(self.n_samples / self.batch_size)

    def __iter__(self) -> Iterator[Batch]:
        # draw the order through a DataLoader so that the random number generator is used as in the replaced loader
        order = torch.cat(list(DataLoader(range(self.n_samples), batch_size=self.batch_size, shuffle=self.shuffle)))
        order_dev = order.to(self.device)
        for start in range(0, self.n_samples, self.batch_size):
            yield self._batch(order_dev[start:start + self.batch_size], order[start:start + self.batch_size])

    def _batch(self, idx: torch.Tensor, idx_host: torch.Tensor) -> Batch:
        basin, row = self._basin_of[idx], self._row_of[idx]
        batch = {}
        for name, keys in self._keys.items():
            batch[name] = {}
            if keys:
                block = self._blocks[name][row[:, None] + self._off[name]]
                if self._nan_streaks:
                    block = self._add_nan_streaks(block, keys, name)
                batch[name] = {k: block[..., j:j + 1] for j, k in enumerate(keys)}
        if self._counters is not None:
            # a new tensor per batch (as collate_fn), so in-place changes never reach the stored counters
            n = len(idx)
            batch["x_d"].update({k: v.unsqueeze(0).repeat(n, 1, 1) for k, v in self._counters.items()})
        batch["y"] = self._y[row[:, None] + self._off_y]
        if self.with_dates:
            rows = self._row_of_np[idx_host.numpy()]
            batch["date"] = self._dates[rows[:, None] + self._off_y_np]
        if self._attributes is not None:
            batch["x_s"] = self._attributes[basin]
        if self._stds is not None:
            batch["per_basin_target_stds"] = self._stds[basin]
        return batch

    def _add_nan_streaks(self, block: torch.Tensor, keys: List[str], name: str) -> torch.Tensor:
        # same distribution as BaseDataset._add_nan_streaks (per sample and feature group), drawn on the device
        groups = {"x_d": self.cfg.dynamic_inputs, "x_d_hindcast": self.cfg.hindcast_inputs,
                  "x_d_forecast": self.cfg.forecast_inputs}[name]
        if not groups or not isinstance(groups[0], list):
            raise ValueError('For dropout streaks, dynamic_inputs must be a list of lists.')
        n, length = block.shape[0], block.shape[1]
        drop_sequences = torch.rand(n, len(groups), device=block.device) < self.cfg.nan_sequence_probability
        all_dropped = drop_sequences.all(dim=1)
        if all_dropped.any():
            # don't allow all sequences of a sample to be dropped out
            keep = torch.randint(len(groups), (n,), device=block.device)
            drop_sequences[all_dropped, keep[all_dropped]] = False
        drop_steps = torch.rand(n, len(groups), length, device=block.device) < self.cfg.nan_step_probability
        drop_masks = drop_sequences[:, :, None] | drop_steps
        column = {k: j for j, k in enumerate(keys)}
        for i, group in enumerate(groups):
            for feature in group:
                j = column[feature]
                block[:, :, j] = torch.where(drop_masks[:, i], torch.nan, block[:, :, j])
        return block


def _check_supported(ds: BaseDataset):
    # fail loudly on dataset features whose samples this loader does not reproduce
    cfg = ds.cfg
    unsupported = {
        "more than one frequency (use_frequencies)": len(ds.frequencies) != 1,
        "autoregressive_inputs": bool(cfg.autoregressive_inputs),
        "evolving_attributes": bool(cfg.evolving_attributes) or bool(ds._x_s),
        "use_basin_id_encoding": bool(ds.id_to_int),
        "mass_inputs": bool(cfg.mass_inputs),
        "hindcast_inputs without forecast_seq_length": bool(cfg.hindcast_inputs_flattened) and not cfg.forecast_seq_length,
    }
    found = [name for name, flag in unsupported.items() if flag]
    if found:
        raise NotImplementedError(f"data_on_gpu does not support: {', '.join(found)}. Set data_on_gpu to False.")
    if len(ds) == 0:
        raise ValueError("Dataset contains no samples.")
