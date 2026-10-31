import logging
import pickle
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import neuralhydrology.training.loss as loss
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.datautils.gpu_loader import GpuBatchLoader
from neuralhydrology.datautils.utils import load_basin_file, load_scaler
from neuralhydrology.evaluation import get_tester
from neuralhydrology.evaluation.tester import BaseTester
from neuralhydrology.modelzoo import get_model
from neuralhydrology.training import get_loss_obj, get_optimizer, get_regularization_obj, get_lr_scheduler
from neuralhydrology.training.early_stopping import EarlyStopping
from neuralhydrology.training.logger import Logger
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.logging_utils import setup_logging

LOGGER = logging.getLogger(__name__)


class BaseTrainer(object):
    """Default class to train a model.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        super(BaseTrainer, self).__init__()
        self.cfg = cfg
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.loss_obj = None
        self.experiment_logger = None
        self.loader = None
        self.validator = None
        self.early_stopping = None
        self.noise_sampler_y = None
        self._target_mean = None
        self._target_std = None
        self._scaler = {}
        self._allow_subsequent_nan_losses = cfg.allow_subsequent_nan_losses
        self._disable_pbar = cfg.verbose == 0
        self._max_updates_per_epoch = cfg.max_updates_per_epoch
        self._start_time = time.time()
        self._best_epoch = None

        # load train basin list and add number of basins to the config
        self.basins = load_basin_file(cfg.train_basin_file)
        self.cfg.number_of_basins = len(self.basins)

        # check at which epoch the training starts
        self._epoch = self._get_start_epoch_number()

        self._create_folder_structure()
        setup_logging(str(self.cfg.run_dir / "output.log"))
        LOGGER.info(f"### Folder structure created at {self.cfg.run_dir}")

        if self.cfg.is_continue_training:
            LOGGER.info(f"### Continue training of run stored in {self.cfg.base_run_dir}")

        if self.cfg.is_finetuning:
            LOGGER.info(f"### Start finetuning with pretrained model stored in {self.cfg.base_run_dir}")

        LOGGER.info(f"### Run configurations for {self.cfg.experiment_name}")
        for key, val in self.cfg.as_dict().items():
            LOGGER.info(f"{key}: {val}")

        self._set_random_seeds()
        self._set_device()

    def _get_dataset(self) -> BaseDataset:
        return get_dataset(cfg=self.cfg, period="train", is_train=True, scaler=self._scaler)

    def _get_model(self) -> torch.nn.Module:
        return get_model(cfg=self.cfg)

    def _get_optimizer(self) -> torch.optim.Optimizer:
        return get_optimizer(model=self.model, cfg=self.cfg)

    def _get_lr_scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        return get_lr_scheduler(optimizer=self.optimizer, cfg=self.cfg)

    def _get_loss_obj(self) -> loss.BaseLoss:
        return get_loss_obj(cfg=self.cfg)

    def _set_regularization(self):
        self.loss_obj.set_regularization_terms(get_regularization_obj(cfg=self.cfg))

    def _get_tester(self) -> BaseTester:
        return get_tester(cfg=self.cfg, run_dir=self.cfg.run_dir, period="validation", init_model=False)

    def _get_data_loader(self, ds: BaseDataset) -> torch.utils.data.DataLoader:
        if self.cfg.data_on_gpu:
            # keep the training data on the device and gather each batch there (same batches as the DataLoader)
            return GpuBatchLoader(ds, batch_size=self.cfg.batch_size, device=self.device, shuffle=True)
        return DataLoader(ds,
                          batch_size=self.cfg.batch_size,
                          shuffle=True,
                          num_workers=self.cfg.num_workers,
                          collate_fn=ds.collate_fn)

    def _freeze_model_parts(self):
        # freeze all model weights
        for param in self.model.parameters():
            param.requires_grad = False

        unresolved_modules = []

        # unfreeze parameters specified in config as tuneable parameters
        if isinstance(self.cfg.finetune_modules, list):
            for module_part in self.cfg.finetune_modules:
                if module_part in self.model.module_parts:
                    module = getattr(self.model, module_part)
                    for param in module.parameters():
                        param.requires_grad = True
                else:
                    unresolved_modules.append(module_part)
        else:
            # if it was no list, it has to be a dictionary
            for module_group, module_parts in self.cfg.finetune_modules.items():
                if module_group in self.model.module_parts:
                    if isinstance(module_parts, str):
                        module_parts = [module_parts]
                    for module_part in module_parts:
                        module = getattr(self.model, module_group)[module_part]
                        for param in module.parameters():
                            param.requires_grad = True
                else:
                    unresolved_modules.append(module_group)
        if unresolved_modules:
            LOGGER.warning(f"Could not resolve the following module parts for finetuning: {unresolved_modules}")

    def initialize_training(self):
        """Initialize the training class.

        This method will load the model, initialize loss, regularization, optimizer, dataset and dataloader,
        tensorboard logging, and Tester class.
        If called in a ``continue_training`` context, this model will also restore the model and optimizer state.
        """
        if self.cfg.is_finetuning:
            # Load scaler from pre-trained model.
            self._scaler = load_scaler(self.cfg.base_run_dir)

        # Initialize dataset before the model is loaded.
        ds = self._get_dataset()
        if len(ds) == 0:
            raise ValueError("Dataset contains no samples.")
        self.loader = self._get_data_loader(ds=ds)

        self.model = self._get_model().to(self.device)
        if self.cfg.checkpoint_path is not None:
            LOGGER.info(f"Starting training from Checkpoint {self.cfg.checkpoint_path}")
            self.model.load_state_dict(torch.load(str(self.cfg.checkpoint_path), map_location=self.device))
        elif self.cfg.checkpoint_path is None and self.cfg.is_finetuning:
            # the default for finetuning is the last model state
            checkpoint_path = [x for x in sorted(list(self.cfg.base_run_dir.glob('model_epoch*.pt')))][-1]
            LOGGER.info(f"Starting training from checkpoint {checkpoint_path}")
            self.model.load_state_dict(torch.load(str(checkpoint_path), map_location=self.device))

        # Freeze model parts from pre-trained model.
        if self.cfg.is_finetuning:
            self._freeze_model_parts()

        self.optimizer = self._get_optimizer()
        self.lr_scheduler = self._get_lr_scheduler()
        self.loss_obj = self._get_loss_obj().to(self.device)

        # Add possible regularization terms to the loss function.
        self._set_regularization()

        # restore optimizer and model state if training is continued
        if self.cfg.is_continue_training:
            self._restore_training_state()

        config_file_name = 'config.yml'
        if self._resumes_training_state():
            # keep the configuration of the interrupted run
            config_file_name = f'config_continue_from_epoch{self._epoch:03d}.yml'
        self.experiment_logger = Logger(cfg=self.cfg, config_file_name=config_file_name)
        if self.cfg.log_tensorboard:
            self.experiment_logger.start_tb()

        if self.cfg.is_continue_training:
            # set epoch and iteration step counter to continue from the selected checkpoint
            self.experiment_logger.epoch = self._epoch
            self.experiment_logger.update = len(self.loader) * self._epoch

        if self.cfg.validate_every is not None:
            if self.cfg.validate_n_random_basins < 1:
                warn_msg = [
                    f"Validation set to validate every {self.cfg.validate_every} epoch(s), but ",
                    "'validate_n_random_basins' not set or set to zero. Will validate on the entire validation set."
                ]
                LOGGER.warning("".join(warn_msg))
                self.cfg.validate_n_random_basins = self.cfg.number_of_basins
            self.validator = self._get_tester()

        if self.cfg.target_noise_std is not None:
            self.noise_sampler_y = torch.distributions.Normal(loc=0, scale=self.cfg.target_noise_std)
            self._target_mean = torch.from_numpy(
                ds.scaler["xarray_feature_center"][self.cfg.target_variables].to_array().values).to(self.device)
            self._target_std = torch.from_numpy(
                ds.scaler["xarray_feature_scale"][self.cfg.target_variables].to_array().values).to(self.device)

        # Initialize early stopping if configured
        if self.cfg.early_stopping_patience is not None:
            self.early_stopping = EarlyStopping(
                patience=self.cfg.early_stopping_patience,
                min_delta=self.cfg.early_stopping_min_delta,
                mode=self.cfg.early_stopping_mode
            )
            self._best_model_path = None
            # best value seen so far, tracked separately from early stopping, which only updates its
            # best value on improvements larger than min_delta
            self._best_metric_value = float('inf') if self.early_stopping.mode == 'min' else float('-inf')
            LOGGER.info(f"Early stopping enabled: patience={self.cfg.early_stopping_patience}, "
                       f"min_delta={self.cfg.early_stopping_min_delta}, "
                       f"mode={self.cfg.early_stopping_mode}, "
                       f"monitoring={self.cfg.early_stopping_metric}")

        # restore the rest of the training state last, right before the next training step, so that building the
        # model and the data does not advance the restored random number generators
        if self._resumes_training_state():
            self._restore_resume_state()

    def train_and_validate(self):
        """Train and validate the model.

        Train the model for the number of epochs specified in the run configuration, and perform validation after every
        ``validate_every`` epochs. Model and optimizer state are saved after every ``save_weights_every`` epochs.
        """
        # a resumed run stops at 'epochs', a plain continued run trains 'epochs' more epochs
        last_epoch = self.cfg.epochs if self._resumes_training_state() else self._epoch + self.cfg.epochs
        longest_epoch = 0.0
        for epoch in range(self._epoch + 1, last_epoch + 1):
            epoch_start = time.time()
            stop_early = False
            self._train_epoch(epoch=epoch)
            
            # Step the learning rate scheduler if one is configured
            if self.lr_scheduler is not None:
                # For ReduceLROnPlateau, we need to pass the validation loss
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # We'll step this scheduler after validation, not here
                    pass
                else:
                    self.lr_scheduler.step()
                
                # Log current learning rate to tensorboard/wandb
                current_lr = self.optimizer.param_groups[0]['lr']
                if self.experiment_logger is not None:
                    self.experiment_logger.log_lr(current_lr)

            avg_losses = self.experiment_logger.summarise()
            loss_str = ", ".join(f"{k}: {v:.5f}" for k, v in avg_losses.items())
            LOGGER.info(f"Epoch {epoch} average loss: {loss_str}")

            if epoch % self.cfg.save_weights_every == 0:
                self._save_weights_and_optimizer(epoch)

            if (self.validator is not None) and (epoch % self.cfg.validate_every == 0):
                self.validator.evaluate(epoch=epoch,
                                        save_results=self.cfg.save_validation_results,
                                        save_all_output=self.cfg.save_all_output,
                                        metrics=self.cfg.metrics,
                                        model=self.model,
                                        experiment_logger=self.experiment_logger.valid())

                valid_metrics = self.experiment_logger.summarise()
                print_msg = f"Epoch {epoch} average validation loss: {valid_metrics['avg_total_loss']:.5f}"
                if self.cfg.metrics:
                    print_msg += f" -- Median validation metrics: "
                    print_msg += ", ".join(f"{k}: {v:.5f}" for k, v in valid_metrics.items() if k != 'avg_total_loss')
                    LOGGER.info(print_msg)
                
                # Step ReduceLROnPlateau scheduler after validation
                if (self.lr_scheduler is not None and 
                    isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
                    self.lr_scheduler.step(valid_metrics['avg_total_loss'])
                    # Log current learning rate after plateau scheduler step
                    current_lr = self.optimizer.param_groups[0]['lr']
                    if self.experiment_logger is not None:
                        self.experiment_logger.log_lr(current_lr)

                # Check early stopping and save best model
                if self.early_stopping is not None:
                    metric_name = self.cfg.early_stopping_metric
                    if metric_name not in valid_metrics:
                        LOGGER.warning(f"Early stopping metric '{metric_name}' not found in validation metrics. "
                                     f"Available metrics: {list(valid_metrics.keys())}. Using 'avg_total_loss'.")
                        metric_name = 'avg_total_loss'
                    
                    metric_value = valid_metrics[metric_name]
                    
                    # Check if this is the best model so far
                    is_best = self._is_best_model(metric_value)
                    if is_best:
                        self._save_best_model(epoch, metric_value)
                    
                    # Check if we should stop training
                    stop_early = self.early_stopping(metric_value, epoch)

            # stored after validation and early stopping, i.e. the state right before the next epoch
            self._save_training_state(epoch)

            if stop_early:
                LOGGER.info(f"Training stopped early at epoch {epoch}")
                if self._best_model_path:
                    LOGGER.info(f"Best model saved at: {self._best_model_path}")
                break

            longest_epoch = max(longest_epoch, time.time() - epoch_start)
            if self.cfg.max_training_hours is not None and epoch < last_epoch:
                elapsed = time.time() - self._start_time
                if (elapsed + longest_epoch) / 3600 > self.cfg.max_training_hours:
                    LOGGER.info(f"Training paused at epoch {epoch} (max_training_hours: {self.cfg.max_training_hours}, "
                                f"elapsed {elapsed / 3600:.2f} h, longest epoch {longest_epoch / 3600:.2f} h). "
                                f"Continue with resume_training_state.")
                    break

        # make sure to close tensorboard to avoid losing the last epoch
        if self.cfg.log_tensorboard:
            self.experiment_logger.stop_tb()

    def _resumes_training_state(self) -> bool:
        return bool(self.cfg.is_continue_training and self.cfg.resume_training_state)

    def _get_start_epoch_number(self):
        if self._resumes_training_state():
            epoch = self._last_resumable_epoch(self.cfg.run_dir)
        elif self.cfg.is_continue_training:
            if self.cfg.continue_from_epoch is not None:
                epoch = self.cfg.continue_from_epoch
            else:
                weight_path = [x for x in sorted(list(self.cfg.run_dir.glob('model_epoch*.pt')))][-1]
                epoch = weight_path.name[-6:-3]
        else:
            epoch = 0
        return int(epoch)

    def _restore_training_state(self):
        if self._resumes_training_state():
            epoch = f"{self._epoch:03d}"
            weight_path = self.cfg.base_run_dir / f"model_epoch{epoch}.pt"
        elif self.cfg.continue_from_epoch is not None:
            epoch = f"{self.cfg.continue_from_epoch:03d}"
            weight_path = self.cfg.base_run_dir / f"model_epoch{epoch}.pt"
        else:
            weight_path = [x for x in sorted(list(self.cfg.base_run_dir.glob('model_epoch*.pt')))][-1]
            epoch = weight_path.name[-6:-3]

        optimizer_path = self.cfg.base_run_dir / f"optimizer_state_epoch{epoch}.pt"

        LOGGER.info(f"Continue training from epoch {int(epoch)}")
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.optimizer.load_state_dict(torch.load(str(optimizer_path), map_location=self.device))

    @staticmethod
    def _last_resumable_epoch(run_dir: Path) -> int:
        """Last epoch of a run with model weights, optimizer state and training state on disk."""
        epochs = sorted(int(p.stem[-3:]) for p in run_dir.glob('training_state_epoch*.pt'))
        epochs = [
            e for e in epochs if (run_dir / f"model_epoch{e:03d}.pt").is_file() and
            (run_dir / f"optimizer_state_epoch{e:03d}.pt").is_file()
        ]
        if not epochs:
            raise FileNotFoundError(f"No epoch in {run_dir} has model weights, optimizer state and training state "
                                    "(training_state_epochNNN.pt), which resume_training_state needs.")
        return epochs[-1]

    def _save_training_state(self, epoch: int):
        """Store everything besides weights and optimizer that the next epoch depends on."""
        np_state = np.random.get_state()
        state = {
            'epoch': epoch,
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            'early_stopping': self.early_stopping.state_dict() if self.early_stopping is not None else None,
            'best_metric_value': getattr(self, '_best_metric_value', None),
            'best_epoch': self._best_epoch,
            'logger_epoch': self.experiment_logger.epoch,
            'logger_update': self.experiment_logger.update,
            'rng_python': random.getstate(),
            # plain Python types, so that the file loads with torch.load(weights_only=True)
            'rng_numpy': (np_state[0], np_state[1].tolist(), int(np_state[2]), int(np_state[3]), float(np_state[4])),
            'rng_torch': torch.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state(self.device) if self.device.type == 'cuda' else None,
        }
        # write to a temporary file first, so that an interrupted write never leaves a broken state file
        path = self.cfg.run_dir / f"training_state_epoch{epoch:03d}.pt"
        tmp_path = path.with_suffix('.tmp')
        torch.save(state, str(tmp_path))
        tmp_path.replace(path)

    def _restore_resume_state(self):
        path = self.cfg.base_run_dir / f"training_state_epoch{self._epoch:03d}.pt"
        # weights_only: the file holds tensors and plain Python types only
        state = torch.load(str(path), map_location='cpu', weights_only=True)
        if state['epoch'] != self._epoch:
            raise RuntimeError(f"{path} holds epoch {state['epoch']}, expected {self._epoch}.")
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(state['lr_scheduler'])
        if self.early_stopping is not None:
            self.early_stopping.load_state_dict(state['early_stopping'])
            self._best_metric_value = state['best_metric_value']
            self._best_epoch = state['best_epoch']
            if self._best_epoch is not None:
                # the interrupted run may have written best_model.pt in an epoch after the resumed one; the weights of
                # the best epoch are only on disk if that epoch was saved (save_weights_every)
                best_model_path = self.cfg.run_dir / "best_model.pt"
                best_epoch_weights = self.cfg.run_dir / f"model_epoch{self._best_epoch:03d}.pt"
                if best_epoch_weights.is_file():
                    shutil.copyfile(best_epoch_weights, best_model_path)
                self._best_model_path = best_model_path
        self.experiment_logger.epoch = state['logger_epoch']
        self.experiment_logger.update = state['logger_update']

        random.setstate(state['rng_python'])
        name, keys, pos, has_gauss, cached_gaussian = state['rng_numpy']
        np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached_gaussian))
        torch.set_rng_state(state['rng_torch'])
        if state['rng_cuda'] is not None and self.device.type == 'cuda':
            torch.cuda.set_rng_state(state['rng_cuda'], self.device)
        LOGGER.info(f"Training state of epoch {self._epoch} restored from {path}")

    def _save_weights_and_optimizer(self, epoch: int):
        weight_path = self.cfg.run_dir / f"model_epoch{epoch:03d}.pt"
        torch.save(self.model.state_dict(), str(weight_path))

        optimizer_path = self.cfg.run_dir / f"optimizer_state_epoch{epoch:03d}.pt"
        torch.save(self.optimizer.state_dict(), str(optimizer_path))

        self.experiment_logger.log_model(weight_path, optimizer_path)

    def _is_best_model(self, metric_value: float) -> bool:
        """Check if the current metric value represents the best model so far."""
        if self.early_stopping is None:
            return False
        
        if self.early_stopping.mode == 'min':
            return metric_value < self._best_metric_value
        else:  # mode == 'max'
            return metric_value > self._best_metric_value

    def _save_best_model(self, epoch: int, metric_value: float):
        """Save the current model as the best model through wandb logger."""
        best_model_path = self.cfg.run_dir / "best_model.pt"
        torch.save(self.model.state_dict(), str(best_model_path))

        # Log as artifact through wandb logger
        if self.experiment_logger is not None:
            self.experiment_logger.log_best_model(best_model_path, epoch)

        self._best_model_path = best_model_path
        self._best_metric_value = metric_value
        self._best_epoch = epoch
        LOGGER.info(f"New best model saved at epoch {epoch} with {self.cfg.early_stopping_metric}: "
                   f"{metric_value:.5f}")

    def _train_epoch(self, epoch: int):
        self.model.train()
        self.experiment_logger.train()

        # process bar handle
        n_iter = min(self._max_updates_per_epoch, len(self.loader)) if self._max_updates_per_epoch is not None else None
        pbar = tqdm(self.loader, file=sys.stdout, disable=self._disable_pbar, total=n_iter)
        pbar.set_description(f'# Epoch {epoch}')

        # Iterate in batches over training set
        nan_count = 0
        for i, data in enumerate(pbar):
            if self._max_updates_per_epoch is not None and i >= self._max_updates_per_epoch:
                break

            for key in data.keys():
                if key.startswith('x_d'):
                    data[key] = {k: v.to(self.device) for k, v in data[key].items()}
                elif not key.startswith('date'):
                    data[key] = data[key].to(self.device)

            # apply possible pre-processing to the batch before the forward pass
            data = self.model.pre_model_hook(data, is_train=True)

            # get predictions
            predictions = self.model(data)

            if self.noise_sampler_y is not None:
                for key in filter(lambda k: 'y' in k, data.keys()):
                    noise = self.noise_sampler_y.sample(data[key].shape)
                    # make sure we add near-zero noise to originally near-zero targets
                    data[key] += (data[key] + self._target_mean / self._target_std) * noise.to(self.device)

            loss, all_losses = self.loss_obj(predictions, data)

            # early stop training if loss is NaN
            if torch.isnan(loss):
                nan_count += 1
                if nan_count > self._allow_subsequent_nan_losses:
                    raise RuntimeError(f"Loss was NaN for {nan_count} times in a row. Stopped training.")
                LOGGER.warning(f"Loss is Nan; ignoring step. (#{nan_count}/{self._allow_subsequent_nan_losses})")
            else:
                nan_count = 0

                # delete old gradients
                self.optimizer.zero_grad()

                # get gradients
                loss.backward()

                if self.cfg.clip_gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_gradient_norm)

                # update weights
                self.optimizer.step()

            pbar.set_postfix_str(f"Loss: {loss.item():.4f}")

            self.experiment_logger.log_step(**{k: v.item() for k, v in all_losses.items()})

    def _set_random_seeds(self):
        if self.cfg.seed is None:
            self.cfg.seed = int(np.random.uniform(low=0, high=1e6))

        # fix random seeds for various packages
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

    def _set_device(self):
        if self.cfg.device is not None:
            if self.cfg.device.startswith("cuda"):
                gpu_id = int(self.cfg.device.split(':')[-1])
                if gpu_id > torch.cuda.device_count():
                    raise RuntimeError(f"This machine does not have GPU #{gpu_id} ")
                else:
                    self.device = torch.device(self.cfg.device)
            elif self.cfg.device == "mps":
                if torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                else:
                    raise RuntimeError("MPS device is not available.")
            else:
                self.device = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        LOGGER.info(f"### Device {self.device} will be used for training")

    def _create_folder_structure(self):
        # a resumed run continues inside the run directory of the interrupted run
        if self._resumes_training_state():
            self.cfg.base_run_dir = self.cfg.run_dir
            self.cfg.train_dir = self.cfg.run_dir / "train_data"
            self.cfg.train_dir.mkdir(parents=True, exist_ok=True)
            if self.cfg.log_n_figures is not None:
                self.cfg.img_log_dir = self.cfg.run_dir / "img_log"
                self.cfg.img_log_dir.mkdir(parents=True, exist_ok=True)
            return

        # create as subdirectory within run directory of base run
        if self.cfg.is_continue_training:
            folder_name = f"continue_training_from_epoch{self._epoch:03d}"

            # store dir of base run for easier access in weight loading
            self.cfg.base_run_dir = self.cfg.run_dir
            self.cfg.run_dir = self.cfg.run_dir / folder_name

        # create as new folder structure
        else:
            now = datetime.now()
            day = f"{now.day}".zfill(2)
            month = f"{now.month}".zfill(2)
            hour = f"{now.hour}".zfill(2)
            minute = f"{now.minute}".zfill(2)
            second = f"{now.second}".zfill(2)
            run_name = f'{self.cfg.experiment_name}_{day}{month}_{hour}{minute}{second}'

            # if no directory for the runs is specified, a 'runs' folder will be created in the current working dir
            if self.cfg.run_dir is None:
                self.cfg.run_dir = Path().cwd() / "runs" / run_name
            else:
                self.cfg.run_dir = self.cfg.run_dir / run_name

        # create folder + necessary subfolder
        if not self.cfg.run_dir.is_dir():
            self.cfg.train_dir = self.cfg.run_dir / "train_data"
            self.cfg.train_dir.mkdir(parents=True)
        else:
            raise RuntimeError(f"There is already a folder at {self.cfg.run_dir}")
        if self.cfg.log_n_figures is not None:
            self.cfg.img_log_dir = self.cfg.run_dir / "img_log"
            self.cfg.img_log_dir.mkdir(parents=True)
