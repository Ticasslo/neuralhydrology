import logging
import warnings
from typing import List, Optional

import torch

import neuralhydrology.training.loss as loss
from neuralhydrology.training import regularization
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


def get_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """Get specific optimizer object, depending on the run configuration.
    
    Currently only 'Adam' and 'AdamW' are supported. Parameters with an ``_optim`` attribute (a dict of optimizer
    hyperparameters, e.g. the SSM parameters of the S4D core) are put into separate parameter groups with these
    hyperparameters, one group per distinct setting, following ``setup_optimizer`` in ``example.py`` of
    https://github.com/state-spaces/s4. All other parameters form the first group with the global learning rate. Without
    such parameters, the optimizer is created from ``model.parameters()`` as before.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to be optimized.
    cfg : Config
        The run configuration.

    Returns
    -------
    torch.optim.Optimizer
        Optimizer object that can be used for model training.
    """
    params = list(model.parameters())
    special_params = [p for p in params if hasattr(p, '_optim')]
    if special_params:
        param_groups = [{'params': [p for p in params if not hasattr(p, '_optim')]}]
        hyperparameters = []
        for p in special_params:
            if p._optim not in hyperparameters:
                hyperparameters.append(p._optim)
        for hp in hyperparameters:
            param_groups.append({'params': [p for p in special_params if p._optim == hp], **hp})
    else:
        param_groups = params

    if cfg.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=cfg.learning_rate)
    elif cfg.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=cfg.learning_rate)
    else:
        raise NotImplementedError(f"{cfg.optimizer} not implemented or not linked in `get_optimizer()`")

    return optimizer


def get_lr_scheduler(optimizer: torch.optim.Optimizer, cfg: Config) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Get learning rate scheduler object, depending on the run configuration.
    
    Supports common PyTorch learning rate schedulers.
    
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer for which to create a learning rate scheduler.
    cfg : Config
        The run configuration containing lr_scheduler settings.

    Returns
    -------
    Optional[torch.optim.lr_scheduler.LRScheduler]
        Learning rate scheduler object that can be used during training.
        Returns None if no scheduler is configured.
        
    Raises
    ------
    NotImplementedError
        If the specified scheduler type is not implemented.
    ValueError
        If required scheduler parameters are missing.
    """
    if not cfg.lr_scheduler or 'type' not in cfg.lr_scheduler:
        return None
    
    scheduler_type = cfg.lr_scheduler['type'].lower()
    
    # Remove 'type' from the config and use the rest as kwargs
    scheduler_kwargs = {k: v for k, v in cfg.lr_scheduler.items() if k != 'type'}
    
    if scheduler_type == 'steplr':
        if 'step_size' not in scheduler_kwargs:
            raise ValueError("StepLR scheduler requires 'step_size' parameter")
        return torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_kwargs)
    
    elif scheduler_type == 'multisteplr':
        if 'milestones' not in scheduler_kwargs:
            raise ValueError("MultiStepLR scheduler requires 'milestones' parameter")
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_kwargs)
    
    elif scheduler_type == 'exponentiallr':
        if 'gamma' not in scheduler_kwargs:
            raise ValueError("ExponentialLR scheduler requires 'gamma' parameter")
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_kwargs)
    
    elif scheduler_type == 'cosineannealinglr':
        if 'T_max' not in scheduler_kwargs:
            raise ValueError("CosineAnnealingLR scheduler requires 'T_max' parameter")
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_kwargs)
    
    elif scheduler_type == 'reducelronplateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)
    
    elif scheduler_type == 'cosineannealingwarmrestarts':
        if 'T_0' not in scheduler_kwargs:
            raise ValueError("CosineAnnealingWarmRestarts scheduler requires 'T_0' parameter")
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_kwargs)
    
    elif scheduler_type == 'linearlr':
        if 'start_factor' not in scheduler_kwargs:
            scheduler_kwargs['start_factor'] = 1.0
        if 'total_iters' not in scheduler_kwargs:
            raise ValueError("LinearLR scheduler requires 'total_iters' parameter")
        return torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler_kwargs)
    
    elif scheduler_type == 'polynomiallr':
        if 'total_iters' not in scheduler_kwargs:
            raise ValueError("PolynomialLR scheduler requires 'total_iters' parameter")
        if 'power' not in scheduler_kwargs:
            scheduler_kwargs['power'] = 1.0
        return torch.optim.lr_scheduler.PolynomialLR(optimizer, **scheduler_kwargs)
    
    else:
        raise NotImplementedError(f"Learning rate scheduler '{scheduler_type}' not implemented or not linked in `get_lr_scheduler()`")


def get_loss_obj(cfg: Config) -> loss.BaseLoss:
    """Get loss object, depending on the run configuration.
    
    Currently supported are 'MSE', 'NSE', 'RMSE', 'GMMLoss', 'CMALLoss', and 'UMALLoss'.
    
    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    loss.BaseLoss
        A new loss instance that implements the loss specified in the config or, if different, the loss required by the 
        head.
    """
    if cfg.loss.lower() == "mse":
        loss_obj = loss.MaskedMSELoss(cfg)
    elif cfg.loss.lower() == "nse":
        loss_obj = loss.MaskedNSELoss(cfg)
    elif cfg.loss.lower() == "weightednse":
        warnings.warn("'WeightedNSE loss has been removed. Use 'NSE' with 'target_loss_weights'", FutureWarning)
        loss_obj = loss.MaskedNSELoss(cfg)
    elif cfg.loss.lower() == "rmse":
        loss_obj = loss.MaskedRMSELoss(cfg)
    elif cfg.loss.lower() == "gmmloss":
        loss_obj = loss.MaskedGMMLoss(cfg)
    elif cfg.loss.lower() == "cmalloss":
        loss_obj = loss.MaskedCMALLoss(cfg)
    elif cfg.loss.lower() == "umalloss":
        loss_obj = loss.MaskedUMALLoss(cfg)
    else:
        raise NotImplementedError(f"{cfg.loss} not implemented or not linked in `get_loss()`")

    return loss_obj


def get_regularization_obj(cfg: Config) -> List[regularization.BaseRegularization]:
    """Get list of regularization objects.
    
    Currently, only the 'tie_frequencies' regularization is implemented.
    
    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    List[regularization.BaseRegularization]
        List of regularization objects that will be added to the loss during training.
    """
    regularization_modules = []
    for reg_item in cfg.regularization:
        if isinstance(reg_item, str):
            reg_name = reg_item
            reg_weight = 1.0
        else:
            reg_name, reg_weight = reg_item
        if reg_name == "tie_frequencies":
            regularization_modules.append(regularization.TiedFrequencyMSERegularization(cfg=cfg, weight=reg_weight))
        elif reg_name == "forecast_overlap":
            regularization_modules.append(regularization.ForecastOverlapMSERegularization(cfg=cfg, weight=reg_weight))
        else:
            raise NotImplementedError(f"{reg_name} not implemented or not linked in `get_regularization_obj()`.")

    return regularization_modules
