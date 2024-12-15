import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from abc import ABC, abstractmethod
import optuna

class BaseTrainer(ABC):
    def __init__(self, model, config):
        self.model = model
        self.config = config
    
    @abstractmethod
    def train(self, X_train, y_train):
        pass
    
    @abstractmethod
    def evaluate(self, X_val, y_val):
        pass
    
    @abstractmethod
    def get_optuna_params(self, trial):
        pass

    @staticmethod
    def _get_loss_function(loss_name, **params):
        loss_fns = {
            'mse': nn.MSELoss,
            'mae': nn.L1Loss,
            'huber': nn.HuberLoss,
            'smooth_l1': nn.SmoothL1Loss
        }
        return loss_fns[loss_name](**params)
    
    @staticmethod
    def _get_optimizer(optim_name, parameters, **params):
        optimizers = {
            'adam': optim.Adam,
            'adamw': optim.AdamW,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop
        }
        return optimizers[optim_name](parameters, **params)
    
    @staticmethod
    def get_optuna_params(trial):
        """Define the hyperparameter search space"""

        optimizer_name = trial.suggest_categorical(
            'optimizer', ['adam', 'adamw', 'sgd', 'rmsprop']
        )
        
        optimizer_params = {
            'lr': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        }
        
        if optimizer_name == 'sgd':
            optimizer_params.update({
                'momentum': trial.suggest_float('momentum', 0.0, 0.99),
                'nesterov': trial.suggest_categorical('nesterov', [True, False])
            })
        elif optimizer_name in ['adam', 'adamw']:
            optimizer_params.update({
                'betas': (
                    trial.suggest_float('beta1', 0.8, 0.99),
                    trial.suggest_float('beta2', 0.9, 0.999)
                ),
                'eps': trial.suggest_float('eps', 1e-8, 1e-6, log=True)
            })
        
        loss_name = trial.suggest_categorical(
            'loss_fn', ["blocky_loss"] # ['mse', 'mae', 'huber', 'smooth_l1']
        )
        
        loss_params = {}
        if loss_name == 'huber':
            loss_params['delta'] = trial.suggest_float('huber_delta', 0.1, 1.0)

        if loss_name == "blocky_loss":
            loss_params['alpha'] = 1.0,
            loss_params['beta'] = 1.0,
            loss_params['lambda_tv'] = 1.0,
            loss_params['lambda_const'] = 1.0,
            loss_params['lambda_val'] = 1.0,

        training_logic = trial.suggest_categorical('training_logic', ['fixed', 'early_stopping', 'convergence'])
        
        training_params = {
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'epochs': trial.suggest_int('epochs', 50, 200),
            'training_logic': training_logic,
        }

        if training_logic == 'early_stopping':
            training_params.update({
                'patience': trial.suggest_int('patience', 5, 20),
                'min_delta': trial.suggest_float('min_delta', 1e-5, 1e-3, log=True)
            })
        elif training_logic == 'convergence':
            training_params.update({
                'convergence_window': trial.suggest_int('convergence_window', 3, 10),
                'convergence_threshold': trial.suggest_float('convergence_threshold', 1e-5, 1e-3, log=True)
            })
        
        model_params = {
            'input_size': 1, # fixed
            'hidden_size': trial.suggest_int('hidden_size', 32, 256),
            'kernel_size': trial.suggest_int('kernel_size', 20, 60),
            'output_size': 1, # fixed
            'dropout_prob': trial.suggest_float('dropout_prob', 0.1, 0.5)
        }
        
        return {
            **model_params,
            'optimizer': optimizer_name,
            'optimizer_params': optimizer_params,
            'loss_fn': loss_name,
            'loss_params': loss_params,
            **training_params
        }
        