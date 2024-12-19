import torch
from torch import nn
from torch import optim
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from loss.loss import combined_penalty

lambda_val = 2.6  # Pretuned hyperparameter - do not touch


def metrics(MAE_list, MSE_list, smoothness_loss_list):
    avg_test_MAE = np.mean(MAE_list)
    std_test_MAE = np.std(MAE_list)
    avg_test_MSE = np.mean(MSE_list)
    std_test_MSE = np.std(MSE_list)
    avg_smoothness_loss = np.mean(smoothness_loss_list)
    std_smoothness_loss = np.std(smoothness_loss_list)

    return {
        "MAE": {"avg": avg_test_MAE, "std": std_test_MAE},
        "MSE": {"avg": avg_test_MSE, "std": std_test_MSE},
        "smoothness": {"avg": avg_smoothness_loss, "std": std_smoothness_loss},
    }


class BaseTrainer(ABC):
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    def full_eval(self, X_val, y_val):
        eval_dataset = TensorDataset(X_val, y_val)
        eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=True)

        results = {}

        loss_MAE = nn.L1Loss()
        loss_MSE = nn.MSELoss()

        flat_MAE_list = []
        flat_MSE_list = []
        flat_smoothness_loss_list = []

        concat_MAE_list = []
        concat_MSE_list = []
        concat_smoothness_loss_list = []

        flat_preds, concat_preds = [], []
        flat_xs, concat_xs = [], []
        flat_ys, concat_ys = [], []

        with torch.no_grad():
            for batch_x, batch_y in eval_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                flat_pred = self.model(batch_x)
                batch_y_reshaped = batch_y.unsqueeze(-1)

                flat_xs.append(batch_x.cpu())
                flat_ys.append(batch_y.cpu())
                flat_preds.append(flat_pred.cpu())

                flat_val_MAE = loss_MAE(flat_pred, batch_y_reshaped)
                flat_val_MSE = loss_MSE(flat_pred, batch_y_reshaped)
                flat_smoothness_loss = lambda_val * combined_penalty(flat_pred)

                flat_MAE_list.append(flat_val_MAE.item())
                flat_MSE_list.append(flat_val_MSE.item())
                flat_smoothness_loss_list.append(flat_smoothness_loss.item())

                concat_batch_x = torch.cat([batch_x, batch_x], dim=1)
                concat_batch_y = torch.cat([batch_y, batch_y], dim=1)

                concat_pred = self.model(concat_batch_x)
                concat_batch_y_reshaped = concat_batch_y.unsqueeze(-1)

                concat_xs.append(concat_batch_x.cpu())
                concat_ys.append(concat_batch_y.cpu())
                concat_preds.append(concat_pred.cpu())

                concat_val_MAE = loss_MAE(concat_pred, concat_batch_y_reshaped)
                concat_val_MSE = loss_MSE(concat_pred, concat_batch_y_reshaped)
                concat_smoothness_loss = lambda_val * combined_penalty(concat_pred)

                concat_MAE_list.append(concat_val_MAE.item())
                concat_MSE_list.append(concat_val_MSE.item())
                concat_smoothness_loss_list.append(concat_smoothness_loss.item())

        results["flat"] = metrics(
            flat_MAE_list, flat_MSE_list, flat_smoothness_loss_list
        )
        results["concat"] = metrics(
            concat_MAE_list, concat_MSE_list, concat_smoothness_loss_list
        )

        flat_xs = torch.cat(flat_xs)
        flat_ys = torch.cat(flat_ys)
        flat_preds = torch.cat(flat_preds)

        concat_xs = torch.cat(concat_xs)
        concat_ys = torch.cat(concat_ys)
        concat_preds = torch.cat(concat_preds)

        return (
            results,
            (flat_xs, flat_ys, flat_preds),
            (concat_xs, concat_ys, concat_preds),
        )

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def evaluate(self, X_val, y_val):
        pass

    @staticmethod
    def _get_loss_function(loss_name, **params):
        loss_fns = {
            "mse": nn.MSELoss,
            "mae": nn.L1Loss,
            "huber": nn.HuberLoss,
            "smooth_l1": nn.SmoothL1Loss,
        }
        return loss_fns[loss_name](**params)

    @staticmethod
    def _get_optimizer(optim_name, parameters, **params):
        optimizers = {
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            "sgd": optim.SGD,
            "rmsprop": optim.RMSprop,
        }
        return optimizers[optim_name](parameters, **params)

    @staticmethod
    def get_optuna_params(trial):
        """Define the hyperparameter search space"""

        optimizer_name = trial.suggest_categorical(
            "optimizer", ["adam", "adamw", "sgd", "rmsprop"]
        )

        optimizer_params = {
            "lr": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        }

        if optimizer_name == "sgd":
            optimizer_params.update(
                {
                    "momentum": trial.suggest_float("momentum", 0.0, 0.99),
                    "nesterov": trial.suggest_categorical("nesterov", [True, False]),
                }
            )
        elif optimizer_name in ["adam", "adamw"]:
            optimizer_params.update(
                {
                    "betas": (
                        trial.suggest_float("beta1", 0.8, 0.99),
                        trial.suggest_float("beta2", 0.9, 0.999),
                    ),
                    "eps": trial.suggest_float("eps", 1e-8, 1e-6, log=True),
                }
            )

        loss_name = trial.suggest_categorical(
            "loss_fn", ["blocky_loss"]  # ['mse', 'mae', 'huber', 'smooth_l1']
        )

        loss_params = {}
        if loss_name == "huber":
            loss_params["delta"] = trial.suggest_float("huber_delta", 0.1, 1.0)

        if loss_name == "blocky_loss":
            loss_params["alpha"] = (1.0,)
            loss_params["beta"] = (1.0,)
            loss_params["lambda_tv"] = (1.0,)
            loss_params["lambda_const"] = (1.0,)
            loss_params["lambda_val"] = (1.0,)

        training_logic = trial.suggest_categorical(
            "training_logic", ["fixed", "early_stopping", "convergence"]
        )

        training_params = {
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            "epochs": trial.suggest_int("epochs", 50, 200),
            "training_logic": training_logic,
        }

        if training_logic == "early_stopping":
            training_params.update(
                {
                    "patience": trial.suggest_int("patience", 5, 20),
                    "min_delta": trial.suggest_float("min_delta", 1e-5, 1e-3, log=True),
                }
            )
        elif training_logic == "convergence":
            training_params.update(
                {
                    "convergence_window": trial.suggest_int(
                        "convergence_window", 3, 10
                    ),
                    "convergence_threshold": trial.suggest_float(
                        "convergence_threshold", 1e-5, 1e-3, log=True
                    ),
                }
            )

        model_params = {
            "input_size": 1,  # fixed
            "hidden_size": trial.suggest_int("hidden_size", 32, 256),
            "kernel_size": trial.suggest_int("kernel_size", 20, 60),
            "output_size": 1,  # fixed
            "dropout_prob": trial.suggest_float("dropout_prob", 0.1, 0.5),
        }

        return {
            **model_params,
            "optimizer": optimizer_name,
            "optimizer_params": optimizer_params,
            "loss_fn": loss_name,
            "loss_params": loss_params,
            **training_params,
        }
