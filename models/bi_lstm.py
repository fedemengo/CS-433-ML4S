import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models.trainer.trainer import BaseTrainer
from utils import get_free_gpu
import json

def pretty_print(data):
    formatted_json = json.dumps(data, 
        indent=2,
        sort_keys=True,
        separators=(',', ': '),
        ensure_ascii=False
    )
    print(formatted_json)

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.2):        
        super(BiLSTMModel, self).__init__()
        
        # Bidirectional LSTM layer
        self.bilstm = nn.LSTM(
            input_size, 
            hidden_size, 
            batch_first=True, 
            num_layers=2, 
            bidirectional=True, 
            dropout=dropout_prob,
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 2x because of bidirectional

    def forward(self, x):
        # Pass through BiLSTM
        lstm_out, _ = self.bilstm(x)  # lstm_out: (batch_size, seq_len, hidden_size * 2)
        
        # Use the output from the last time step
        output = self.fc(lstm_out)  # Last time step output
        
        return output.squeeze(-1)


class BiLSTMTrainer(BaseTrainer):
    model_cls = BiLSTMModel
    
    def __init__(self, model=None, config=None):
        pretty_print(config)
        self.model = model
        self.config = config or {}
        self.device = torch.device(f'cuda:{get_free_gpu()}' if torch.cuda.is_available() else 'cpu')
        print(f"running on {self.device}")
        if model:
            self.model.to(self.device)
            self._setup_training_components()
    
    def _setup_training_components(self):
        self.criterion = self._get_loss_function(
            self.config.get('loss_fn', 'mse'),
            **self.config.get('loss_params', {})
        )
        
        self.optimizer = self._get_optimizer(
            self.config.get('optimizer', 'adam'),
            self.model.parameters(),
            **self.config.get('optimizer_params', {})
        )
        
        if self.config.get('use_scheduler', False):
            self.scheduler = self._get_scheduler(
                self.config.get('scheduler', 'reduce_lr'),
                **self.config.get('scheduler_params', {})
            )

    def train(self, X_train, y_train):
        self.model.train()
        batch_size = self.config.get('batch_size', 32)
        n_epochs = self.config.get('epochs', 100)
        best_loss = float('inf')

        print("X_train", X_train.shape)
        print(f"learning params: batch_size={batch_size}, n_epochs={n_epochs}")
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.FloatTensor(y_train)
        )

        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        for epoch in range(n_epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                
                if self.config.get('clip_grad', False):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.get('clip_value', 1.0)
                    )
                
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            
            # Learning rate scheduling
            if hasattr(self, 'scheduler'):
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_loss)
                else:
                    self.scheduler.step()
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                if self.config.get('save_best', False):
                    self.save_model('best_model.pth')
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {avg_loss:.6f}')
                
    def evaluate(self, X_val, y_val):
        print("evaluating")
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_val).to(self.device)
            y_tensor = torch.FloatTensor(y_val).to(self.device)
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
        return loss.item()
    
    @staticmethod
    def _get_loss_function(loss_name, **params):
        loss_fns = {
            'mse': nn.MSELoss,
            'mae': nn.L1Loss,
            'huber': nn.HuberLoss,
            'cross_entropy': nn.CrossEntropyLoss  # if you have a classification task
        }
        return loss_fns[loss_name](**params)
    
    @staticmethod
    def _get_optimizer(optim_name, parameters, **params):
        optimizers = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
        }
        return optimizers[optim_name](parameters, **params)
    
    def _get_scheduler(self, scheduler_name, **params):
        schedulers = {
            'reduce_lr': optim.lr_scheduler.ReduceLROnPlateau,
            'cosine': optim.lr_scheduler.CosineAnnealingLR,
            'step': optim.lr_scheduler.StepLR
        }
        return schedulers[scheduler_name](self.optimizer, **params)

    @staticmethod
    def get_optuna_params(trial):
        """Define the hyperparameter search space"""
        model_params = {
            'input_size': 1, # fixed single voxel
            'hidden_size': trial.suggest_int('hidden_size', 32, 128, log=True),
            'output_size': 1, # fixed single voxel
            'dropout_prob': trial.suggest_float('dropout_prob', 0.1, 0.5, log=True)
        }
        
        epochs = trial.suggest_int('epochs', 50, 100, log=True)
        
        training_params = {
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'epochs': epochs,
        }
        
        optimizer_name = trial.suggest_categorical(
            'optimizer', ['adam', 'sgd']
        )
        
        optimizer_params = {
            'lr': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        }
        
        if optimizer_name == 'sgd':
            optimizer_params['momentum'] = trial.suggest_float('momentum', 0.0, 0.99, log=True)
        
        use_scheduler = trial.suggest_categorical('use_scheduler', [True, False])
        scheduler_params = {}
        
        if use_scheduler:
            scheduler_name = trial.suggest_categorical(
                'scheduler', ['reduce_lr', 'cosine', 'step']
            )
            if scheduler_name == 'step':
                scheduler_params.update({
                    'step_size': trial.suggest_int('step_size', 5, 15, log=True),
                    'gamma': trial.suggest_float('gamma', 0.1, 0.5, log=True)
                })
            elif scheduler_name == 'reduce_lr':
                scheduler_params.update({
                    'patience': trial.suggest_int('patience', 3, 10, log=True),
                    'factor': trial.suggest_float('factor', 0.1, 0.5, log=True)
                })
            elif scheduler_name == 'cosine':
                scheduler_params.update({
                    'T_max': epochs
                })

        return {
            **model_params,
            **training_params,
            'optimizer': optimizer_name,
            'optimizer_params': optimizer_params,
            'use_scheduler': use_scheduler,
            'scheduler': scheduler_name if use_scheduler else None,
            'scheduler_params': scheduler_params if use_scheduler else {}
        }
    
    def predict(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
        
        return outputs.cpu().numpy()