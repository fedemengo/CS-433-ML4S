import torch
import torch.nn as nn
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from utils import get_free_gpu
from loss.blocky_loss import blocky_loss
from models.trainer.trainer import BaseTrainer

class LSTM_attention(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, dropout_prob=0.2):
        super(LSTM_attention, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2, dropout=dropout_prob)
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_size)
        
        # Apply attention mechanism at each time step
        attn_weights = F.softmax(self.attn(lstm_out), dim=2)  # Shape: (batch_size, seq_len, hidden_size)
        weighted_out = attn_weights * lstm_out  # Apply attention: (batch_size, seq_len, hidden_size)
        
        # Pass weighted outputs through fully connected layer
        output = self.fc(weighted_out)  # Shape: (batch_size, seq_len, output_size)
        return output




class LSTM_attention_Trainer(BaseTrainer):
    def __str__(self):
        criterion = self.base_criterion
        if criterion is None:
            criterion = "blocky"
        return f"LSTM_attention_{criterion}"

    def __init__(self, model=None, config=None, base_criterion=None):
        super().__init__(model=model, config=config)
        self.model_cls = LSTM_attention
        self.model = model
        self.config = config or {}
        self.base_criterion = base_criterion
        print(f"running on {self.device}")
        
        if model:
            self.model.to(self.device)
            self._setup_training_components()
    
    def _setup_training_components(self):
        print("LOSS", self.config.get('loss_fn', 'blocky_loss'))
        print("LOSS params", self.config.get('loss_params', {}))
        print("base criterion", self.base_criterion)

        if self.base_criterion:
            self.criterion = self.base_criterion
        else:
            self.criterion = self._get_loss_function(
                self.config.get('loss_fn', 'blocky_loss'),
                **self.config.get('loss_params', {})
            )
        
        self.optimizer = self._get_optimizer(
            self.config.get('optimizer', 'adam'),
            self.model.parameters(),
            **self.config.get('optimizer_params', {
                'lr': 0.0001,
                'weight_decay': 1e-5,
            })
        )

    @staticmethod
    def get_optuna_params(trial):
        """Define the hyperparameter search space"""

        loss_name = trial.suggest_categorical(
            'loss_fn', ["blocky_loss"]
        )
        
        loss_params = {}
        if loss_name == "blocky_loss":
            loss_params = {
                'alpha': 8.0,
                'beta': 1.0,
                'lambda_tv': 1.0,
                'lambda_const': 0.2,
                'lambda_val': 3.6788,
            }

        training_logic = trial.suggest_categorical('training_logic', ['fixed'])
        
        training_params = {
            'batch_size': trial.suggest_categorical('batch_size', [16, 32]),
            'epochs': trial.suggest_int('epochs', 50, 200),
            'training_logic': training_logic,
        }

        if training_logic == 'early_stopping':
            training_params.update({
                'patience': trial.suggest_int('patience', 5, 20),
                'min_delta': trial.suggest_float('min_delta', 1e-5, 1e-3, log=True)
            })
        
        model_params = {
            'input_size': 1, # fixed
            'hidden_size': trial.suggest_int('hidden_size', 32, 128, log=True),
            'kernel_size': trial.suggest_int('kernel_size', 20, 60, log=True),
            'output_size': 1, # fixed
        }
        
        return {
            **model_params,
            'loss_fn': loss_name,
            'loss_params': loss_params,
            **training_params
        }

    def train(self, X_tensor, y_tensor):
        self.model.train()
        batch_size = self.config.get('batch_size', 32)
        n_epochs = self.config.get('epochs', 30)
        
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        patience = self.config.get('patience', 10)
        min_delta = self.config.get('min_delta', 1e-4)
        best_loss = float('inf')
        patience_counter = 0
        
        # loss convergence
        convergence_window = self.config.get('convergence_window', 5)
        loss_history = []
        convergence_threshold = self.config.get('convergence_threshold', 1e-4)
        
        for epoch in range(n_epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # print("batch size", batch_X.shape)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)

                # print("pred shape", outputs.shape)
                # print("true shape", batch_y.shape)
                # print("expanded", batch_y.unsqueeze(2).shape)
                loss = self.criterion(outputs, batch_y.unsqueeze(2))
                
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_loss)
            
            if epoch % 5 == 0:
                print(f'Epoch {epoch}, Loss: {avg_loss:.6f}')
            
            # early stopping check
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch} epochs')
                break
                
            # convergence check not learning noothing
            if len(loss_history) >= convergence_window:
                recent_losses = loss_history[-convergence_window:]
                loss_variance = np.var(recent_losses)
                if loss_variance < convergence_threshold:
                    print(f'Loss converged after {epoch} epochs')
                    break

    def evaluate(self, X_val, y_val):
        print("evaluating")
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_val).to(self.device)
            y_tensor = torch.FloatTensor(y_val).to(self.device)
            outputs = self.model(X_tensor)

            loss = self.criterion(outputs, y_tensor.unsqueeze(2))
        return loss.item()

    def predict(self, X, batch_size=32):
        self.model.eval()
        predictions = []
        
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i + batch_size]).to(self.device)
            with torch.no_grad():
                pred = self.model(batch)
            predictions.append(pred.cpu().numpy())
            
        return np.concatenate(predictions)

    @staticmethod
    def _get_loss_function(loss_name, **params):
        print(params)
        loss_fns = {
            'blocky_loss': blocky_loss,
        }
        return loss_fns[loss_name](**params)