import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from models.trainer.trainer import BaseTrainer

# todo: optimize over conv out size? 
class CNNRNNModel(nn.Module):
   def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.2):
       super(CNNRNNModel, self).__init__()
        # 1D Convolutional Layer
       self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
       self.pool = nn.MaxPool1d(kernel_size=1)
       
       # RNN Layer (LSTM or GRU)
       self.rnn = nn.LSTM(64, hidden_size, batch_first=True, num_layers=2, dropout=dropout_prob)
       
       # Fully connected layer
       self.fc = nn.Linear(hidden_size, output_size)

   def forward(self, x):
       # Pass through CNN
       x = x.permute(0, 2, 1)  # Change shape to (batch_size, channels, seq_len)
       x = torch.relu(self.conv1(x))
       x = self.pool(x)
       
       # Reshape for RNN input
       x = x.permute(0, 2, 1)  # (batch_size, seq_len, channels)
       
       # Pass through RNN
       rnn_out, _ = self.rnn(x)
       
       # Pass the RNN output (all time steps) through the FC layer
       output = self.fc(rnn_out)  # (batch_size, seq_len, output_size)
       return output


class CNNRNNTrainer(BaseTrainer):
    model_cls = CNNRNNModel
    
    def __init__(self, model=None, config=None):
        self.model = model
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    def train(self, X_train, y_train):
        self.model.train()
        batch_size = self.config.get('batch_size', 32)
        n_epochs = self.config.get('epochs', 100)

        print(X_train)
        print(y_train)
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train.values), 
            torch.FloatTensor(y_train.values)
        )
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
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_loss)
            
            if epoch % 10 == 0:
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
            'loss_fn', ['mse', 'mae', 'huber', 'smooth_l1']
        )
        
        loss_params = {}
        if loss_name == 'huber':
            loss_params['delta'] = trial.suggest_float('huber_delta', 0.1, 1.0)

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