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

class OutputRNNbi(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(OutputRNNbi, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True,num_layers=3,bidirectional=True) # Either GRU or LSTM
        self.fc = nn.Linear(hidden_size*2, output_size)  # Maps hidden state to output

    def forward(self, x):
        # Pass through RNN
        rnn_out, _ = self.rnn(x)  # rnn_out: (batch_size, seq_len, hidden_size)
        
        # Apply fully connected layer
        output = self.fc(rnn_out)  # output: (batch_size, seq_len, output_size)
        return output

class KernelRNNbi(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(KernelRNNbi, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2,bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, kernel_size)  # Output kernel weights

    def forward(self, y):
        # y: [batch_size, time_steps, input_dim]
        rnn_out, _ = self.rnn(y)  # RNN output: [batch_size, time_steps, hidden_dim]
        
        # Select the last time step's output
        last_output = rnn_out[:, -1, :]  # Shape: [batch_size, hidden_dim]
        
        # Generate a single kernel for the entire batch
        kernel = self.fc(last_output)  # Shape: [batch_size, kernel_size]
        return kernel  # Shape: [batch_size, kernel_size]
# [batch_size, time_steps, kernel_size]

class DeconvolutionCNNbi(nn.Module):
    def __init__(self, kernel_size):
        super(DeconvolutionCNNbi, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, y, kernel):
        # y: [batch_size, time_steps, 1]
        # kernel: [batch_size, kernel_size]
        
        # Move time dimension to the end of the channel dimension
        # y: [batch_size, 1, time_steps]
        y = y.permute(0, 2, 1)
        kernel = kernel.unsqueeze(1)  # kernel: [batch_size, 1, kernel_size]

        # Calculate padding
        total_padding = self.kernel_size - 1
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding

        # Apply padding to the entire batch at once
        # F.pad can handle batching: input [batch_size, C, L] -> output [batch_size, C, L + padding]
        padded_y = F.pad(y, (left_padding, right_padding), mode='reflect') 
        # padded_y: [batch_size, 1, time_steps + total_padding]

        # Reshape input for grouped convolution:
        # We want each batch sample to be treated as a separate group.
        # Convert [batch_size, 1, length] to [1, batch_size, length]
        padded_y = padded_y.permute(1, 0, 2)  # Now: [1, batch_size, length]

        # Perform grouped convolution
        # groups = batch_size, in_channels = batch_size, out_channels = batch_size
        # kernel: [batch_size, 1, kernel_size]
        # input: [1, batch_size, length]
        x_hat = F.conv1d(padded_y, kernel, groups=kernel.size(0))

        # x_hat: [1, batch_size, time_steps]
        # Permute back: [batch_size, 1, time_steps]
        x_hat = x_hat.permute(1, 0, 2)

        # Finally, permute to [batch_size, time_steps, 1]
        x_hat = x_hat.permute(0, 2, 1)

        return x_hat
    
# Combined RNN-CNN Model
class RNNCNNDeconvolutionRNNbi(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, kernel_size=40, output_dim=1):
        super(RNNCNNDeconvolutionRNNbi, self).__init__()
        self.kernel_rnn = KernelRNNbi(input_dim, hidden_dim, kernel_size)
        self.deconv_cnn = DeconvolutionCNNbi(kernel_size)
        self.output_rnn = OutputRNNbi(input_dim,hidden_dim,output_dim)
    def forward(self, y):
        kernel = self.kernel_rnn(y)  # Predict kernel with RNN
        x_hat = self.deconv_cnn(y, kernel)  # Deconvolve signal
        x_out = self.output_rnn(x_hat)
        return x_out  # Shape: [batch_size, timepoints, 1]



class RNNCNNDeconvolutionRNN_bi_Trainer(BaseTrainer):
    def __str__(self):
        criterion = self.base_criterion
        if criterion is None:
            criterion = "blocky"
        return f"RNNCNNDeconvolutionRNN_bi_{criterion}"

    def __init__(self, model=None, config=None, base_criterion=None):
        super().__init__(model=model, config=config)
        self.model_cls = RNNCNNDeconvolutionRNNbi
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
                loss = self.criterion(outputs, batch_y.unsqueeze(2),epoch=epoch)
                
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
            elif epoch>31:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch} epochs')
                break
                
            # convergence check not learning noothing
#            if len(loss_history) >= convergence_window:
#                recent_losses = loss_history[-convergence_window:]
#                loss_variance = np.var(recent_losses)
#                if loss_variance < convergence_threshold:
#                    print(f'Loss converged after {epoch} epochs')
#                    break

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
