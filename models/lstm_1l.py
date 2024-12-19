import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from loss.blocky_loss import blocky_loss
from models.trainer.trainer import BaseTrainer


class LSTM1l(nn.Module):
    def __init__(self, input_size=1, hidden_size=40, output_size=1):
        super(LSTM1l, self).__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, batch_first=True, num_layers=1
        )  # Either GRU or LSTM
        self.fc = nn.Linear(hidden_size, output_size)  # Maps hidden state to output

    def forward(self, x):
        # Pass through RNN
        rnn_out, _ = self.rnn(x)  # rnn_out: (batch_size, seq_len, hidden_size)

        # Apply fully connected layer
        output = self.fc(rnn_out)  # output: (batch_size, seq_len, output_size)
        return output


class LSTM1lTrainer(BaseTrainer):
    def __str__(self):
        criterion = self.base_criterion
        if criterion is None:
            criterion = "blocky"
        return f"LSTM1l_{criterion}"

    def __init__(self, model=None, config=None, base_criterion=None):
        super().__init__(model=model, config=config)
        self.model_cls = LSTM1l
        self.model = model
        self.config = config or {}
        self.base_criterion = base_criterion
        print(f"running on {self.device}")

        if model:
            self.model.to(self.device)
            self._setup_training_components()

    def _setup_training_components(self):
        print("loss", self.config.get("loss_fn", "blocky_loss"))
        print("loss params", self.config.get("loss_params", {}))
        print("base criterion", self.base_criterion)

        if self.base_criterion:
            self.criterion = self.base_criterion
        else:
            self.criterion = self._get_loss_function(
                self.config.get("loss_fn", "blocky_loss"),
                **self.config.get("loss_params", {}),
            )

        self.optimizer = self._get_optimizer(
            self.config.get("optimizer", "adam"),
            self.model.parameters(),
            **self.config.get(
                "optimizer_params",
                {
                    "lr": 0.0001,
                    "weight_decay": 1e-5,
                },
            ),
        )

    @staticmethod
    def _get_loss_function(loss_name, **params):
        print(params)
        loss_fns = {
            "blocky_loss": blocky_loss,
        }
        return loss_fns[loss_name](**params)

    def train(self, X_tensor, y_tensor):
        self.model.train()
        batch_size = self.config.get("batch_size", 32)
        n_epochs = self.config.get("epochs", 30)

        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        patience = self.config.get("patience", 10)
        min_delta = self.config.get("min_delta", 1e-4)
        best_loss = float("inf")
        patience_counter = 0

        # loss convergence
        convergence_window = self.config.get("convergence_window", 5)
        loss_history = []
        convergence_threshold = self.config.get("convergence_threshold", 1e-4)

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
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

            # early stopping check
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

            # convergence check not learning noothing
            if len(loss_history) >= convergence_window:
                recent_losses = loss_history[-convergence_window:]
                loss_variance = np.var(recent_losses)
                if loss_variance < convergence_threshold:
                    print(f"Loss converged after {epoch} epochs")
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
            batch = torch.FloatTensor(X[i : i + batch_size]).to(self.device)
            with torch.no_grad():
                pred = self.model(batch)
            predictions.append(pred.cpu().numpy())

        return np.concatenate(predictions)
