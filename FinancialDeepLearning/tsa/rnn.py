import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from tqdm import tqdm
from FinancialDeepLearning.tsa.time_series_cell import AlphaRNNCell, AlphatRNNCell
from FinancialDeepLearning.base.summary import TimeSeriesResult

__docformat__ = 'restructuredtext en'
__author__ = "<Tommy Lee>"
__all__ = ['RecurrentNeuralNetwork','AlphaRNN','AlphatRNN','SimpleRecurrentNeuralNet']

class RecurrentNeuralNetwork(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            num_layers: int = 2
    ) -> None:
        super(RecurrentNeuralNetwork, self).__init__()
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):
        out, hidden = self.rnn(x)
        if len(out.shape) == 3:
            out = out[:, -1, :]
        else:
            out = out
        out = self.fc(out)
        return out

class AlphatRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1):
        super(AlphatRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cells = nn.ModuleList(
            [AlphatRNNCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.fc = nn.Linear(hidden_size, output_size)  # 최종 출력 레이어 추가

    def forward(self, input, hidden=None, smoothed_hidden=None):
        batch_size, seq_len, _ = input.size()

        if hidden is None:
            hidden = self.init_hidden(batch_size)
        if smoothed_hidden is None:
            smoothed_hidden = self.init_smoothed_hidden(batch_size)

        for t in range(seq_len):
            x = input[:, t, :]  # Input at time t
            for i, cell in enumerate(self.cells):
                hidden[i], smoothed_hidden[i], alpha_t = cell(x, hidden[i], smoothed_hidden[i])
                x = hidden[i]  # Last hidden state is passed to the next layer

        out = self.fc(x)
        return out, hidden, smoothed_hidden

    def init_hidden(self, batch_size):
        return [torch.zeros(batch_size, self.hidden_size, device=self.cells[0].phi.device) for _ in
                range(self.num_layers)]

    def init_smoothed_hidden(self, batch_size):
        return [torch.zeros(batch_size, self.hidden_size, device=self.cells[0].phi.device) for _ in
                range(self.num_layers)]

class AlphaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1, alpha=0.5):
        super(AlphaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.alpha = alpha

        self.cells = nn.ModuleList(
            [AlphaRNNCell(input_size if i == 0 else hidden_size, hidden_size, alpha) for i in range(num_layers)]
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden=None, smoothed_hidden=None):
        batch_size, seq_len, _ = input.size()

        if hidden is None:
            hidden = self.init_hidden(batch_size)
        if smoothed_hidden is None:
            smoothed_hidden = self.init_smoothed_hidden(batch_size)

        for t in range(seq_len):
            x = input[:, t, :]  # Input at time t
            for i, cell in enumerate(self.cells):
                hidden[i], smoothed_hidden[i] = cell(x, hidden[i], smoothed_hidden[i])
                x = hidden[i]  # Last hidden state is passed to the next layer

        out = self.fc(x)
        return out, hidden, smoothed_hidden

    def init_hidden(self, batch_size):
        return [torch.zeros(batch_size, self.hidden_size, device=self.cells[0].phi.device) for _ in
                range(self.num_layers)]

    def init_smoothed_hidden(self, batch_size):
        return [torch.zeros(batch_size, self.hidden_size, device=self.cells[0].phi.device) for _ in
                range(self.num_layers)]


class SimpleRecurrentNeuralNet(object):
    def __init__(
            self,
            data: pd.DataFrame,
            p: int,
            method: str,
            exog: pd.DataFrame = None) -> None:
        super(SimpleRecurrentNeuralNet, self).__init__()
        self.data = data
        self.p = p
        self.exog = exog
        self.method = method

        self.X = None
        self.y = None

        self.model = None

    def fit(
            self,
            criterion: str,
            optimizer: str,
            activation_function: str = 'relu',
            hidden_layers: int = 1,
            learning_rate: float = 0.001,
            num_epochs: int = 10,
            batch_size: int = 2,
            alpha: float = 0.5
        ) -> torch.nn.Module:

        # Define the loss function
        if criterion == 'mse':
            cr = nn.MSELoss()
        elif criterion == 'mae':
            cr = nn.L1Loss()
        else:
            raise ValueError(f'Criterion {criterion} is not supported.')

        X = pd.concat(
            [self.data.shift(p) for p in range(1, self.p + 1)], axis=1
        )
        y = self.data.iloc[self.p:]
        X = X.iloc[self.p:]

        X.columns = [f'RNN.L{p}' for p in range(1, self.p + 1)]
        y.name = 'target'

        self.X = X
        self.y = y

        X_train_values = X.values.reshape(-1, 1, self.p)
        y_train_values = y.values.reshape(-1, 1)

        X_tensor = torch.tensor(X_train_values, dtype=torch.float32)
        y_tensor = torch.tensor(y_train_values, dtype=torch.float32)

        output_size = 1

        if self.method.lower() == 'rnn':
            self.model = RecurrentNeuralNetwork(
                self.p,
                self.p,
                output_size,
                hidden_layers
            )
        elif self.method.lower() == 'alpha-rnn':
            self.model = AlphaRNN(
                self.p,
                self.p,
                output_size,
                hidden_layers,
                alpha=alpha
            )
        elif self.method.lower() == 'alphat-rnn':
            self.model = AlphatRNN(
                self.p,
                self.p,
                output_size,
                hidden_layers,
            )
        else:
            assert ValueError('Method is not supported.')

        if optimizer.lower() == 'sgd':
            opt = optim.SGD(
                self.model.parameters(),
                lr=learning_rate
            )
        elif optimizer.lower() == 'adam':
            opt = optim.Adam(
                self.model.parameters(),
                lr=learning_rate
            )
        else:
            raise ValueError(f'Optimizer {optimizer} is not supported.')

        dataset = torch.utils.data.TensorDataset(
            X_tensor, y_tensor
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )

        for epoch in tqdm(range(num_epochs)):
            self.model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                opt.zero_grad()

                # Handle different model types (RNN, AlphaRNN, Alpha-t RNN)
                if isinstance(self.model, (AlphaRNN, AlphatRNN)):
                    outputs, _, _ = self.model(batch_X)  # Ignore hidden states
                else:
                    outputs = self.model(batch_X)

                # Squeeze the outputs and match with target shape
                loss = cr(outputs.squeeze(), batch_y.squeeze())  # Squeeze to ensure matching shapes
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * batch_X.size(0)
            epoch_loss /= len(dataloader.dataset)

        return self.model

    def predict(self, test_data: pd.Series) -> pd.Series:
        X_test = pd.concat(
            [test_data.shift(p) for p in range(1, self.p + 1)], axis=1
        )
        X_test = X_test.iloc[self.p:]

        X_test_tensor = torch.tensor(
            X_test.values,
            dtype=torch.float32
        )
        self.model.eval()
        with torch.no_grad():
            # Handle different model types
            if isinstance(self.model, (AlphaRNN, AlphatRNN)):
                predicted_full, _, _ = self.model(X_test_tensor)  # Only use predicted output
            else:
                predicted_full = self.model(X_test_tensor)

            predicted_full = predicted_full.squeeze().numpy()

        res = pd.Series(
            predicted_full,
            index=test_data.index[self.p:]
        )
        return res

    def fittedvalues(self) -> pd.Series:
        X_tensor = torch.tensor(
            self.X.values.reshape(-1, 1, self.p),
            dtype=torch.float32
        )
        self.model.eval()  # set evaluation mode
        with torch.no_grad():
            # Handle different model types
            if isinstance(self.model, (AlphaRNN, AlphatRNN)):
                predicted, _, _ = self.model(X_tensor)  # Only use predicted output
            else:
                predicted = self.model(X_tensor)

            predicted = predicted.squeeze().numpy()
        res = pd.DataFrame(predicted, index=self.data.index[self.p:], columns=['fittedvalues'])
        return res

    def params(self) -> list:
        params = list(self.model.parameters())
        return params

    def resid(self) -> pd.Series:
        y_pred = self.fittedvalues()
        resid = self.y - y_pred['fittedvalues']
        resid.name = 'residual'
        return resid

    def summary(self) -> TimeSeriesResult.summary:
        y_pred = self.fittedvalues()
        resid = self.resid()

        if self.method.lower() == 'rnn':
            model_name = 'RNN'
        elif self.method.lower() == 'alpha-rnn':
            model_name = 'Alpha RNN'
        elif self.method.lower() == 'alphat-rnn':
            model_name = 'Alpha-t RNN'
        else:
            assert ValueError('Method is not supported.')

        smry = TimeSeriesResult(
            self.model,
            self.X, self.y,
            residuals=resid,
            y_pred=y_pred
        ).summary(model_name, self.p)
        return smry