import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

__docformat__ = 'reStructuredText en'
__author__ = '<Tommy Lee>'
__all__ = ['Conv1DNet','Conv2DNet','ConvolutionalNeuralNetwork']

class Conv1DNet(nn.Module) :
    def __init__(
            self, window_size : int,
            nb_input_series : int,
            nb_filter : int,
            filter_length : int,
            nb_outputs : int,
            use_shrinkage : bool = False,
            shrinkage_dim : int = 16,
            use_dropout : bool = False,
            dropout_rate : float = 0.5,
            activation_function: str = 'tanh'
        ) -> None :
        super(Conv1DNet, self).__init__()
        # Conv1D Layer
        self.conv1d = nn.Conv1d(
            in_channels = nb_input_series,
            out_channels = nb_filter,
            kernel_size = filter_length
        )
        self.use_shrinkage = use_shrinkage
        if self.use_shrinkage :
            self.shrinkage_layer = nn.Linear(
                (window_size - filter_length + 1) * nb_filter,
                shrinkage_dim
            )
            self.output_dim = shrinkage_dim
        else :
            self.output_dim = (window_size - filter_length + 1) * nb_filter

        self.use_dropout = use_dropout
        if self.use_dropout :
            self.dropout = nn.Dropout(p = dropout_rate)

        # Fully connected layer
        self.fc = nn.Linear(
            (window_size - filter_length + 1) * nb_filter,
            nb_outputs
        )
        self.activation_function = activation_function

    def forward(
            self, x : torch.Tensor
        ) -> torch.Tensor :
        # PyTorch expects input shape as (batch, channels, length)
        x = x.permute(0, 2, 1)  # Reshape to (batch, channels, length)
        if self.activation_function.lower() == 'tanh' :
            x = F.tanh(self.conv1d(x))
        elif self.activation_function.lower() == 'relu' :
            x = F.relu(self.conv1d(x))
        elif self.activation_function.lower() == 'sigmoid' :
            x = F.sigmoid(self.conv1d(x))
        elif self.activation_function.lower() == 'silu' :
            x = F.silu(self.conv1d(x))
        elif self.activation_function.lower() == 'softmax' :
            x = F.softmax(self.conv1d(x), dim = 1)
        elif self.activation_function.lower() == 'elu' :
            x = F.elu(self.conv1d(x))
        elif self.activation_function.lower() == 'none' :
            x = x
        else :
            raise ValueError(f'activation function {self.activation_function} is not supported')

        x = x.reshape(x.size(0), -1)  # Flatten
        if self.use_shrinkage:
            x = F.relu(self.shrinkage_layer(x))

        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc(x)

        return x

class Conv2DNet(nn.Module) :
    def __init__(
            self, number_of_assets: int,  # number of assets
            window: int,  # total periods
            nb_input_series: int,  # number of channels
            nb_filter: int,  # number of filters
            kernel_size: tuple,  # filter size
            nb_outputs: int,  # output dimension
            use_shrinkage: bool = False,
            shrinkage_dim: int = 128,  # dimension after shrinkage
            use_dropout: bool = False,  # dropout option
            dropout_rate: float = 0.5,  # dropout rate
            activation_function: str = 'tanh'
    ) -> None:
        super(Conv2DNet, self).__init__()
        # Conv2D layer
        self.conv2d = nn.Conv2d(
            in_channels = nb_input_series,
            out_channels = nb_filter,
            kernel_size = kernel_size
        )

        self.use_shrinkage = use_shrinkage
        conv_output_height = number_of_assets - kernel_size[0] + 1
        conv_output_width = window - kernel_size[1] + 1
        flattened_dim = nb_filter * conv_output_height * conv_output_width

        if self.use_shrinkage:
            self.shrinkage_layer = nn.Linear(flattened_dim, shrinkage_dim)
            self.output_dim = shrinkage_dim
        else:
            self.output_dim = flattened_dim

        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer
        self.fc = nn.Linear(self.output_dim, nb_outputs)
        self.activation_function = activation_function

    def forward(self, x) -> torch.Tensor:
        if self.activation_function.lower() == 'tanh':
            x = F.tanh(self.conv2d(x))
        elif self.activation_function.lower() == 'relu':
            x = F.relu(self.conv2d(x))
        elif self.activation_function.lower() == 'sigmoid':
            x = F.sigmoid(self.conv2d(x))
        elif self.activation_function.lower() == 'silu':
            x = F.silu(self.conv2d(x))
        elif self.activation_function.lower() == 'softmax':
            x = F.softmax(self.conv2d(x), dim=1)
        elif self.activation_function.lower() == 'elu':
            x = F.elu(self.conv2d(x))
        elif self.activation_function.lower() == 'none':
            x = x
        else:
            raise ValueError(f'activation function {self.activation_function} is not supported')
        x = x.reshape(x.size(0), -1)  # Flatten

        if self.use_shrinkage:
            x = F.tanh(self.shrinkage_layer(x))

        if self.use_dropout:
            x = self.dropout(x)

        x = self.fc(x)
        return x

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(
            self,
            data : pd.Series | pd.DataFrame,
        ) -> None :
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.X = None
        self.y = None
        self.data = data
        self.model = None # initialize model

    def _lagged_data(self, p) -> tuple :
        X = []
        y = []
        if (self.data.values.ndim == 1) or (self.data.values.shape[1] == 1):
            for i in range(len(self.data) - p) :
                X.append(self.data.iloc[i:i+p].values)
                y.append(self.data.iloc[p])
            return np.array(X), np.array(y)
        elif (self.data.values.ndim == 2) and (self.data.values.shape[1] >= 2):
            for i in range(len(self.data) - p):
                X.append(self.data.iloc[i:i + p].values.T)
                y.append(self.data.iloc[i + p].mean())
            return np.array(X), np.array(y)

    def fit(
            self,
            window: int,  # total periods
            nb_input_series: int,  # number of channels
            nb_filter: int,  # number of filters
            kernel_size: tuple or int,  # filter size
            nb_outputs: int,  # output dimension
            use_shrinkage: bool = False,
            shrinkage_dim: int = 128,  # dimension after shrinkage
            use_dropout: bool = False,  # dropout option
            dropout_rate: float = 0.5,  # dropout rate
            activation_function: str = 'tanh',
            criterion : str = 'mse',
            optimizer: str = 'adam',
            num_epochs : int = 100,
            batch_size: int = 32,
            learning_rate: float = 1e-3
        ) -> nn.Module :
        self.X, self.y = self._lagged_data(p = window)

        if (self.data.values.ndim == 1) or (self.data.values.shape[1] == 1):
            self.X_tensor = torch.tensor(self.X, dtype=torch.float32).unsqueeze(-1)
            self.y_tensor = torch.tensor(self.y, dtype=torch.float32).unsqueeze(-1)
            self.window = window

            dataset = TensorDataset(self.X_tensor, self.y_tensor)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False
            )
            self.model = Conv1DNet(
                window_size = window,
                nb_input_series = nb_input_series,
                nb_filter = nb_filter,
                filter_length = kernel_size,
                nb_outputs = nb_outputs,
                use_shrinkage = use_shrinkage,
                shrinkage_dim = shrinkage_dim,
                use_dropout = use_dropout,
                dropout_rate = dropout_rate,
                activation_function = activation_function
            )
        elif (self.data.values.ndim == 2) and (self.data.values.shape[1] >= 2):
            self.X_tensor = torch.tensor(self.X, dtype=torch.float32).unsqueeze(1)
            self.y_tensor = torch.tensor(self.y, dtype=torch.float32).unsqueeze(-1)
            self.window = window

            dataset = TensorDataset(self.X_tensor, self.y_tensor)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False
            )
            self.model = Conv2DNet(
                number_of_assets = self.data.values.shape[1],
                window = window,
                nb_input_series = nb_input_series,
                nb_filter = nb_filter,
                kernel_size = kernel_size,
                nb_outputs = nb_outputs,
                use_shrinkage = use_shrinkage,
                shrinkage_dim = shrinkage_dim,
                use_dropout = use_dropout,
                activation_function = activation_function
            )
        else :
            assert ValueError('invalid number of inputs')

        if optimizer.lower() == 'adam' :
            self.optimizer = optim.Adam(self.model.parameters(), lr = learning_rate)
        elif optimizer.lower() == 'sgd' :
            self.optimizer = optim.SGD(self.model.parameters(), lr = learning_rate)
        elif optimizer.lower() == 'rmsprop' :
            self.optimizer = optim.RMSprop(self.model.parameters(), lr = learning_rate)
        else :
            assert ValueError('invalid optimizer')

        if criterion.lower() == 'mse' :
            self.criterion = nn.MSELoss()
        elif criterion.lower() == 'mae' :
            self.criterion = nn.L1Loss()
        else :
            assert ValueError('invalid criterion')

        for _ in tqdm(range(num_epochs)) :
            for X_batch, y_batch in dataloader :
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

        return self.model

    def fittedvalues(self) -> pd.Series :
        with torch.no_grad() :
            output = self.model(self.X_tensor)
        res = pd.DataFrame(
            output,
            index = self.data.index[self.window:],
            columns = ['CNN smoothed']
        )

        return res