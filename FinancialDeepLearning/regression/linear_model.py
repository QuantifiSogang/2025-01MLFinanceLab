import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from tqdm import tqdm
from FinancialDeepLearning.base.summary import LinearResult

__docformat__ = 'restructuredtext en'
__author__ = "<Tommy Lee>"
__all__ = ['LinearRegression','LinearNet']


class LinearRegression(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_layers: list,
            output_dim: int,
            activation_function : str = None
    ) -> None:
        super(LinearRegression, self).__init__()

        layers = []
        prev_dim = input_dim

        # if activation_function is None:
        #     # Create hidden layers
        #     for hidden_dim in hidden_layers:
        #         layers.append(nn.Linear(prev_dim, hidden_dim))
        #
        #         layers.append(nn.ReLU())  # Activation function for hidden layers
        #         prev_dim = hidden_dim
        #
        # else :
        # Create hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation_function.lower() == 'relu' :
                layers.append(nn.ReLU())  # Activation function for hidden layers
            elif activation_function.lower() == 'sigmoid' :
                layers.append(nn.Sigmoid())
            elif activation_function.lower() == 'tanh' :
                layers.append(nn.Tanh())
            elif activation_function.lower() == 'silu' :
                layers.append(nn.SiLU())
            else :
                pass
            prev_dim = hidden_dim


        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class LinearNet(object):
    def __init__(self, X: pd.DataFrame, y: pd.Series | pd.DataFrame) -> None:
        super(LinearNet, self).__init__()
        self.X = X
        self.y = y
        self.model = None

    def fit(
        self,
        criterion: str,
        optimizer: str,
        activation_function: str = 'relu',
        hidden_layers: list = [],  # List of hidden layer dimensions, if the list is empty, then the model has no hidden layer
        learning_rate: float = 0.001,
        num_epochs: int = 10
    ) -> torch.nn.Module :

        # Define the loss function
        if criterion.lower() == 'mse':
            cr = nn.MSELoss()
        elif criterion.lower() == 'mae':
            cr = nn.L1Loss()
        else:
            raise ValueError(f'Criterion {criterion} is not supported.')

        X_train_tensor = torch.tensor(self.X.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.y.values, dtype=torch.float32).unsqueeze(1)

        output_dim = 1 if isinstance(self.y, pd.Series) else self.y.shape[1]
        input_dim = self.X.shape[1]

        # Initialize the model with hidden layers
        self.model = LinearRegression(
            input_dim = input_dim,
            hidden_layers = hidden_layers,
            output_dim = output_dim,
            activation_function = activation_function
        )

        # Optimizer selection
        if optimizer.lower() == 'sgd':
            opt = optim.SGD(self.model.parameters(), lr=learning_rate)
        elif optimizer.lower() == 'adam':
            opt = optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f'Optimizer {optimizer} is not supported.')

        # Training loop
        for epoch in tqdm(range(num_epochs)):
            self.model.train()
            opt.zero_grad()

            # Forward pass
            outputs = self.model(X_train_tensor)
            loss = cr(outputs, y_train_tensor)

            # Backward pass and optimization
            loss.backward()
            opt.step()

        return self.model

    def predict(self, X_test) -> pd.Series:
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

        self.model.eval()  # Set evaluation mode
        with torch.no_grad():
            predicted = self.model(X_test_tensor).numpy()

        res = pd.Series(predicted.squeeze(), index=X_test.index, name='predicted')

        return res

    def fittedvalues(self) -> pd.Series:
        X_train_tensor = torch.tensor(self.X.values, dtype=torch.float32)

        self.model.eval()  # Set evaluation mode
        with torch.no_grad():
            predicted = self.model(X_train_tensor).numpy()

        res = pd.Series(predicted.squeeze(), index=self.X.index, name='fittedvalues')
        return res

    def resid(self) -> pd.Series:
        y_pred = self.fittedvalues()
        resid = self.y - y_pred
        resid.name = 'residual'
        return resid

    def calculate_weights_and_bias(self):
        params = list(self.model.parameters()) # params list setting

        W_combined = params[0].detach().numpy().T  # (in_features, out_features)
        b_combined = params[1].detach().numpy()

        for i in range(2, len(params), 2):
            W_next = params[i].detach().numpy().T  # (in_features, out_features)로 전치
            b_next = params[i + 1].detach().numpy()  # (out_features,)

            # W_combined (in_features, prev_out_features) dot W_next (prev_out_features, out_features)
            W_combined = np.dot(W_combined, W_next)  # (in_features, out_features)

            # b_combined (prev_out_features,) dot W_next (prev_out_features, out_features) + b_next (out_features,)
            b_combined = np.dot(b_combined, W_next) + b_next  # (out_features,)

        return W_combined.flatten(), b_combined.flatten()

    def summary(self) -> LinearResult.summary:
        y_pred = self.fittedvalues()
        resid = self.y - y_pred

        weights, bias = self.calculate_weights_and_bias()

        smry = LinearResult(
            self.model,
            self.X, self.y,
            residuals=resid,
            y_pred=y_pred
        ).summary(weights, bias)
        return smry
