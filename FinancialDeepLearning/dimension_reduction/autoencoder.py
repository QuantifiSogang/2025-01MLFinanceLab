import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

__docformat__ = 'reStructuredText en'
__author__ = '<Tommy Lee>'
__all__ = ['Autoencoder','LinearAutoEncoder']

class Autoencoder(nn.Module) :
    def __init__(self, input_dim, hidden_dim) -> None :
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x) -> torch.tensor :
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class LinearAutoEncoder(nn.Module) :
    def __init__(self, data : pd.Series | pd.DataFrame) -> None :
        super(LinearAutoEncoder, self).__init__()
        self.data = data

        self.y = torch.tensor(
            data.values,
            dtype = torch.float32
        )

    def fit(
            self,
            hidden_dim : int,
            epochs : int = 100,
            batch_size : int = 16,
            learning_rate : float = 0.001,
            criterion : str = 'mse',
            optimizer : str = 'adam',
            regularization : float = 0
        ) -> nn.Module :
        input_dim = self.y.shape[1]
        self.model = Autoencoder(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

        # criterion
        if criterion.lower() == 'mse' :
            criterion = nn.MSELoss()
        elif criterion.lower() == 'mae' :
            criterion = nn.L1Loss()
        else :
            assert ValueError('Invalid criterion')

        # optimizer
        if optimizer.lower() == 'adam' :
            optimizer = optim.Adam(
                self.model.parameters(),
                lr = learning_rate,
                weight_decay = regularization
            )
        elif optimizer.lower() == 'rmsprop' :
            optimizer = optim.RMSprop(
                self.model.parameters(),
                lr = learning_rate,
                weight_decay = regularization
            )
        elif optimizer.lower() == 'sgd' :
            optimizer = optim.SGD(self.model.parameters(), lr = learning_rate)
        else :
            assert ValueError('Invalid optimizer')

        for epoch in tqdm(range(epochs)) :
            self.model.train() # set a instance to train modes
            epoch_loss = 0
            permutation = torch.randperm(self.y.size(0))
            for i in range(0, self.y.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_x = self.y[indices]

                optimizer.zero_grad()  # initialize gradient
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_x)  # caculate loss
                loss.backward()  # backpropagation
                optimizer.step()  # update params

                epoch_loss += loss.item()

        return self.model

    def reconstruct(self) -> pd.DataFrame :
        self.model.eval() # evaluation mode
        with torch.no_grad() :
            ae_recon = self.model(self.y)

        res = pd.DataFrame(
            ae_recon.numpy(),
            index = self.data.index,
            columns = self.data.columns
        )
        return res

    def left_singular_vector(self) -> pd.DataFrame :
        w2 = self.model.decoder.weight.data.numpy()

        # calculate SVD values
        ae_decoder_lsv, _, _ = np.linalg.svd(w2, full_matrices = False)
        return ae_decoder_lsv

    def projected_components(self):
        mu = np.mean(self.data.values, axis = 0)
        ae_decoder_lsv = self.left_singular_vector()
        ae_lsv_projections = np.array(self.data.values - mu) @ ae_decoder_lsv

        res = pd.DataFrame(
            ae_lsv_projections,
            index = self.data.index,
            columns = [f'AE_{i + 1}' for i in range(self.hidden_dim)]
        )
        return res

    def covariance(self):
        mu = np.mean(self.data.values, axis = 0)
        C = np.dot(
            (self.data.values - mu).T, self.data.values - mu
        )
        res = pd.DataFrame(
            C,
            index = self.data.columns,
            columns = self.data.columns
        )
        return res

    def total_variance(self) -> float :
        mu = np.mean(self.data.values, axis=0)
        C = np.dot(
            (self.data.values - mu).T, self.data.values - mu
        )
        total_variance = np.sum(np.diag(C))
        return total_variance

    def decoder_weights_lambda(self) -> pd.DataFrame :
        w2 = self.model.decoder.weight.data.numpy()
        mu = np.mean(self.data.values, axis=0)
        C = np.dot(
            (self.data.values - mu).T, self.data.values - mu
        )
        res = pd.DataFrame(
            w2.T @ C @ w2,
            index = [f'AE_{i + 1}' for i in range(self.hidden_dim)],
            columns = [f'AE_{i + 1}' for i in range(self.hidden_dim)]
        )
        return res

    def decoder_lsv_lambda(self) -> pd.DataFrame :
        ae_decoder_lsv = self.left_singular_vector()
        mu = np.mean(self.data.values, axis=0)
        C = np.dot(
            (self.data.values - mu).T, self.data.values - mu
        )
        res = pd.DataFrame(
            ae_decoder_lsv.T @ C @ ae_decoder_lsv,
            index=[f'AE_{i + 1}' for i in range(self.hidden_dim)],
            columns=[f'AE_{i + 1}' for i in range(self.hidden_dim)]
        )
        return res

