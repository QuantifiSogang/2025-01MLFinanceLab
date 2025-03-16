# library import
# dependency : pandas, numpy, statsmodels, torch, scipy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd

from tqdm import tqdm

__docformat__ = 'restructuredtext en'
__author__ = "<Tommy Lee>"
__all__ = ['AlphaRNNCell','AlphatRNNCell']


class AlphaRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, alpha=0.5):
        super(AlphaRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.alpha = alpha

        self.phi = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.hidden_weights = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.phi, a=0)
        nn.init.kaiming_uniform_(self.hidden_weights, a=0)
        nn.init.zeros_(self.bias)

    def forward(self, input, hidden, smoothed_hidden):
        # h_hat = torch.sigmoid(F.linear(input, self.phi) + smoothed_hidden)
        h_hat = F.linear(input, self.phi) + F.linear(hidden, self.hidden_weights) + self.bias
        smoothed_h = self.alpha * hidden + (1 - self.alpha) * smoothed_hidden

        return h_hat, smoothed_h

class AlphatRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AlphatRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.phi = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.hidden_weights = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))

        self.alpha_fc = nn.Linear(input_size + hidden_size, 1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.phi, a=0)
        nn.init.kaiming_uniform_(self.hidden_weights, a=0)
        nn.init.zeros_(self.bias)
        nn.init.xavier_uniform_(self.alpha_fc.weight)

    def forward(self, input, hidden, smoothed_hidden):
        combined = torch.cat((input, hidden), dim=1)
        alpha_t = self.alpha_fc(combined)

        # h_hat = φ(input) + W(hidden) + bias
        h_hat = F.linear(input, self.phi) + F.linear(hidden, self.hidden_weights) + self.bias

        # Smoothed hidden state: α_t * h_{t-1} + (1 - α_t) * smoothed_hidden_{t-1}
        smoothed_h = alpha_t * hidden + (1 - alpha_t) * smoothed_hidden

        return h_hat, smoothed_h, alpha_t

