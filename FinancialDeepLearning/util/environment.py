import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random

def seed_anchor(random_seed : int = 0) -> None:
    '''
    Anchored python random seed
    :param random_seed: set random seed, ex : 42
    :return: None
    '''
    random.seed(random_seed) # python random library, for torchvision
    np.random.seed(random_seed) # numpy random library
    torch.manual_seed(random_seed) # torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    return None