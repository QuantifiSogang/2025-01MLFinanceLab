import numpy as np
import pandas as pd

# def get_test_data(
#         n_features: int = 40,
#         n_informative: int = 10,
#         n_redundant: int = 10,
#         n_samples: int = 10000,
#         random_state: int = 42,
#         sigma_std: float = 0.0
# ):
#     from sklearn.datasets import make_classification
#
#     np.random.seed(random_state)
#
#     trnsX, cont = make_classification(
#         n_samples=n_samples,
#         n_features=n_features,
#         n_informative=n_informative,
#         n_redundant=n_redundant,
#         random_state=random_state,
#         shuffle=False
#     )
#
#     df0_index = pd.date_range(start=pd.to_datetime('today') - pd.to_timedelta(n_samples, unit='d'),
#                               periods=n_samples, freq='B')
#
#     trnsX, cont = pd.DataFrame(trnsX, index=df0_index), pd.Series(cont, index=df0_index).to_frame('bin')
#
#     df0 = ['I_' + str(i) for i in range(n_informative)] + ['R_' + str(i) for i in range(n_redundant)]
#     df0 += ['N_' + str(i) for i in range(n_features - len(df0))]
#
#     trnsX.columns = df0
#     cont['w'] = 1. / cont.shape[0]
#     cont['t1'] = pd.Series(cont.index, index=cont.index)
#
#     return trnsX, cont

def get_test_data(
        n_features: int = 40,
        n_informative: int = 10,
        n_redundant: int = 10,
        n_samples: int = 10000,
        random_state: int = 42,
        sigma_std: float = 0.0
):
    from sklearn.datasets import make_classification

    np.random.seed(random_state)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features-n_redundant,
        n_informative=n_informative,
        n_redundant=0,
        random_state=random_state,
        shuffle=False
    )

    cols = ['I_'+str(i) for i in range(n_informative)]
    cols += ['N_'+str(i) for i in range(n_features - n_informative - n_redundant)]

    X, y = pd.DataFrame(X, columns = cols), pd.Series(y)
    i = np.random.choice(range(n_informative), size = n_redundant)

    for k,j in enumerate(i):
        X['R_'+str(k)] = X['I_'+str(j)] + np.random.normal(size = X.shape[0]) * sigma_std

    X.index = pd.date_range(start=pd.to_datetime('today') - pd.to_timedelta(n_samples, unit='d'), periods=n_samples, freq='B')
    y = y.to_frame('bin')
    y.index = X.index

    y['w'] = 1 / y.shape[0]
    y['t1'] = pd.Series(y.index, index=y.index)

    return X, y