import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import Summary
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ARFIMAResults():
    def __init__(self, model, p, d, q, data):
        self.model = model
        self.data = data
        self.p = p
        self.d = d
        self.q = q
        self.fittedvalues = self.data
        self.resid = model.resid
        self.params = model.params

    def _hessian_opg(self, params, **kwargs):
        return np.eye(len(params))

    def summary(self):
        smry = Summary()

        hetero_res = self.model.test_heteroskedasticity(method=None)
        jb = self.model.test_normality(method='jarquebera')
        lb = self.model.test_serial_correlation(method='ljungbox')

        model_info = [
            ('Dep. Variable:', 'predicted'),
            ('Model:', f'ARFIMA({self.p}, {self.d}, {self.q})'),
            ('Date:', pd.Timestamp.now().strftime('%a, %d %b %Y')),
            ('Time:', pd.Timestamp.now().strftime('%H:%M:%S')),
            ('Covariance Type:', 'opg'),
            ('Sample:', f'{self.data.index[0].strftime("%m-%d-%Y")} - {self.data.index[-1].strftime("%m-%d-%Y")}'),
            ('No. Observations:', f"{len(self.model.model.endog)}"),

            ('Log Likelihood', f"{self.model.llf:.3f}"),

            ('AIC', f"{self.model.aic:.3f}"),

            ('BIC', f"{self.model.bic:.3f}"),
            ('HQIC', f"{self.model.hqic:.3f}"),

        ]

        smry.add_dict(dict(model_info), ncols=2)

        params_data = pd.DataFrame({
            "coef": self.params,
            "std err": self.model.bse,
            "t": self.model.tvalues,
            "P>|t|": self.model.pvalues,
            "[0.025": self.model.conf_int().iloc[:, 0],
            "0.975]": self.model.conf_int().iloc[:, 1]
        })

        smry.add_df(params_data)  # , title="Parameter Estimates")

        additional_info = [
            ('Ljung-Box (L1) (Q):', f"{lb[0][0][0]:.2f}"),
            ('Prob(Q):', f"{lb[0][1][0]:.2f}"),
            ('Heteroskedasticity (H):', f"{hetero_res[0][0]:.2f}"),
            ('Prob(H) (two-sided):', f"{hetero_res[0][1]:.2f}"),
            ('Jarque-Bera (JB):', f"{jb[0][0]:.2f}"),
            ('Prob(JB):', f"{jb[0][1]:.2f}"),
            ('Skew:', f"{jb[0][2]:.2f}"),
            ('Kurtosis:', f"{jb[0][3]:.2f}"),
        ]

        smry.add_dict(dict(additional_info))

        smry.title = 'ARFIMA Results'

        return smry


class ARFIMA:
    def __init__(self, data, order: tuple):
        if isinstance(data, pd.Series):
            data = data.to_frame()
        self.data = data
        self.p = order[0]
        self.d = order[1]
        self.q = order[2]
        self.diff_series = self.fracDiff_FFD(data)

    def fracDiff_FFD(self, series, thres=1e-5):
        if isinstance(series, pd.Series):
            series = series.to_frame()
        w = self.getWeights_FFD(thres)
        width = len(w) - 1
        df = {}
        for name in series.columns:
            seriesF = series[[name]].ffill().dropna()
            df_ = pd.Series(dtype=float)
            for iloc1 in range(width, seriesF.shape[0]):
                loc0 = seriesF.index[iloc1 - width]
                loc1 = seriesF.index[iloc1]
                if not np.isfinite(series.loc[loc1, name]):
                    continue
                df_.loc[loc1] = np.dot(w.T, seriesF.loc[loc0: loc1])[0, 0]
            df[name] = df_.copy(deep=True)
        df = pd.concat(df, axis=1)
        return df

    def getWeights_FFD(self, thres):
        w = [1.]
        k = 1
        while abs(w[-1]) >= thres:
            w_ = -w[-1] / k * (self.d - k + 1)
            w.append(w_)
            k += 1
        w = np.array(w[::-1]).reshape(-1, 1)[1:]
        return w

    def fit(self):
        self.diff_series, self.data = self.diff_series.align(self.data, join='inner', axis=0)
        model = sm.tsa.statespace.SARIMAX(
            self.data,
            order=(self.p, 0, self.q)
        ).fit()
        res = ARFIMAResults(
            model=model,
            p=self.p,
            d=self.d,
            q=self.q,
            data=self.diff_series,
        )
        return res

def getWeights(d, size):
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[:: -1]).reshape(-1, 1)
    return w

def fracDiff(series, d, thres=.01):
    w = getWeights(d, series.shape[0])
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thres].shape[0]
    df = {}
    for name in series.columns:
        seriesF = series[[name]].ffill().dropna()
        df_ = pd.Series(dtype = float)
        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc]

            test_val = series.loc[loc, name]
            if isinstance(test_val, (pd.Series, pd.DataFrame)):
                test_val = test_val.resample('1m').mean()
            if not np.isfinite(test_val).any():
                continue
            try:
                df_.loc[loc] = np.dot(w[-(iloc + 1):, :].T, seriesF.loc[:loc])[0, 0]
            except:
                continue
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


def calculate_integration(
        data,
        col_name='Close',
        progress=True
):
    cols = ['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr']
    out = pd.DataFrame(columns=cols)
    if progress == True:
        for d in tqdm(np.linspace(0, 1, 21)):
            try:
                df1 = data[[col_name]]
                df2 = fracDiff(df1, d=d, thres=1e-5)
                corr = np.corrcoef(df1.loc[df2.index, col_name], df2[col_name])[0, 1]
                df2 = sm.tsa.stattools.adfuller(df2[col_name], maxlag=1, regression='c', autolag=None)
                out.loc[round(d, 2)] = list(df2[:4]) + [df2[4]['5%']] + [corr]
            except Exception as e:
                continue
        out.index.name = 'd'
    else:
        for d in tqdm(np.linspace(0, 1, 21)):
            try:
                df1 = data[[col_name]]
                df2 = fracDiff(df1, d=d, thres=1e-5)
                corr = np.corrcoef(df1.loc[df2.index, col_name], df2[col_name])[0, 1]
                df2 = sm.tsa.stattools.adfuller(df2[col_name], maxlag=1, regression='c', autolag=None)
                out.loc[round(d, 2)] = list(df2[:4]) + [df2[4]['5%']] + [corr]
            except Exception as e:
                continue
        out.index.name = 'd'

    return out


def get_optimal_integration(data, col_name='Close', progress=True):
    cols = ['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr']
    out = pd.DataFrame(columns=cols)
    if progress == True:
        for d in tqdm(np.linspace(0, 1, 21)):
            try:
                df1 = data[[col_name]]
                df2 = fracDiff(df1, d=d, thres=1e-5)
                corr = np.corrcoef(df1.loc[df2.index, col_name], df2[col_name])[0, 1]
                df2 = sm.tsa.stattools.adfuller(df2[col_name], maxlag=1, regression='c', autolag=None)
                out.loc[round(d, 2)] = list(df2[:4]) + [df2[4]['5%']] + [corr]
            except Exception as e:
                continue
        out.index.name = 'd'
    else:
        for d in tqdm(np.linspace(0, 1, 21)):
            try:
                df1 = data[[col_name]]
                df2 = fracDiff(df1, d=d, thres=1e-5)
                corr = np.corrcoef(df1.loc[df2.index, col_name], df2[col_name])[0, 1]
                df2 = sm.tsa.stattools.adfuller(df2[col_name], maxlag=1, regression='c', autolag=None)
                out.loc[round(d, 2)] = list(df2[:4]) + [df2[4]['5%']] + [corr]
            except Exception as e:
                continue
        out.index.name = 'd'
    return {
        'optimal_d': out[out['pVal'] < 0.05].iloc[0].name,
        'adf_stats': out[out['pVal'] < 0.05].iloc[0]['adfStat'],
        'p_value': out[out['pVal'] < 0.05].iloc[0]['pVal'],
        '95% conf': out[out['pVal'] < 0.05].iloc[0]['95% conf'],
        'correlation': out[out['pVal'] < 0.05].iloc[0]['corr']
    }