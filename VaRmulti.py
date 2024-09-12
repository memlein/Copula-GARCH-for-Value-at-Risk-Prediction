import pandas as pd 
import numpy as np 
import yfinance as yf
import matplotlib.pyplot as plt 
import seaborn as sns; sns.set()
import pyvinecopulib as pv
import statsmodels.api as sm
import statsmodels.formula.api as smf
from arch import arch_model
from statsmodels.distributions.copula.api import GaussianCopula, StudentTCopula
from multiprocessing import Pool

class VaR:
    def __init__(self, tickers: list, start: str, end: str, VaR_level: float, window: int) -> None:
        self.tickers = tickers
        self.start = start
        self.end = end
        self.log_returns = self._get_logarithmic_returns(
            tickers=self.tickers, start=self.start, end=self.end
        )
        self.n_assets = self.log_returns.shape[1]
        self.VaR_level = VaR_level
        self.window = window
        self.VaR_forecast = None
        
    def _get_logarithmic_returns(
        self, tickers: list, start: str, end: str
    ) -> pd.DataFrame:
        data = yf.download(tickers=tickers, start=start, end=end)["Adj Close"]
        data = pd.DataFrame(
            np.diff(np.log(data[1:]), axis=0) * 100, columns=data.columns
        )
        data = data.dropna(axis=0)
        return data
    
    
    def _predict_next_period(self, data):
        garch_models = {}
        for column in data.columns:
            garch_models[column] = dict()
            am = arch_model(data[column], vol="Garch", p=1, o=0, q=1, dist="t", mean="constant").fit(disp='off')
            z = am.resid/am.conditional_volatility 
            u = am.model.distribution.cdf(z, parameters=am.params['nu'])
            garch_models[column]['arch_model'] = am
            garch_models[column]['z'] = z
            garch_models[column]['u'] = u

        u_data = pd.DataFrame()
        for ticker in data.columns:
            u_series = pd.Series(garch_models[ticker]['u'], name=ticker)
            u_data = pd.concat([u_data, u_series], axis=1)

        copula = StudentTCopula(k_dim=data.shape[1], df=3)
        copula.fit_corr_param(u_data)
        u_sim = pd.DataFrame(copula.rvs(nobs=1000), columns=data.columns)

        for ticker in garch_models.keys():
            z_sim = garch_models[ticker]['arch_model'].model.distribution.ppf(u_sim[ticker], parameters=am.params['nu'])
            garch_models[ticker]['z_sim'] = z_sim
            params = garch_models[ticker]['arch_model'].params
            sigma_next_period = np.sqrt(params['omega'] + params['alpha[1]'] * np.square(data[ticker].iloc[-1]) + params['beta[1]'] * np.square(garch_models[ticker]['arch_model']._volatility[-1]))
            next_period_returns = pd.Series(params['mu'] + sigma_next_period * garch_models[ticker]['z_sim'], name=ticker).sort_values()
            garch_models[ticker]['sigma_next_period'] = sigma_next_period
            garch_models[ticker]['next_period_returns'] = next_period_returns

        VaR_pred = np.mean([v['next_period_returns'].quantile(self.VaR_level) for v in garch_models.values()])
        
        return VaR_pred
    def _get_windows(self, df, window_size):
        list_of_windows = []
        for i in range(df.shape[0]):
            window = df.iloc[i:i + window_size, :]
            list_of_windows.append(window)
            if i == df.shape[0] - window_size:
                break

        return list_of_windows

 
    def calculate_VaR(self):
        result = pd.DataFrame(columns=['Portfolio', 'VaR'])
        with Pool() as pool:
            # issue tasks and process results
            print('pool started')
            list_of_windows = self._get_windows(df=self.log_returns, window_size=self.window)
            for result in pool.map(self._predict_next_period, list_of_windows):
                print(f'>got {result}')


        # for i in range(0, self.log_returns.shape[0] - self.window + 1):
        #     start = i
        #     end = i + self.window
        #     temp_data = self.log_returns.iloc[start:end]
        #     res = self._predict_next_period(data=temp_data)
        #     if end < self.log_returns.shape[0] - 1: 
        #         result.loc[end+1, 'VaR'] = res
        #         result.loc[end+1, 'Portfolio'] = np.mean(self.log_returns.iloc[end+1,:])
        
        print('VaR forecast calculated')
        self.VaR_forecast = result

if __name__ == '__main__':
    VaRtest = VaR(tickers=["AAPL", "GOOG", "BAS.DE", "BMW.DE"], start = "2010-01-01", end = "2011-12-31", VaR_level=0.05, window=1000)
    VaRtest.calculate_VaR()