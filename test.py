import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
from statsmodels.distributions.copula.api import StudentTCopula
import multiprocessing as mp

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
            z = am.resid / am.conditional_volatility 
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
            sigma_next_period = np.sqrt(
                params['omega'] + 
                params['alpha[1]'] * np.square(data[ticker].iloc[-1]) + 
                params['beta[1]'] * np.square(garch_models[ticker]['arch_model']._volatility[-1])
            )
            next_period_returns = pd.Series(
                params['mu'] + sigma_next_period * garch_models[ticker]['z_sim'], 
                name=ticker
            ).sort_values()
            garch_models[ticker]['sigma_next_period'] = sigma_next_period
            garch_models[ticker]['next_period_returns'] = next_period_returns

        VaR_pred = np.mean([v['next_period_returns'].quantile(self.VaR_level) for v in garch_models.values()])
        
        return VaR_pred
    
    def _calculate_VaR_for_window(self, i):
        """
        Calculate VaR for a single rolling window.
        This function is used for parallel execution.
        """
        start = i
        end = i + self.window
        temp_data = self.log_returns.iloc[start:end]
        res = self._predict_next_period(data=temp_data)
        portfolio_return = np.mean(self.log_returns.iloc[end+1, :]) if end < self.log_returns.shape[0] - 1 else None
        return end + 1, res, portfolio_return
    
    def calculate_VaR(self):
        result = pd.DataFrame(columns=['Portfolio', 'VaR'])

        # Get the range for windows
        window_range = range(0, self.log_returns.shape[0] - self.window + 1)

        # Use multiprocessing Pool for parallel execution
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(self._calculate_VaR_for_window, window_range)

        # Collect results
        for res in results:
            end, VaR_value, portfolio_return = res
            if end is not None:
                result.loc[end, 'VaR'] = VaR_value
                result.loc[end, 'Portfolio'] = portfolio_return

        self.VaR_forecast = result

if __name__ == '__main__':
    VaRtest = VaR(tickers=["AAPL", "GOOG", "BAS.DE", "BMW.DE", "RWE.DE", "DBK.DE"], start = "2010-01-01", end = "2018-12-31", VaR_level=0.05, window=1000)
    VaRtest.calculate_VaR()