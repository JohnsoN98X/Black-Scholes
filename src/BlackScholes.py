import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class BlackScholes:
    """
    Black-Scholes option pricing model for European options.

    Parameters
    ----------
    St : float
        Current price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to maturity (in years).
    sigma : float
        Volatility of the underlying asset (annualized).
    r : float
        Risk-free interest rate (annualized).
    q : float
        Continuous dividend yield (annualized).
    option_type : str
        Type of the option, either 'call' or 'put'.
    """

    def __init__(self, St, K, T, sigma, r, q, option_type):
        if option_type not in ['call', 'put']:
            raise ValueError('Option type must be "call" or "put".')
        if any(x <= 0 for x in [St, K, T, sigma]):
            raise ValueError("St, K, T, and sigma must be strictly positive.")

        self.option_type = option_type
        self.st = St
        self.k = K
        self.t = T
        self.sigma = sigma
        self.r = r
        self.q = q
        self._d1 = (np.log(self.st / self.k) + (self.r - self.q + self.sigma ** 2 / 2) * self.t) / (self.sigma * np.sqrt(self.t))
        self._d2 = self._d1 - self.sigma * np.sqrt(self.t)

    def __repr__(self):
        """Machine-readable string representation of the object."""
        cls = self.__class__.__name__
        return (f"{cls}("
                f"St={self.st}, "
                f"K={self.k}, "
                f"T={self.t}, "
                f"sigma={self.sigma}, "
                f"r={self.r}, "
                f"q={self.q}, "
                f"option_type='{self.option_type}')")

    def evaluate(self):
        """
        Compute the theoretical price of the option using the Black-Scholes formula.

        Returns
        -------
        float
            Theoretical option price.
        """
        if self.option_type == 'call':
            return self.st * np.exp(-self.q * self.t) * norm.cdf(self._d1) - self.k * np.exp(-self.r * self.t) * norm.cdf(self._d2)
        else:
            return -self.st * np.exp(-self.q * self.t) * norm.cdf(-self._d1) + self.k * np.exp(-self.r * self.t) * norm.cdf(-self._d2)

    def compute_delta(self):
        """
        Calculate the option Delta.

        Returns
        -------
        float
            Sensitivity of option price to changes in the underlying asset price.
        """
        return norm.cdf(self._d1) if self.option_type == 'call' else norm.cdf(self._d1) - 1

    def compute_vega(self):
        """
        Calculate the option Vega.

        Returns
        -------
        float
            Sensitivity of option price to changes in volatility (per 1% change, per day).
        """
        return (self.st * norm.pdf(self._d1) * np.sqrt(self.t)) / 365

    def compute_gamma(self):
        """
        Calculate the option Gamma.

        Returns
        -------
        float
            Second derivative of option price with respect to the underlying asset.
        """
        return norm.pdf(self._d1) / (self.st * self.sigma * np.sqrt(self.t))

    def compute_theta(self):
        """
        Calculate the option Theta.

        Returns
        -------
        float
            Sensitivity of option price to time decay (per day).
        """
        term1 = -(self.st * norm.pdf(self._d1) * self.sigma) / (2 * np.sqrt(self.t))
        if self.option_type == 'call':
            term2 = -self.r * self.k * np.exp(-self.r * self.t) * norm.cdf(self._d2)
        else:
            term2 = self.r * self.k * np.exp(-self.r * self.t) * norm.cdf(-self._d2)
        return (term1 + term2) / 365

    def compute_rho(self):
        """
        Calculate the option Rho.

        Returns
        -------
        float
            Sensitivity of option price to changes in the risk-free interest rate (per 1% change).
        """
        if self.option_type == 'call':
            return (self.k * self.t * np.exp(-self.r * self.t) * norm.cdf(self._d2)) / 100
        else:
            return (-self.k * self.t * np.exp(-self.r * self.t) * norm.cdf(-self._d2)) / 100

    def summary(self, as_frame=False):
        """
        Return a summary of all option Greeks and theoretical price.

        Parameters
        ----------
        as_frame : bool, default=False
            If True, returns result as a Pandas DataFrame.

        Returns
        -------
        dict or pd.DataFrame
            Option price and Greeks.
        """
        summary = {
            'option price': self.evaluate(),
            'delta': self.compute_delta(),
            'vega': self.compute_vega(),
            'theta': self.compute_theta(),
            'gamma': self.compute_gamma(),
            'rho': self.compute_rho()
        }
        if as_frame:
            return pd.DataFrame(data=summary.values(), index=summary.keys(), columns=['value'])
        return summary

    def greeks_over_price(self, S_range):
        """
        Evaluate Greeks for a range of underlying asset prices.

        Parameters
        ----------
        S_range : array-like
            Sequence of underlying asset prices.

        Returns
        -------
        pd.DataFrame
            DataFrame of Greeks across underlying prices.
        """
        if np.any(np.array(S_range) <= 0):
            raise ValueError("All price (S) values must be strictly positive")
        results = []
        for S in S_range:
            tmp = BlackScholes(S, self.k, self.t, self.sigma, self.r, self.q, self.option_type)
            results.append(tmp.summary())
        return pd.DataFrame(results, index=S_range)

    def greeks_over_time(self, T_range):
        """
        Evaluate Greeks over a range of times to maturity.

        Parameters
        ----------
        T_range : array-like
            Sequence of times to maturity.

        Returns
        -------
        pd.DataFrame
            DataFrame of Greeks across maturities.
        """
        if np.any(np.array(T_range) <= 0):
            raise ValueError("All time values must be strictly positive")
        results = []
        for T in T_range:
            tmp = BlackScholes(self.st, self.k, T, self.sigma, self.r, self.q, self.option_type)
            results.append(tmp.summary())
        return pd.DataFrame(results, index=T_range)

    def greeks_over_volatility(self, V_range):
        """
        Evaluate Greeks over a range of volatilities.

        Parameters
        ----------
        V_range : array-like
            Sequence of volatility values.

        Returns
        -------
        pd.DataFrame
            DataFrame of Greeks across volatility values.
        """
        if np.any(np.array(V_range) <= 0):
            raise ValueError("All volatility values must be strictly positive")
        results = []
        for v in V_range:
            tmp = BlackScholes(self.st, self.k, self.t, v, self.r, self.q, self.option_type)
            results.append(tmp.summary())
        return pd.DataFrame(results, index=V_range)

    def implied_volatility(self, market_price):
        """
        Estimate the implied volatility from a given market price.

        Parameters
        ----------
        market_price : float
            Observed market price of the option.

        Returns
        -------
        float
            Implied volatility (annualized).
        """
        def objective(sigma):
            if sigma <= 0:
                return 1e10
            tmp = BlackScholes(self.st, self.k, self.t, sigma, self.r, self.q, self.option_type)
            return tmp.evaluate() - market_price

        try:
            return brentq(objective, 1e-6, 5.0)
        except ValueError:
            raise ValueError("Could not find implied volatility in the specified range.")

    def volatility_surface(self, market_price, K_range, T_range, n_jobs=1):
        """
        Construct an implied volatility surface over strike and maturity.

        Parameters
        ----------
        market_price : float
            Observed market price.
        K_range : array-like
            Sequence of strike prices.
        T_range : array-like
            Sequence of maturities.
        n_jobs : int, default=1
            Number of parallel jobs.

        Returns
        -------
        np.ndarray
            Matrix of implied volatilities.
        """
        self._krange = K_range
        self._trange = T_range
        param_grid = [(k, t) for k in K_range for t in T_range]

        def compute_iv(k, t):
            try:
                bs = BlackScholes(self.st, k, t, 0.2, self.r, self.q, self.option_type)
                return bs.implied_volatility(market_price)
            except:
                return np.nan

        results = Parallel(n_jobs=n_jobs)(delayed(compute_iv)(k, t) for k, t in param_grid)
        self._surface = np.array(results).reshape(len(K_range), len(T_range))
        return self._surface

    def plot_volatility_surface(self, elev=10, azim=30, cmap='OrRd'):
        """
        Plot the implied volatility surface.

        Parameters
        ----------
        elev : int, default=10
            Elevation angle for 3D plot.
        azim : int, default=30
            Azimuth angle for 3D plot.
        cmap : str, default='OrRd'
            Colormap used for the surface.
        """
        if not hasattr(self, '_surface'):
            raise KeyError("Volatility surface does not exist. Please call 'volatility_surface' first.")
        K, T = np.meshgrid(self._krange, self._trange, indexing='ij')
        fig = plt.figure(figsize=(18, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(K, T, self._surface, cmap=cmap)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('Strike')
        ax.set_ylabel('Time to Maturity')
        ax.set_zlabel('Implied Volatility')
        plt.title('Implied Volatility Surface')
        plt.tight_layout()
        plt.show()

    @property
    def d1(self):
        """Return the Black-Scholes d1 term."""
        return self._d1

    @property
    def d2(self):
        """Return the Black-Scholes d2 term."""
        return self._d2
