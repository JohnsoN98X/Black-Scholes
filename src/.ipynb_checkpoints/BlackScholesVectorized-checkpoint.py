import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
from scipy.optimize import brentq

class BlackScholesVectorized:
    def __init__(self, df, column_map=None):
        """
        Initialize the Black-Scholes vectorized engine.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing option data with required columns.
        column_map : dict, optional
            Mapping from expected column names to DataFrame column names.
            If not provided, default names are assumed.

        Raises
        ------
        ValueError
            If DataFrame is missing required columns or if any of the
            relevant input values are not strictly positive.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("'df' must be pandas DataFrame")
        if column_map is not None and not isinstance(column_map, dict):
            raise ValueError("'column_map' must be a dict")

        default_map = {
            'symbol': 'symbol',
            'St': 'St',
            'K': 'K',
            'T': 'T',
            'sigma': 'sigma',
            'r': 'r',
            'q': 'q',
            'option_type': 'option_type'
        }

        if column_map:
            default_map.update(column_map)

        missing = [col_name for key, col_name in default_map.items() if col_name not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in DataFrame: {', '.join(missing)}")

        self._data = df
        self._map = column_map
        self._names = df[default_map['symbol']].to_numpy()
        self._st = df[default_map['St']].to_numpy()
        self._k = df[default_map['K']].to_numpy()
        self._t = df[default_map['T']].to_numpy()
        self._sigma = df[default_map['sigma']].to_numpy()
        self._r = df[default_map['r']].to_numpy()
        self._q = df[default_map['q']].to_numpy()
        self._option_type = df[default_map['option_type']].to_numpy()

        if (self._st <= 0).any() or (self._k <= 0).any() or (self._t <= 0).any() or (self._sigma <= 0).any():
            raise ValueError("St, K, T, and sigma must be strictly positive.")

        self._d1 = (np.log(self._st / self._k) - (self._r - self._q + 0.5*self._sigma**2)*self._t) / (self._sigma * np.sqrt(self._t))
        self._d2 = self._d1 - self._sigma * np.sqrt(self._t)

        self._call_mask = self._option_type == 'call'
        self._put_mask = self._option_type == 'put'

    def __repr__(self):
        """
        Return a string representation of the object.

        Returns
        -------
        str
            Informative string with the number of options.
        """
        if hasattr(self, '_map'):
            return f"BlackScholesVectorized(n={len(self._data)}, map={self._map})"
        return f"BlackScholesVectorized(n={len(self._data)}"

    def evaluate(self, symbols=False):
        """
        Calculate Black-Scholes theoretical option prices.

        Parameters
        ----------
        symbols : bool, default False
            If True, return prices as a dictionary keyed by symbol.

        Returns
        -------
        numpy.ndarray or dict
            Calculated option prices for each row.
        """
        prices = np.zeros(shape=len(self._names))
        prices[self._call_mask] = (
            self._st[self._call_mask] * np.exp(-self._q[self._call_mask] * self._t[self._call_mask]) * norm.cdf(self._d1[self._call_mask]) -
            self._k[self._call_mask] * np.exp(-self._r[self._call_mask] * self._t[self._call_mask]) * norm.cdf(self._d2[self._call_mask]))
        prices[self._put_mask] = (
            -self._st[self._put_mask] * np.exp(-self._q[self._put_mask] * self._t[self._put_mask]) * norm.cdf(-self._d1[self._put_mask]) + 
            self._k[self._put_mask] * np.exp(-self._r[self._put_mask] * self._t[self._put_mask]) * norm.cdf(-self._d2[self._put_mask]))
        if symbols:
            return dict(zip(self._names, prices))
        return prices

    def compute_delta(self, symbols=False):
        """
        Compute the option delta for each row.

        Parameters
        ----------
        symbols : bool, default False
            If True, return deltas as a dictionary keyed by symbol.

        Returns
        -------
        numpy.ndarray or dict
            Option deltas.
        """
        delta = np.zeros(shape=len(self._names))
        delta[self._call_mask] = norm.cdf(self._d1[self._call_mask])
        delta[self._put_mask] = norm.cdf(self._d1[self._put_mask]) - 1
        if symbols:
            return dict(zip(self._names, delta))
        return delta

    def compute_vega(self, symbols=False):
        """
        Compute the option vega (sensitivity to volatility).

        Parameters
        ----------
        symbols : bool, default False
            If True, return vegas as a dictionary keyed by symbol.

        Returns
        -------
        numpy.ndarray or dict
            Option vegas (per 1% change in volatility).
        """
        vega = self._st * norm.pdf(self._d1) * np.sqrt(self._t)
        if symbols:
            return dict(zip(self._names, vega / 100))
        return vega / 100

    def compute_gamma(self, symbols=False):
        """
        Compute the option gamma (second derivative of option value w.r.t. price).

        Parameters
        ----------
        symbols : bool, default False
            If True, return gammas as a dictionary keyed by symbol.

        Returns
        -------
        numpy.ndarray or dict
            Option gammas.
        """
        gamma = norm.pdf(self._d1) / (self._st * self._sigma * np.sqrt(self._t))
        if symbols:
            return dict(zip(self._names, gamma))
        return gamma

    def compute_theta(self, symbols=False):
        """
        Compute the option theta (sensitivity to time decay).

        Parameters
        ----------
        symbols : bool, default False
            If True, return thetas as a dictionary keyed by symbol.

        Returns
        -------
        numpy.ndarray or dict
            Option thetas, annualized and divided by 365.
        """
        theta = np.zeros(shape=len(self._names))

        call_term1 = -(self._st[self._call_mask] * norm.pdf(self._d1[self._call_mask]) * self._sigma[self._call_mask]) / (2 * np.sqrt(self._t[self._call_mask]))
        put_term1  = -(self._st[self._put_mask]  * norm.pdf(self._d1[self._put_mask])  * self._sigma[self._put_mask])  / (2 * np.sqrt(self._t[self._put_mask]))

        call_term2 = -self._r[self._call_mask] * self._k[self._call_mask] * np.exp(-self._r[self._call_mask] * self._t[self._call_mask]) * norm.cdf(self._d2[self._call_mask])
        put_term2  =  self._r[self._put_mask]  * self._k[self._put_mask]  * np.exp(-self._r[self._put_mask]  * self._t[self._put_mask])  * norm.cdf(-self._d2[self._put_mask])

        theta[self._call_mask] = (call_term1 + call_term2) / 365
        theta[self._put_mask]  = (put_term1 + put_term2) / 365

        if symbols:
            return dict(zip(self._names, theta))
        return theta

    def compute_rho(self, symbols=False):
            """
            Compute the option rho (sensitivity to interest rate changes).
    
            Parameters
            ----------
            symbols : bool, default False
                If True, return rhos as a dictionary keyed by symbol.
    
            Returns
            -------
            numpy.ndarray or dict
                Option rhos, scaled by 1/100 (per 1% change in interest rate).
            """
            rho = np.zeros(shape=len(self._names))
            rho[self._call_mask] = self._k[self._call_mask] * self._t[self._call_mask] * np.exp(-self._r[self._call_mask] * self._t[self._call_mask]) * norm.cdf(self._d2[self._call_mask])
            rho[self._put_mask] = -self._k[self._put_mask] * self._t[self._put_mask] * np.exp(-self._r[self._put_mask] * self._t[self._put_mask]) * norm.cdf(-self._d2[self._put_mask])
            if symbols:
                return dict(zip(self._names, rho / 100))
            return rho / 100


    def summary(self, as_frame=True):
        """
        Compute a full summary of Black-Scholes outputs including price and all Greeks.

        Parameters
        ----------
        as_frame : bool, default True
            If True, return results as a pandas DataFrame; otherwise, return a dictionary.

        Returns
        -------
        pd.DataFrame or dict
            Summary of calculated values for each option.
        """
        results = dict(
            price = self.evaluate(), 
            delta = self.compute_delta(),
            vega = self.compute_vega(),
            gamma = self.compute_gamma(),
            theta = self.compute_theta(),
            rho = self.compute_rho()
        )
        if as_frame:
            frame = pd.DataFrame(columns=self._names,
                                index=results.keys(),
                                data=results.values())
            return frame
        return results

    def evaluate_over_time(self, T_range, as_frame=True):
        """
        Evaluate option prices across a range of maturities.

        Parameters
        ----------
        T_range : array-like
            List, NumPy array, or Pandas Series of time-to-maturity values.
        as_frame : bool, default True
            If True, return results as a pandas DataFrame; otherwise, return a dictionary.

        Returns
        -------
        pd.DataFrame or dict
            Option prices evaluated over different maturities.
        """
        if not isinstance(T_range, (list, np.ndarray, pd.Series)):
            raise ValueError("'T_range' must be a list, numpy ndarray, or Pandas Serie")
        if isinstance(T_range, pd.Series):
            T_range = T_range.to_numpy()
        if isinstance(T_range, list):
            T_range = np.array(T_range)
        if (T_range <= 0).any():
            raise ValueError('All time to maturity (T) must be strictly positive')
            
        results = {}
        data = self._data.copy()
        for t in T_range:
            data['T'] = t
            tmp = BlackScholesVectorized(data, self._map)
            evals = tmp.evaluate()
            results[f'{t}'] = evals
        if as_frame:
            frame = pd.DataFrame(columns = self._names,
                                index = results.keys(),
                                data = results.values())
            return frame
        return results

    def greeks_over_time(self, T_range, price=False, as_frame=True):
        """
        Compute option Greeks over a range of maturities.

        Parameters
        ----------
        T_range : array-like
            List, NumPy array, or Pandas Series of time-to-maturity values.
        price : bool, default False
            If True, include price in the result; otherwise, exclude it.
        as_frame : bool, default True
            If True, return results as a pandas DataFrame; otherwise, return a list of DataFrames.

        Returns
        -------
        pd.DataFrame or list
            Greeks (and optionally price) evaluated over different maturities.
        """
        if not isinstance(T_range, (list, np.ndarray, pd.Series)):
            raise ValueError("'T_range' must be a list, numpy ndarray, or Pandas Serie")
        if isinstance(T_range, pd.Series):
            T_range = T_range.to_numpy()
        if isinstance(T_range, list):
            T_range = np.array(T_range)
        if (T_range <= 0).any():
            raise ValueError('All time to maturity (T) must be strictly positive')
            
        results = []
        index = []
        data = self._data.copy()
        for t in T_range:
            data['T'] = t
            tmp = BlackScholesVectorized(data, self._map)
            greeks = tmp.summary()
            if not price:
                greeks = tmp.summary().drop(index='price')
                
            results.append(greeks)
            index.append(t)
            
        if as_frame:
            frame = pd.concat(results, keys=index, names=['time', 'greek'])
            return frame
        return results

    def greeks_over_price(self, S_range, price=False, as_frame=True):
        """
        Compute option Greeks over a range of the underlying asset price.
    
        Parameters
        ----------
        T_range : array-like
            List, NumPy array, or Pandas Series of time-to-maturity values.
        price : bool, default False
            If True, include price in the result; otherwise, exclude it.
        as_frame : bool, default True
            If True, return results as a pandas DataFrame; otherwise, return a list of DataFrames.
    
        Returns
        -------
        pd.DataFrame or list
            Greeks (and optionally price) evaluated over different maturities.
        """
        if not isinstance(S_range, (list, np.ndarray, pd.Series)):
            raise ValueError("'T_range' must be a list, numpy ndarray, or Pandas Serie")
        if isinstance(S_range, pd.Series):
            S_range = S_range.to_numpy()
        if isinstance(S_range, list):
            S_range = np.array(S_range)
        if (S_range <= 0).any():
            raise ValueError('All time to maturity (T) must be strictly positive')
            
        results = []
        index = []
        data = self._data.copy()
        for s in S_range:
            data['St'] = s
            tmp = BlackScholesVectorized(data, self._map)
            greeks = tmp.summary()
            if not price:
                greeks = tmp.summary().drop(index='price')
                
            results.append(greeks)
            index.append(s)
            
        if as_frame:
            frame = pd.concat(results, keys=index, names=['St', 'greek'])
            return frame
        return results


    def greeks_over_volatility(self, V_range, price=False, as_frame=True):
        """
        Compute option Greeks over a range of volatility values.

        Parameters
        ----------
        V_range : array-like
            List, NumPy array, or Pandas Series of volatility values.
        price : bool, default False
            If True, include price in the result; otherwise, exclude it.
        as_frame : bool, default True
            If True, return results as a pandas DataFrame; otherwise, return a list of DataFrames.

        Returns
        -------
        pd.DataFrame or list
            Greeks (and optionally price) evaluated over different volatilities.
        """
        if not isinstance(V_range, (list, np.ndarray, pd.Series)):
            raise ValueError("'V_range' must be a list, numpy ndarray, or Pandas Serie")
        if isinstance(V_range, pd.Series):
            V_range = V_range.to_numpy()
        if isinstance(V_range, list):
            V_range = np.array(V_range)
        if (V_range <= 0).any():
            raise ValueError('All volatility value (V) must be strictly positive')
            
        data = self._data.copy()
        results = []
        index = []
        for v in V_range:
            data['sigma'] = v
            tmp = BlackScholesVectorized(data, self._map)
            greeks = tmp.summary()
            if not price:
                greeks = greeks.drop(index='price')
            results.append(greeks)
            index.append(v)
        if as_frame:
            frame = pd.concat(results, keys=index, names=['time', 'greek'])
            return frame
        return results

    def implied_volatility(self, market_price, n_jobs=1):
        """
        Compute the implied volatility for each option using Brent's method.

        Parameters
        ----------
        market_price : np.ndarray
            Array of observed market prices for the options.
        n_jobs : int, default 1
            Number of parallel jobs to use. If 1, runs sequentially.

        Returns
        -------
        np.ndarray
            Implied volatilities for each option.
        """
        if not isinstance(market_price, np.ndarray):
            raise ValueError("'market_price' must be a numpy array")
        if not len(market_price)==len(self._data):
            raise ValueError(
                    "Input validation error: The number of market prices must match the number of options in the DataFrame. "
                    f"Received {len(market_price)} market prices and {len(self._data)} options."
                )

        data = self._data.copy()
        def select_row(i):
            row = data.iloc[i].copy()
            def objective(sigma):
                if sigma <= 0:
                    return 1e10
                row['sigma'] = sigma
                tmp = BlackScholesVectorized(pd.DataFrame([row]), column_map=self._map)
                return tmp.evaluate()[0] - market_price[i]

            try:
                return brentq(objective, 1e-5, 2, maxiter=500)
            except (ValueError, RuntimeError):
                return np.nan

        if n_jobs==1:
            ivs = [select_row(i) for i in range(len(self._data))]
        else:
            from joblib import Parallel, delayed
            ivs = Parallel(n_jobs=n_jobs)(delayed(select_row)(i) for i in range(len(self._data)))
                
        return np.array(ivs)

    @property
    def d1(self):
        """
        Return d1 values as a dictionary keyed by symbol.

        Returns
        -------
        dict
            Mapping of option symbol to d1 value.
        """
        return dict(zip(self._names, self._d1))

    @property
    def d2(self):
        """
        Return d2 values as a dictionary keyed by symbol.

        Returns
        -------
        dict
            Mapping of option symbol to d2 value.
        """
        return dict(zip(self._names, self._d2))

    @property
    def symbols(self):
        """
        Return the list of option symbols.

        Returns
        -------
        np.ndarray
            Array of option symbols.
        """
        return self._names