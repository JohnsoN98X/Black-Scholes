# 📘 Black-Scholes Pricing Engine

A robust Python library implementing the **Black-Scholes model** for European options, supporting both individual computations and high-performance vectorized operations.

---

## 📦 Modules

### 🔹 `BlackScholes`
Object-oriented engine for pricing a **single option** and computing all standard Greeks.

#### Features:
- Theoretical price calculation
- Greeks: Delta, Vega, Gamma, Theta, Rho
- Implied volatility estimation using Brent's method
- Volatility surface generation and visualization

---

### 🔹 `BlackScholesVectorized`
Vectorized and parallelized engine for computing **option prices and Greeks across large datasets** efficiently.

#### Features:
- Batch evaluation of price and Greeks
- Vectorized operations using NumPy and pandas
- Parallel computation of implied volatilities
- Time-, price- and volatility-based Greek grids

---

## 🛠 Requirements

```bash
numpy
pandas
scipy
matplotlib
joblib
```

Install the required packages with:
```bash
pip install numpy pandas scipy matplotlib joblib
```

---

## 🧪 Example: Single Option

```python
from BlackScholes import BlackScholes

bs = BlackScholes(
    St=100, K=105, T=0.5,
    sigma=0.2, r=0.01, q=0.0,
    option_type='call'
)

print(bs.evaluate())      # option price
print(bs.compute_delta()) # delta
print(bs.implied_volatility(market_price=5.5))
```

---

## 🧪 Example: Batch Computation

```python
import pandas as pd
from BlackScholesVectorized import BlackScholesVectorized

df = pd.read_csv('options.csv')  # must include columns like St, K, T, etc.

engine = BlackScholesVectorized(df)
summary = engine.summary()       # returns price + Greeks for all options
iv = engine.implied_volatility(df['market_price'].to_numpy(), n_jobs=4)
```

---

## 📊 Volatility Surface (single option)

```python
K_range = np.linspace(90, 110, 25)
T_range = np.linspace(0.01, 1, 25)

surface = bs.volatility_surface(
    market_price=5.5,
    K_range=K_range,
    T_range=T_range,
    n_jobs=4
)

bs.plot_volatility_surface()
```

---

## 📂 Project Structure

```text
src/
├── __init__.py
├── BlackScholes.py          # Single option engine
├── BlackScholesVectorized.py # Batch engine
notebooks/
├── BlackScholes.ipynb
├── BlackScholesVectorized.ipynb
README.md
```

---

## ⚖️ License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).  
© 2025 Yehonatan Zvi Dror — free to use, copy, modify, and distribute without restriction.