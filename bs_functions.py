import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type='Call'):
    """
    Calculate European option price using the Black-Scholes Model.

    Parameters:
    - S: Current price of the underlying asset
    - K: Strike price
    - T: Time to expiration in years
    - r: Annual risk-free interest rate
    - sigma: Annualized volatility
    - option_type: 'Call' or 'Put'

    Returns:
    - option_price: Calculated option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'Call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'Put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'Call' or 'Put'.")

    return option_price

def option_greeks(S, K, T, r, sigma, option_type='Call'):
    """
    Calculate the Greeks for a European option using the Black-Scholes Model.

    Parameters:
    - S: Current price of the underlying asset
    - K: Strike price
    - T: Time to expiration in years
    - r: Annual risk-free interest rate
    - sigma: Annualized volatility
    - option_type: 'Call' or 'Put'

    Returns:
    - delta, gamma, theta, vega: Calculated Greeks
    """
    # Convert inputs to numpy arrays for vectorization
    S = np.array(S, dtype=float)
    K = np.array(K, dtype=float)
    T = np.array(T, dtype=float)
    r = np.array(r, dtype=float)
    sigma = np.array(sigma, dtype=float)

    # Avoid division by zero
    sigma = np.where(sigma == 0, 1e-10, sigma)
    T = np.where(T == 0, 1e-10, T)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate Greeks
    if option_type == 'Call':
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) \
                - r * K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'Put':
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) \
                + r * K * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'Call' or 'Put'.")

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)

    return delta, gamma, theta, vega
