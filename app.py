import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs_functions import black_scholes, option_greeks

def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    /* Adjust padding and margins */
    .reportview-container .main .block-container{
        padding-top: 1rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 1rem;
    }

    /* Adjust font sizes */
    h1 {
        font-size: 2.5rem;
    }
    h2 {
        font-size: 2rem;
    }
    h3 {
        font-size: 1.75rem;
    }

    /* Adjust metric labels and values */
    .metric-label {
        font-size: 1.5rem;
    }
    .metric-value {
        font-size: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title and LinkedIn Icon
    st.title("Black-Scholes Option Pricer")

    linkedin_url = "https://www.linkedin.com/in/ryan-tandiono-331584207/"
    linkedin_icon_url = 'https://cdn-icons-png.flaticon.com/512/174/174857.png'

    st.sidebar.markdown(
        f"""
        <a href="{linkedin_url}" target="_blank">
            <img src="{linkedin_icon_url}" alt="LinkedIn" width="30" height="30">
        </a>
        """,
        unsafe_allow_html=True
    )

    # Sidebar inputs
    with st.sidebar:
        st.header("Option Parameters")
        S = st.number_input("Underlying Price", value=100.0, min_value=0.01)
        K = st.number_input("Strike Price", value=100.0, min_value=0.01)
        T = st.number_input("Time to Expiry (Years)", value=1.0, min_value=0.01)
        r = st.number_input("Risk-Free Interest Rate", value=0.05, min_value=0.0, max_value=1.0)
        sigma = st.number_input("Volatility (σ)", value=0.2, min_value=0.0, max_value=1.0)
        option_type = st.selectbox("Option Type", ("Call", "Put"))

    # Calculate option price and Greeks
    option_price = black_scholes(S, K, T, r, sigma, option_type)
    delta, gamma, theta, vega = option_greeks(S, K, T, r, sigma, option_type)

    # Display option price and Greeks
    st.markdown('## Option Price and Greeks')
    st.markdown('### Option Price')
    st.metric(label=f"{option_type} Option Price", value=f"${option_price:.2f}")

    st.markdown('### Option Greeks')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Delta", value=f"{delta:.2f}")
    with col2:
        st.metric(label="Gamma", value=f"{gamma:.4f}")
    with col3:
        st.metric(label="Theta", value=f"{theta:.2f}")
    with col4:
        st.metric(label="Vega", value=f"{vega:.4f}")

    st.markdown("""
    ### Understanding the Greeks
    - **Delta** measures the sensitivity of the option price to changes in the underlying asset's price.
    - **Gamma** measures the rate of change of Delta with respect to changes in the underlying price.
    - **Theta** represents the rate of decline of the option's value with time (time decay).
    - **Vega** measures the sensitivity to volatility.
    """)

    # Sensitivity Analysis
    st.markdown("## Sensitivity Analysis")
    st.markdown("### Option Price Heatmap")

    S_min = st.number_input("Minimum Underlying Price", value=S * 0.5, min_value=0.01)
    S_max = st.number_input("Maximum Underlying Price", value=S * 1.5, min_value=S)
    sigma_min = st.number_input("Minimum Volatility", value=0.1, min_value=0.0, max_value=sigma)
    sigma_max = st.number_input("Maximum Volatility", value=0.5, min_value=sigma, max_value=1.0)

    if S_min >= S_max or sigma_min >= sigma_max:
        st.error("Minimum values must be less than maximum values.")
        return

    # Generate values for S and sigma
    S_values = np.linspace(S_min, S_max, 10)
    sigma_values = np.linspace(sigma_min, sigma_max, 10)

    # Create meshgrid
    S_grid, sigma_grid = np.meshgrid(S_values, sigma_values)

    # Calculate option prices over the grid
    option_prices = black_scholes(S_grid, K, T, r, sigma_grid, option_type)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(option_prices,
                xticklabels=np.round(S_values, 2),
                yticklabels=np.round(sigma_values, 2),
                cmap='viridis',
                ax=ax,
                annot=True,
                fmt=".2f",
                annot_kws={"size": 7})

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10)

    ax.set_xlabel("Underlying Price (S)", fontsize=12)
    ax.set_ylabel("Volatility (σ)", fontsize=12)
    ax.set_title(f"{option_type} Option Price Heatmap", fontsize=14)
    fig.tight_layout()
    st.pyplot(fig)

    # Plot Greeks
    st.markdown("### Option Greeks vs Underlying Price")
    S_plot = np.linspace(S_min, S_max, 100)
    delta_p, gamma_p, theta_p, vega_p = option_greeks(S_plot, K, T, r, sigma, option_type)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    # Delta
    axs[0].plot(S_plot, delta_p, color='cyan')
    axs[0].set_title('Delta vs Underlying Price', fontsize=12)
    axs[0].set_xlabel('Underlying Price (S)', fontsize=10)
    axs[0].set_ylabel('Delta', fontsize=10)
    axs[0].grid(True)

    # Gamma
    axs[1].plot(S_plot, gamma_p, color='magenta')
    axs[1].set_title('Gamma vs Underlying Price', fontsize=12)
    axs[1].set_xlabel('Underlying Price (S)', fontsize=10)
    axs[1].set_ylabel('Gamma', fontsize=10)
    axs[1].grid(True)

    # Theta
    axs[2].plot(S_plot, theta_p, color='yellow')
    axs[2].set_title('Theta vs Underlying Price', fontsize=12)
    axs[2].set_xlabel('Underlying Price (S)', fontsize=10)
    axs[2].set_ylabel('Theta', fontsize=10)
    axs[2].grid(True)

    # Vega
    axs[3].plot(S_plot, vega_p, color='green')
    axs[3].set_title('Vega vs Underlying Price', fontsize=12)
    axs[3].set_xlabel('Underlying Price (S)', fontsize=10)
    axs[3].set_ylabel('Vega', fontsize=10)
    axs[3].grid(True)

    fig.tight_layout(pad=3.0)
    st.pyplot(fig)

    # Footer
    st.markdown("""
    ---
    *Developed by Ryan Tandiono. For educational purposes only.*
    """)

if __name__ == "__main__":
    main()
