# Adaptive Volatility Modeling with HMM & GARCH

A quantitative trading strategy that uses **Hidden Markov Models** for regime detection and **GARCH** for dynamic position sizing. This project demonstrates a complete quantitative research workflow, from raw data processing to real-world constraint modeling.

## Key Features
* **Regime Detection:** 3-state Gaussian HMM (Baum-Welch & Viterbi) to identify Bull, Bear, and Sideways markets.
* **Risk Management:** Dynamic volatility targeting using GARCH(1,1) to adjust exposure based on forecast variance.
* **Realistic Backtesting:** Implementation of a 10% rebalancing buffer and 10 bps transaction costs to simulate real-world slippage and commissions.
* **Market-Specific Tuning:** Customized feature engineering for high-noise environments like the Nikkei 225.

## Project Structure
- `HMM_Engine.py`: Core mathematical engine containing custom HMM and Viterbi implementations.
- `Projekt_końcowy.ipynb`: The research notebook containing data analysis, strategy logic, and performance visualizations.

## Results
> **Note:** The project highlights the "Fee Bleed" effect and demonstrates how smart rebalancing buffers can protect capital during high-turnover periods.
