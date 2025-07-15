# Trii Stocks Allocation: Portfolio Optimization Project

## Overview
This project implements a portfolio optimization workflow that forecasts future weekly returns and volatility of stocks, then finds the allocation that maximizes the Sharpe ratio. The approach combines financial statistics, machine learning, and risk management techniques, with a focus on stocks available through the Trii platform.

## Project Structure
- **Notebooks/**: Contains the main Jupyter notebooks for each step of the workflow:
  - `1. Trii Catalog Stock Pre-selection.ipynb`: Downloads, preprocesses, and filters stocks using technical indicators.
  - `2. Future returns and Covariance matrix estimation.ipynb`: Estimates expected returns and the covariance matrix using statistical and ML models.
  - `3. Trii Catalog Sharpe-Ratio Maximizing Allocation.ipynb`: Optimizes the portfolio allocation to maximize the Sharpe ratio, subject to constraints.
  - `4. Trii Catalog CPPI Strategy on Chosen Allocation with Brownian Motion Simulation.ipynb`: Simulates a drawdown-based CPPI (Constant Proportion Portfolio Insurance) strategy on the optimized allocation.
  - **99.x Notebooks**: Experimental/extra notebooks with additional modeling and tests. These are not essential for the main workflow.
- **src/risk_kit.py**: Core module with reusable functions for financial statistics, risk metrics, portfolio optimization, and simulation.
- **requirements.txt**: Lists all required Python dependencies.

## Main Workflow
1. **Stock Pre-selection**: Download historical price data, preprocess, and filter stocks using technical indicators.
2. **Forecasting**: Estimate future returns and the covariance matrix using statistical and machine learning models (e.g., VAR, GARCH, etc.).
3. **Portfolio Optimization**: Find the allocation that maximizes the Sharpe ratio, with constraints on individual stock weights.
4. **Risk Management Simulation**: Apply and simulate a CPPI strategy to manage drawdown risk, using Brownian motion for scenario analysis.

## Setup
1. Clone the repository and navigate to the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the main notebooks in Jupyter Lab or Jupyter Notebook.

## Usage
- Run the notebooks in order (`1.` to `4.`) for the full workflow.
- The `src/risk_kit.py` module is imported in the notebooks and provides all necessary financial and optimization functions.
- Data files (CSV) are generated and used between steps for seamless workflow.

## Notes
- Notebooks starting with `99.` are for experimentation and are not required for the main analysis.
- The project is designed for educational and research purposes, and can be adapted for other stock universes or platforms.

## Dependencies
Key packages:
- `pandas`, `numpy`, `scikit-learn`, `yfinance`, `matplotlib`, `seaborn`
- `tensorflow`, `statsmodels`, `mgarch`, `ipywidgets`

See `requirements.txt` for the full list.

Written by Cursor AI.