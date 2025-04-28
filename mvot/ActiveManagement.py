# Library Install
import math
import yfinance as yf
import pandas as pd
import numpy as np

def fetch_stock_data(tickers, start="2022-01-01", end="2024-01-01"):
    data = yf.download(tickers, start=start, end=end)["Close"]
    data = data.dropna(axis=1)  # Drop any tickers with missing data
    return data

def covariance_matrix_build(calibration_data):
    returns = calibration_data.pct_change().dropna()
    covariance_matrix = returns.cov()
    return covariance_matrix

def momentum_calculator(calibration_data):
    returns = calibration_data.pct_change().dropna()
    alphas = {}

    momentum = returns[-90:].mean()
    momentum_rank = momentum.rank(ascending=False)
    alpha = momentum_rank/len(momentum_rank)
    alphas = alpha.to_dict()
    return alphas

