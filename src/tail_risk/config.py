from .core import rolling_normal_variance, ewma_normal_variance

K_RATIO = 0.10
K_OVER_T_SHARES = (0.02, 0.035, 0.05)
VAR_WINDOW = 504
VAR_LEVEL = 0.01

MARKET_LABELS = {"sp500": "S&P 500", "euronext": "Euronext"}

VAR_MODELS = {
    "normal_rolling": rolling_normal_variance,
    "normal_ewma": ewma_normal_variance,
}

MODEL_LABELS = {
    "normal_rolling": r"normal, rolling $\sigma$",
    "normal_ewma": r"normal, EWMA $\sigma$",
}
