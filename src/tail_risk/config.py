from functools import partial

from .core import rolling_sigma, ewma_sigma, normal_var, student_t_var

K_RATIO = 0.10
K_OVER_T_SHARES = (0.02, 0.035, 0.05)
VAR_WINDOW = 504
VAR_LEVEL = 0.01

MARKET_LABELS = {"sp500": "S&P 500", "euronext": "Euronext"}

VAR_MODELS = {
    "normal_rolling": partial(normal_var, sigma_function=rolling_sigma),
    "normal_ewma": partial(normal_var, sigma_function=ewma_sigma),
    "t_rolling": partial(student_t_var, sigma_function=rolling_sigma),
    "t_ewma": partial(student_t_var, sigma_function=ewma_sigma),
}

MODEL_LABELS = {
    "normal_rolling": r"normal, rolling $\sigma$",
    "normal_ewma": r"normal, EWMA $\sigma$",
    "t_rolling": r"Student's t, rolling $\sigma$",
    "t_ewma": r"Student's t, EWMA $\sigma$",
}
