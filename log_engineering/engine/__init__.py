from log_engineering.engine import train
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor

TABULAR_MODELS = {
    "RF": RandomForestRegressor,
    "KNN": KNeighborsRegressor,
    "SVM": SVR,
    "MLP": MLPRegressor,
    "XGB": XGBRegressor,
    # "LGB": "LGBMClassifier",
    # "CAT": "CatBoostClassifier",
}

DEFAULT_HYPERPARAMS = {
    "RF": {
        "n_jobs": -1,
        "random_state": 42,
    },
    "KNN": {
        "n_jobs": -1,
    },
    "SVM": {
        "kernel": "rbf",
    },
    "MLP": {
        "random_state": 42,
    },
    "XGB": {
        "n_jobs": -1,
        "random_state": 42,
    },
}
__all__ = [
    "train",
    # "RandomForestRegressor",
    # "KNeighborsRegressor",
    # "SVR",
    # "MLPRegressor",
    # "XGBRegressor",
]
