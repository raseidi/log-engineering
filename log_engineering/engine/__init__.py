from log_engineering.engine import train
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor

# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor
# from xgboost import XGBRegressor

TABULAR_MODELS = {
    "RF": RandomForestRegressor,
    "KNN": KNeighborsRegressor,
    "SVM": SVR,
    "MLP": MLPRegressor,
    # "XGB": XGBRegressor,
    "RFc": RandomForestClassifier, 
    "KNNc": KNeighborsClassifier,
    "SVMc": SVC,
}

DEFAULT_HYPERPARAMS = {
    "RF": {
        "n_jobs": -1,
        "random_state": 42,
    },
    "RFc": {
        "n_jobs": -1,
        "random_state": 42,
    },
    "KNN": {
        "n_jobs": -1,
    },
    "KNNc": {
        "n_jobs": -1,
    },
    "SVM": {
        "kernel": "rbf",
    },
    "SVMc": {
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
]
