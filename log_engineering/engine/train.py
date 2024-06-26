import warnings
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import wandb

    _has_wandb = True
except ImportError:
    warnings.warn("wandb is not installed, please install it if you want to use it")
    _has_wandb = False


def tabular(train, test, args):
    if args.wandb and _has_wandb:
        wandb.init(project=args.wandb_name, config=args)
        wandb.config.update(args)

    if args.hyperparam_selection:
        from sklearn.model_selection import GridSearchCV

        model = GridSearchCV(
            # RandomForestRegressor(random_state=42),
            args.TABULAR_MODELS[args.model](random_state=42),
            param_grid=args.TABULAR_PARAMS[args.model + "p"],
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            cv=5,
        )
        model.fit(train.drop([args.target], axis=1), train[args.target])
        model = model.best_estimator_
    else:
        model = args.TABULAR_MODELS[args.model](**args.DEFAULT_HYPERPARAMS[args.model])
        model.fit(train.drop([args.target], axis=1), train[args.target])

    # save feature importance if RF and wandb
    if args.model.startswith("RF") and args.wandb and _has_wandb:
        wandb.sklearn.plot_feature_importances(model, train.drop([args.target], axis=1).columns)
        
    
    preds = model.predict(test.drop([args.target], axis=1))
    
    # if args.model.endswith("c"): # classification suffix
    #     from sklearn.metrics import f1_score, precision_score, recall_score
    #     accuracy = (preds == test[args.target]).mean()
    #     f1 = f1_score(test[args.target], preds, average="macro")
    #     precision = precision_score(test[args.target], preds, average="macro")
    #     recall = recall_score(test[args.target], preds, average="macro")
    #     performances = {
    #         "accuracy": accuracy,
    #         "f1": f1,
    #         "precision": precision,
    #         "recall": recall,
    #         "random_seed": args.random_seed,
    #     }
    # else:
    mae_day = get_error(test[args.target].values, preds, format="days", fn="mae")
    mae_sec = get_error(test[args.target].values, preds, format="seconds", fn="mae")
    mse_day = get_error(test[args.target].values, preds, format="days", fn="mse")
    mse_sec = get_error(test[args.target].values, preds, format="seconds", fn="mse")
    performances = {
        "MAE (days)": mae_day,
        "MAE (secs)": mae_sec,
        "MSE (days)": mse_day,
        "MSE (secs)": mse_sec,
        "random_seed": args.random_seed,
    }
    if args.wandb and _has_wandb:
        wandb.log(performances)
        wandb.finish()

    return preds

def get_error(true, pred, format="seconds", fn="mae"):
    arr1 = np.expm1(true)
    arr2 = np.expm1(pred)
    if format == "days":
        arr1 = arr1 / (24 * 60 * 60)  # days
        arr2 = arr2 / (24 * 60 * 60)  # days
    elif format == "seconds":
        pass
    else:
        warnings.warn("Format invalid. Providing default.")
    if fn == "mae":
        return mean_absolute_error(arr1, arr2)
    elif fn == "mse":
        return mean_squared_error(arr1, arr2)
