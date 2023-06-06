import os
import pandas as pd
from log_engineering import engine, prepare_data 
import argparse
import warnings

warnings.filterwarnings("ignore")

# from tpot import TPOTRegressor

FEATURES = {"TF": {"prefix": "tf_"}, "MF": {"prefix": "mf_"}, "EF": {"prefix": "ef_"}}


def get_args_parser(add_help=True):
    """Get args parser"""
    parser = argparse.ArgumentParser(
        description="Event feature engineering for predictive monitoring.",
        add_help=add_help,
    )

    parser.add_argument(
        "--dataset",
        default="PrepaidTravelCost",
        type=str,
        help="Dataset name",
    )

    parser.add_argument(
        "--target",
        default="tf_remaining_time",
        type=str,
        help="Target column name",
    )

    parser.add_argument(
        "--encoding-dim",
        default=8,
        type=int,
        help="Dimensionality of encoding",
    )

    parser.add_argument(
        "--features",
        default="all",
        type=str,
        help="features to include (all or list of MF, EF, TF; e.g. MF,EF)",
    )

    parser.add_argument(
        "--model",
        default="RF",
        type=str,
        help="Learning model name",
    )

    parser.add_argument(
        "--hyperparam-selection",
        default=False,
        type=bool,
        help="if True, use grid search, else use default parameters",
    )

    parser.add_argument(
        "--wandb",
        default=True,
        type=bool,
        help="wandb logging",
    )

    parser.add_argument(
        "--wandb-name",
        default="log-engineering",
        type=str,
        help="wandb project name",
    )
    
    args = parser.parse_args()
    args.TABULAR_MODELS = engine.TABULAR_MODELS
    args.DEFAULT_HYPERPARAMS = engine.DEFAULT_HYPERPARAMS
    return args


def save_performances(p, args):
    performances_path = "data/results/performances.csv"
    if os.path.exists(performances_path):
        performances = pd.read_csv(performances_path)
    else:
        performances = pd.DataFrame()

    p.update(args.__dict__)
    p = pd.DataFrame([p])

    performances = pd.concat((performances, p))
    performances.to_csv(performances_path, index=False)


if __name__ == "__main__":
    args = get_args_parser()
    print("=" * 80)
    print(args)
    if args.model in engine.TABULAR_MODELS.keys():
        train, test = prepare_data.tabular(args)
        print("training...")

        # including only the features specified in args
        if args.features != "all":
            features = [
                FEATURES[f]["prefix"] for f in FEATURES if f in args.features.split(",")
            ]
            features = train.columns[
                [c.startswith(tuple(features)) for c in train.columns]
            ].tolist() + [args.target]
        else:
            features = train.columns.tolist()
        train, test = train[features], test[features]

        p = engine.train.tabular(train, test, args)
        print(p)
    else:
        raise Exception(
            f"Model {args.model} not supported. Available models: {engine.TABULAR_MODELS.keys()}"
        )
    print("=" * 80)
    print()

    # if not args.wandb:
    #     save_performances(p, args)
