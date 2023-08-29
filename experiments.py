import os
import pandas as pd
from log_engineering import engine, prepare_data, utils
import argparse
import warnings

warnings.filterwarnings("ignore")

# from tpot import TPOTRegressor

FEATURES = {
    # "TF": {"prefix": "tf_"},
    # "MF": {"prefix": "mf_"},
    "EF": {"prefix": "ef_"},
    "OH": {"prefix": "oh_"},
    "TIME": {"prefix": "fast_"},
    "MFTIME": {"prefix": "fmf_"},
}


def get_args_parser(add_help=True):
    """Get args parser"""
    parser = argparse.ArgumentParser(
        description="Event feature engineering for predictive monitoring.",
        add_help=add_help,
    )

    parser.add_argument(
        "--dataset",
        default="PermitLog",
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
        default="EF,TIME,MFTIME",
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

    parser.add_argument(
        "--wandb-login",
        default="raseidi",
        type=str,
        help="wandb username",
    )

    parser.add_argument(
        "--overwrite",
        default=False,
        type=bool,
        help="Overwrite experiment if exists",
    )

    parser.add_argument(
        "--random-seed",
        default=42,
        type=int,
        help="Seed",
    )

    args = parser.parse_args()
    args.TABULAR_MODELS = engine.TABULAR_MODELS
    args.DEFAULT_HYPERPARAMS = engine.DEFAULT_HYPERPARAMS
    return args


if __name__ == "__main__":
    args = get_args_parser()

    if utils.experiment_exists(args) and args.wandb and not args.overwrite:
        print("=" * 80)
        print("Experiment already exists. Skipping...")
        print("=" * 80)
        # exit(0)
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
            ].tolist()
        else:
            features = train.columns.tolist()
        if args.target not in features:
            features.append(args.target)
        # train, test = train[features], test[features]

        # if model is RF we also run the random seeds
        if args.model in ["RF", "RFc"]:
            N = 1
            for seed in range(N):
                args.DEFAULT_HYPERPARAMS[args.model]["random_state"] = seed
                args.random_seed = seed
                print("=" * 80)
                print(args.dataset, args.model, args.features)
                print("=" * 80)
                if utils.experiment_exists(args) and not args.overwrite:
                    print("Experiment already exists. Skipping...")
                    continue
                predictions = engine.train.tabular(train[features], test[features], args)
                if f"{args.target}_predicted" in test.columns:
                    test.loc[:, f"{args.target}_predicted"] += predictions
                else:
                    test.loc[:, f"{args.target}_predicted"] = predictions
            test.loc[:, f"{args.target}_predicted"] /= N
            test[["case:concept:name","concept:name", args.target, f"{args.target}_predicted"]].to_csv(f"data/results/predictions_{args.dataset}_model={args.model}_features={args.features}.csv", index=False)
            
        else:
            if utils.experiment_exists(args) and not args.overwrite:
                print("Experiment already exists. Skipping...")
            else:
                predictions = engine.train.tabular(train[features], test[features], args)
                test.loc[:, f"{args.target}_predicted"] = predictions
                test[["case:concept:name","concept:name", args.target, f"{args.target}_predicted"]].to_csv(f"data/results/predictions_{args.dataset}_model={args.model}_features={args.features}.csv", index=False)
    else:
        raise Exception(
            f"Model {args.model} not supported. Available models: {engine.TABULAR_MODELS.keys()}"
        )
    print("=" * 80)
    print()
