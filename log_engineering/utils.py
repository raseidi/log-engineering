import os
import pandas as pd


def read_splitted_data(path: str, return_all_cols: bool = False):
    train_log = pd.read_csv(f"{path}/train.csv")
    test_log = pd.read_csv(f"{path}/test.csv")
    train_log["split_set"] = "train"
    test_log["split_set"] = "test"
    data = pd.concat((train_log, test_log))
    drop_cases = data.groupby("case:concept:name").size()
    drop_cases = drop_cases[drop_cases < 5].index
    data = data[~data["case:concept:name"].isin(drop_cases)]
    data["time:timestamp"] = pd.to_datetime(
        data["time:timestamp"], infer_datetime_format=True
    ).dt.tz_localize(None)

    data.reset_index(inplace=True, drop=True)
    data["event_id"] = pd.RangeIndex(0, len(data))
    if not return_all_cols:
        data = data.loc[
            :,
            [
                "case:concept:name",
                "concept:name",
                "time:timestamp",
                "event_id",
                "split_set",
            ],
        ]

    return data


def read_data(path: str):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        raise FileNotFoundError(f"Dataset {path} not found.")

def experiment_exists(args):
    import wandb
    # check if experiment already exists from wandb
    api = wandb.Api()
    runs = api.runs(f"{args.wandb_name}/{args.dataset}")
    runs = api.runs(f"log-engineering", )
    common_args = ["dataset", "target", "encoding_dim", "features", "model", "hyperparam_selection"]
    config_list = []
    for run in runs:
        if run.state == "crashed":
            continue
        run.config.pop("DEFAULT_HYPERPARAMS")
        run.config.pop("TABULAR_MODELS")

        config_list.append(
        {k: v for k,v in run.config.items()
         if not k.startswith('_')})

    for config in config_list:
        if all([config[arg] == args.__dict__[arg] for arg in common_args]):
            return True
    return False
