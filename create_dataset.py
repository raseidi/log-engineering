import os
from time import time
from log_engineering import LOGS_PATH
from log_engineering.feature import TIME_METHODS
from log_engineering.feature.encoding import encode
from log_engineering.utils import read_splitted_data
from log_engineering.feature import meta_trace

import argparse
import warnings

warnings.filterwarnings("ignore")


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
        "--encoding-dim",
        default=8,
        type=int,
        help="Encoding dimension",
    )

    return parser.parse_args()

if __name__ == "__main__":
    params = get_args_parser()
    save_path = os.path.join("data", f"{params.dataset}_dim={params.encoding_dim}.csv")

    # check if file exists
    if os.path.exists(save_path):
        print(
            f"File {params.dataset}_dim={params.encoding_dim} already exists. Skipping..."
        )
        exit(0)

    print("=" * 80)
    print(params)
    print()
    file_path = os.path.join(LOGS_PATH, params.dataset, "train_test")

    log = read_splitted_data(file_path)

    # elapsed time
    start = time() 
     
    """ time-based features """
    for fn_name, fn_config in TIME_METHODS.items():
        attr_name = "tf_" + fn_name
        if fn_config["groupbycase"]:
            log[attr_name] = log.groupby("case:concept:name")[
                "time:timestamp"
            ].transform(fn_config["fn"])
        else:
            log[attr_name] = log["time:timestamp"].transform(fn_config["fn"])
        log[attr_name] = log[attr_name].apply(lambda x: x.total_seconds())
        # to convert to days = (24 * 60 * 60)

    """ encoded dataset """
    encoded_log = encode(log[log.split_set == "train"], log[log.split_set == "test"])
    encoded_log.sort_values(by="event_id")
    log = log.merge(
        encoded_log.drop(["case:concept:name", "split_set"], axis=1), on="event_id"
    )

    """ statistical meta-features from the encoded dataset """
    FEATURE = "tf_execution_time"
    log[FEATURE].fillna(0, inplace=True)

    import numpy as np
    import pandas as pd

    traces = []
    for case in log["case:concept:name"].unique():
        meta = meta_trace(
            log[log["case:concept:name"] == case][FEATURE].values[1:] + 1
        )  # suming +1 since scipy has some internal exp/log functions
        meta = {
            "case:concept:name": case,
            **{f"mf_{i}": item for i, item in enumerate(meta)},
        }
        traces.append(meta)

    log = log.merge(pd.DataFrame(traces), on="case:concept:name")
    """ save """
    log.fillna(0, inplace=True)
    end = time() - start
    log["elapsed_time"] = end

    log.to_csv(save_path, index=False)
