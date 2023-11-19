import os
import numpy as np
import pandas as pd
from time import time
from log_engineering import LOGS_PATH
from log_engineering.feature import TIME_METHODS
from log_engineering.feature.encoding import encode, one_hot
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
        default="PermitLog",
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

    print("=" * 80)
    print(params)
    print()
    file_path = os.path.join(LOGS_PATH, params.dataset, "train_test")

    if not os.path.exists(save_path):
        log = read_splitted_data(file_path)
    else:
        import pandas as pd

        log = pd.read_csv(save_path)
        log["time:timestamp"] = pd.to_datetime(log["time:timestamp"], infer_datetime_format=True)
    
    # print(*sorted(log.columns), sep='\n')

    current_features = {x.split("_")[0] for x in log.columns}
    current_features = current_features - {"fmf", "fast_tf"}
    encoding_methods = ["fast_tf", "fmf"]
    drop_cols = [c for c in log.columns if c.startswith("fast_tf")]
    drop_cols2 = [c for c in log.columns if c.startswith("fmf")]
    log.drop(drop_cols + drop_cols2, axis=1, inplace=True)
    tb_time = None
    enc_time = None
    mf_time = None
    oh_time = None
    ftb_time = None
    fmf_time = None
    FEATURE = "tf_execution_time" # to extract meta-features
    group = (log
        .sort_values(by="time:timestamp", ascending=True)
        .groupby(["case:concept:name", "split_set"], as_index=False, observed=True, group_keys=False)
    )
    if "fast_tf" not in current_features:
        start = time()
        for fn_name, fn_config in TIME_METHODS.items():
            attr_name = "fast_tf_" + fn_name
            if fn_name == "execution_time":
                log[attr_name] = (
                    group["time:timestamp"]
                    .diff()
                    .loc[log.index]
                    ["time:timestamp"]
                    .dt.total_seconds()
                    .fillna(0)
                )               
            elif fn_name == "accumulated_time":
                # log[attr_name] = (
                #     group["time:timestamp"]
                #     .apply(lambda x: x - x.min())
                #     .loc[log.index]
                #     .dt.total_seconds()
                # )
                log[attr_name] = (
                    group["time:timestamp"]
                    .transform("min")
                )
                log[attr_name] = pd.to_timedelta(log["time:timestamp"] -log[attr_name]).dt.total_seconds()
            elif fn_name == "remaining_time":
                # log[attr_name] = (
                #     group["time:timestamp"]
                #     .apply(lambda x: x.max() - x)
                #     .loc[log.index]
                #     .dt.total_seconds()
                # )
                log[attr_name] = (
                    group["time:timestamp"]
                    .transform("max")
                )
                log[attr_name] = pd.to_timedelta(log[attr_name] - log["time:timestamp"]).dt.total_seconds()
            elif fn_name == "within_day":
                log[attr_name] = (
                    pd
                    .to_timedelta(
                        log['time:timestamp']
                        .dt.time.astype(str)
                        )
                    .dt.total_seconds().values
                )
            elif fn_name == "within_week":
                # amount of seconds since the beginning of the week
                # it is not possible to convert it back to a date
                log[attr_name] =  (
                    pd.to_timedelta(log["time:timestamp"].dt.dayofweek, unit="d")
                    + pd.to_timedelta(log["time:timestamp"].dt.time.astype(str))
                ).dt.total_seconds()
                
                # let's also include the day of the week
                log["fast_tf_day_of_week"] = log["time:timestamp"].dt.dayofweek.astype(np.float32)
                    
        # to convert to days = (24 * 60 * 60)
        ftb_time = time() - start

    if "ef" not in current_features:
        """encoded dataset"""
        start = time()
        encoded_log = encode(
            log[log.split_set == "train"], log[log.split_set == "test"]
        )
        enc_time = time() - start
        encoded_log.sort_values(by="event_id")
        log = log.merge(
            encoded_log.drop(["case:concept:name", "split_set"], axis=1), on="event_id"
        )
        log[FEATURE].fillna(0, inplace=True)

    """ meta-features from the encoded dataset """
    if "fmf" not in current_features:
        start = time()
        fmf_times = {}
        import multiprocessing
        
        for feature in ["fast_tf_execution_time_caise", "fast_tf_accumulated_time_caise", "fast_tf_within_day_caise", "fast_tf_within_week_caise"]:
            fmf_time = True
            start = time()
            with multiprocessing.Pool(20) as pool:
                    data = pool.starmap(meta_trace, group[FEATURE])
            end = time() - start
            fmf_times.update({f"fmf_{feature}": end})
            df = dict(case_id=[], split_set=[], meta_trace=[])
            for d in data:
                    df["case_id"].append(d[0][0])
                    df["split_set"].append(d[0][1])
                    df["meta_trace"].append(d[1])

            df = pd.DataFrame(df)
            df = df.rename(columns={"case_id": "case:concept:name"})
            df[[f"fmf_{i}" for i in range(0, len(d[1]))]] = pd.DataFrame(df["meta_trace"].tolist(), index=df.index)
            log.join(df.set_index(["case:concept:name", "split_set"]), on=["case:concept:name", "split_set"])
            

        """ save """
        log.fillna(0, inplace=True)
        
    
    # if "fmf" not in current_features:
    #     group = (log
    #         .sort_values(by="time:timestamp", ascending=True)
    #         .groupby(["case:concept:name", "split_set"])
    #     )
    #     from pandarallel import pandarallel
    #     pandarallel.initialize(nb_workers=min(os.cpu_count(), 20))
    #     warnings.filterwarnings("ignore")
    #     fmf_time = True
    #     fmf_times = {}
    #     for feature in ["fast_tf_execution_time", "fast_tf_accumulated_time", "fast_tf_within_day", "fast_tf_within_week"]:
    #         start = time()
    #         mf = group.parallel_apply(lambda x: meta_trace(x[feature]))
    #         end = time() - start
    #         values = mf.tolist()
    #         mf = mf.reset_index()
    #         mf.loc[:, [f"fmf_{feature}_{i}" for i in range(len(values[0]))]] = values
    #         mf.drop(columns=[0], inplace=True, errors="ignore")
    #         log = log.merge(mf, on=["case:concept:name", "split_set"])
            
    #         fmf_times.update({f"fmf_{feature}": end})

    
    """ one-hot encoding """
    if "oh" not in current_features:
        start = time()
        oh = one_hot(log)
        oh_time = time() - start
        log = log.merge(oh, on=["case:concept:name", "split_set"])

    """ updating meta info """
    try:
        meta_info = pd.read_csv("data/results/meta_info.csv")
    except:
        meta_info = pd.DataFrame()

    # remind: i'm gonna regret having written like this
    if params.dataset in meta_info.dataset.unique():
        # if we have more parameters, we need to check them too
        if tb_time is not None:
            meta_info.loc[
                (meta_info.dataset == params.dataset)
                & (meta_info.encoding_dim == params.encoding_dim),
                "elapsed_time_time",
            ] = tb_time
        if ftb_time is not None:
            meta_info.loc[
                (meta_info.dataset == params.dataset)
                & (meta_info.encoding_dim == params.encoding_dim),
                "elapsed_time_fast_time_caise",
            ] = ftb_time
        if enc_time is not None:
            meta_info.loc[
                (meta_info.dataset == params.dataset)
                & (meta_info.encoding_dim == params.encoding_dim),
                "elapsed_time_enc",
            ] = enc_time
        if mf_time is not None:
            meta_info.loc[
                (meta_info.dataset == params.dataset)
                & (meta_info.encoding_dim == params.encoding_dim),
                "elapsed_time_meta_caise",
            ] = mf_time
        if fmf_time is not None:
            for key, value in fmf_times.items():
                meta_info.loc[
                    (meta_info.dataset == params.dataset)
                    & (meta_info.encoding_dim == params.encoding_dim),
                    f"elapsed_time_{key}",
                ] = value
                
        if oh_time is not None:
            meta_info.loc[
                (meta_info.dataset == params.dataset)
                & (meta_info.encoding_dim == params.encoding_dim),
                "elapsed_time_oh",
            ] = oh_time
    else:
        new_row = pd.DataFrame(
            [
                {
                    "dataset": params.dataset,
                    "encoding_dim": params.encoding_dim,
                    "elapsed_time_enc": enc_time,
                    "elapsed_time_time": tb_time,
                    "elapsed_time_meta": mf_time,
                    # "elapsed_time_oh": oh_time,
                }
            ]
        )
        meta_info = pd.concat([meta_info, new_row])

    meta_info.to_csv("data/results/meta_info.csv", index=False)
    log.to_csv(save_path, index=False)

