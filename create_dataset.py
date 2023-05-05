from log_engineering.feature import TIME_METHODS
from log_engineering.feature.encoding import encode
from log_engineering.utils import read_splitted_data

if __name__ == "__main__":
    dataset = "PrepaidTravelCost"
    file_path = f"/home/seidi/datasets/logs/{dataset}/train_test"
    dimensions = 8

    log = read_splitted_data(file_path)

    """ time-based features """
    for fn_name, fn_config in TIME_METHODS.items():
        if fn_config["groupbycase"]:
            log[fn_name] = log.groupby("case:concept:name")["time:timestamp"].transform(
                fn_config["fn"]
            )
        else:
            log[fn_name] = log["time:timestamp"].transform(fn_config["fn"])
        log[fn_name] = log[fn_name].apply(lambda x: x.total_seconds())
        # to convert to days = (24 * 60 * 60)

    """ encoded dataset """
    encoded_log = encode(log[log.split_set == "train"], log[log.split_set == "test"])
    encoded_log.sort_values(by="event_id")
    log = log.merge(
        encoded_log.drop(["case:concept:name", "split_set"], axis=1), on="event_id"
    )

    """ statistical meta-features from the encoded dataset """
    # TODO

    """ save """
    log.fillna(0, inplace=True)
    log.to_csv(f"data/{dataset}_dim={dimensions}.csv", index=False)
