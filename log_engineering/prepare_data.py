import pickle
import numpy as np

from argparse import Namespace
from log_engineering.utils import read_data

def tabular(args: Namespace):
    from sklearn.preprocessing import StandardScaler

    file_path = f"data/{args.dataset}_dim={args.encoding_dim}.csv"
    try:
        df = read_data(file_path)
    except FileNotFoundError:
        raise FileNotFoundError("File not found.")

    df = df.dropna()
    df.drop(columns=["tf_remaining_time"], inplace=True, axis=1)
    df = df.rename(columns={"fast_tf_remaining_time": "tf_remaining_time"})
    
    ignore_columns = [
        "case:concept:name",
        "concept:name",
        "time:timestamp",
        "event_id",
        "split_set",
        "elapsed_time",
    ]
    if args.target != "last_o_activity":
        ignore_columns.append("last_o_activity")
    
    train = df.loc[
        df["split_set"] == "train", df.columns.difference(ignore_columns)
    ].copy()
    test = df.loc[
        df["split_set"] == "test", df.columns.difference(ignore_columns)
    ].copy()
    sc = StandardScaler()
    sc.fit(train.drop(columns=args.target))
    train.loc[:, train.columns.difference([args.target])] = sc.transform(
        train.drop(columns=args.target)
    )
    test.loc[:, test.columns.difference([args.target])] = sc.transform(
        test.drop(columns=args.target)
    )

    # apply log if numeric, otherwise transform from categorical to numeric
    if train[args.target].dtype in [np.float64, np.int64]:
        train.loc[:, args.target] = train[args.target].apply(np.log1p)
        test.loc[:, args.target] = test[args.target].apply(np.log1p)
    else:
        targets = {k: v for v, k in enumerate(sorted(train[args.target].unique()))}
        train.loc[:, args.target] = train[args.target].apply(lambda x: targets[x])
        test.loc[:, args.target] = test[args.target].apply(lambda x: targets[x])
        
    test[["case:concept:name", "concept:name",]] = df.loc[df["split_set"] == "test", ['case:concept:name', 'concept:name']]
    return train, test


def save_vocab(vocab: dict, path: str = "vocab.pkl"):
    with open(path, "wb") as f:
        pickle.dump(vocab, f)


def read_vocab(path: str = "vocab.pkl"):
    with open(path, "rb") as f:
        loaded_vocab = pickle.load(f)
    return loaded_vocab
