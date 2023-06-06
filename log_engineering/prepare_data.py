from argparse import Namespace
import os
import torch
import pickle
import numpy as np
import pandas as pd
from log_engineering.utils import read_data


def get_vocab(data, dataset_name):
    if os.path.exists(f"data/vocab/{dataset_name}.pkl"):
        vocab = read_vocab(f"data/vocab/{dataset_name}.pkl")
    else:
        vocab = {"<unk>": 0, "<eos>": 1}
        vocab.update({v: k + 2 for k, v in enumerate(data["concept:name"].unique())})
        save_vocab(vocab, path=f"data/vocab/{dataset_name}.pkl")

    return vocab


def tabular(args: Namespace):
    from sklearn.preprocessing import StandardScaler

    file_path = f"data/{args.dataset}_dim={args.encoding_dim}.csv"
    try:
        df = read_data(file_path)
    except FileNotFoundError:
        raise FileNotFoundError("File not found.")
    # df["tf_remaining_time"] /= 86400 # converting to days
    df = df.dropna()
    # vocab = get_vocab(df, dataset_name=args.dataset)
    # df.loc[:, "concept:name"] = df["concept:name"].apply(lambda x: vocab[x])

    ignore_columns = [
        "case:concept:name",
        "concept:name",
        "time:timestamp",
        "event_id",
        "split_set",
        "elapsed_time"
    ]
    train = df.loc[
        df["split_set"] == "train", df.columns.difference(ignore_columns)
    ].copy()
    test = df.loc[
        df["split_set"] == "test", df.columns.difference(ignore_columns)
    ].copy()
    del df
    sc = StandardScaler()
    sc.fit(train.drop(columns=args.target))
    train.loc[:, train.columns.difference([args.target])] = sc.transform(
        train.drop(columns=args.target)
    )
    test.loc[:, test.columns.difference([args.target])] = sc.transform(
        test.drop(columns=args.target)
    )

    train.loc[:, args.target] = train[args.target].apply(np.log1p)
    test.loc[:, args.target] = test[args.target].apply(np.log1p)
    return train, test


def save_vocab(vocab: dict, path: str = "vocab.pkl"):
    with open(path, "wb") as f:
        pickle.dump(vocab, f)


def read_vocab(path: str = "vocab.pkl"):
    with open(path, "rb") as f:
        loaded_vocab = pickle.load(f)
    return loaded_vocab
