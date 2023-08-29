import pandas as pd
import numpy as np
import networkx as nx
from pm4py.read import read_xes
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from karateclub.node_embedding.neighbourhood import Node2Vec


def one_hot(log):
    """One hot encoding for the event log

    Args:
        log (pandas.DataFrame): an event log as a dataframe

    Returns:
        df (pandas.DataFrame): the one hot encoded event log
        
    ref: https://github.com/irhete/predictive-monitoring-benchmark/blob/master/transformers/AggregateTransformer.py#L34
    """
    df = log[["case:concept:name", "concept:name", "split_set"]].copy()
    df = (
        pd.get_dummies(df, columns=["concept:name"])
        .groupby(["case:concept:name", "split_set"])
        .sum()
    )
    df.columns = [f"oh_{c}" for c in df.columns]
    df.reset_index(inplace=True)
    return df


def create_graph(log) -> nx.Graph:
    """
    Creates a graph using the pm4py library and converts to a networkx DiGraph

    Parameters
    -----------------------
    import_path: str,
        Path and file name to be imported
    Returns
    -----------------------
    graph: nx.DiGraph()
        A graph generated from the event log (includes edge weights based on transition occurrences)
        
    ref: https://github.com/gbrltv/business_process_encoding/blob/master/compute_encoding/node2vec_.py
    """
    graph = nx.Graph()

    dfg = dfg_discovery.apply(log)
    for edge in dfg:
        graph.add_weighted_edges_from([(edge[0], edge[1], dfg[edge])])

    return graph


def convert_trace_mapping(trace, mapping):
    """
    Convert traces activity name using a given mapping

    Parameters
    -----------------------
    traces: List,
        List of traces
    mapping: dict:
        Dictionary containing activities mapping
    Returns
    -----------------------
        List of converted traces
    """

    return [mapping[act] for act in trace]


def trace_feature_vector_from_edges(embeddings, trace, dimension):
    """
    Computes average feature vector for a trace

    Parameters
    -----------------------
    embeddings,
        Text-based model containing the computed encodings
    traces: List,
        List of traces treated as sentences by the model
    Returns
    -----------------------
    vectors: List
        list of vector encodings for each trace
    """

    trace_vector_average = []
    for i in range(len(trace) - 1):
        try:
            emb1, emb2 = embeddings[trace[i]], embeddings[trace[i + 1]]
            trace_vector_average.append((emb1 + emb2) / 2.0)
        except KeyError:
            pass
    if len(trace_vector_average) == 0:
        return np.zeros(dimension)

    return np.array(trace_vector_average).mean(axis=0)


def _encode(data, model, mapping, dimensions):
    case_id, event_id, embeddings = [], [], []
    for group in data.groupby("case:concept:name"):
        for i in range(1, len(group[1]) + 1):
            events = list(group[1].iloc[:i, 1])
            trace = ["".join(x) for x in events]
            embeddings.append(
                trace_feature_vector_from_edges(
                    model.get_embedding(),
                    convert_trace_mapping(trace, mapping),
                    dimensions,
                )
            )
            event_id.append(group[1].iloc[i - 1, 3])
            case_id.append(group[0])

    data = pd.DataFrame(embeddings, columns=[f"ef_{i}" for i in range(dimensions)])
    data.insert(0, "case:concept:name", case_id)
    data.insert(len(data.columns), "event_id", event_id)

    return data


def encode(
    train: pd.DataFrame,
    test: pd.DataFrame,
    dimensions: int = 8,
    concat_sets: bool = True,
):
    graph = create_graph(train)
    mapping = dict(zip(graph.nodes(), [i for i in range(len(graph.nodes()))]))
    graph = nx.relabel_nodes(graph, mapping)

    model = Node2Vec(dimensions=dimensions)
    model.fit(graph)

    # test set is encoded with model from training
    train = _encode(train, model, mapping, dimensions)
    test = _encode(test, model, mapping, dimensions)

    if concat_sets:
        train["split_set"] = "train"
        test["split_set"] = "test"
        return pd.concat((train, test))
    else:
        return train, test
