import pandas as pd
import numpy as np
import networkx as nx
from pm4py.read import read_xes
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from karateclub.node_embedding.neighbourhood import Node2Vec


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
            # ['case:concept:name', 'concept:name', 'time:timestamp', 'split_set','remaining_time', 'execution_time', 'accumulated_time', 'within_day','within_week', 'start_timestamp']
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

    data = pd.DataFrame(embeddings, columns=[f"feature_{i}" for i in range(dimensions)])
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