import numpy as np
import pandas as pd
import networkx as nx


def distance(x, y):
    """TODO describe function

    :param x:
    :param y:
    :returns:

    """
    return np.sqrt(sum((x - y) ** 2))


def bipartite_match(true_centers, pred_centers, max_distance, dim_resolution=[1, 1, 1]):
    """
    Match true and predicted markers using max-weight bipartite matching

    Parameters
    ----------
    true_centers : numpy.array or pandas.DataFrame
      array of dim(n_cells, n_dim)

    pred_centers : numpy.array or pandas.DataFrame
      array of dim(n_pred_cells, n_dim)

    max_distance : float
      distance (in um) below which to markers are never matched

    dim_resolution: list of float
      resolution of input data (e.g. micron per voxel axes)

    return:
      G : bipartite graph
      mate: list of matches
    """

    true_centers = np.array(true_centers)
    pred_centers = np.array(pred_centers)
    dim_resolution = np.array(dim_resolution)

    G = nx.Graph()

    for i, c in enumerate(true_centers):
        node = "t_%d" % i
        G.add_node(
            node,
            x=c[0] * dim_resolution[0],
            y=c[1] * dim_resolution[1],
            z=c[2] * dim_resolution[2],
        )

    for i, c in enumerate(pred_centers):
        node = "p_%d" % i
        G.add_node(
            node,
            x=c[0] * dim_resolution[0],
            y=c[1] * dim_resolution[1],
            z=c[2] * dim_resolution[2],
        )

    for ni in [n for n in G.nodes() if n[0] == "t"]:
        for nj in [n for n in G.nodes() if n[0] == "p"]:
            A = np.array([G.nodes[ni]["x"], G.nodes[ni]["y"], G.nodes[ni]["z"]])
            B = np.array([G.nodes[nj]["x"], G.nodes[nj]["y"], G.nodes[nj]["z"]])

            d = distance(A, B)

            if d < max_distance:
                w = 1.0 / max(0.001, d)
                G.add_edge(ni, nj, weight=w)

    mate = nx.algorithms.matching.max_weight_matching(G, maxcardinality=False)

    # get node names of TP, FP and FN

    TP_p = [node for m in mate for node in m if node.startswith("p")]
    TP_t = [node for m in mate for node in m if node.startswith("t")]

    F = list(set(G.nodes) - set(TP_p) - set(TP_t))
    FP = [node for node in F if node.startswith("p")]
    FN = [node for node in F if node.startswith("t")]

    # create data frame of labeled centers
    colnames = ["x", "y", "z", "radius", "shape", "name", "comment", "R", "G", "B"]
    node_eval = []
    for node in list(G.nodes):
        if node in TP_p:
            x, y, z = np.array(list(G.nodes[node].values())) / dim_resolution
            node_eval.append([x, y, z, 0, 1, "TP", "predicted", 0, 255, 0])
        if node in FP:
            x, y, z = np.array(list(G.nodes[node].values())) / dim_resolution
            node_eval.append([x, y, z, 0, 1, "FP", "predicted", 255, 0, 0])
        if node in FN:
            x, y, z = np.array(list(G.nodes[node].values())) / dim_resolution
            node_eval.append([x, y, z, 0, 1, "FN", "true", 255, 128, 0])

    node_eval = pd.DataFrame(node_eval, columns=colnames)

    return node_eval
