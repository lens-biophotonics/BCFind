import scipy as sp
import numpy as np
import pandas as pd
import networkx as nx


def distance(x, y):
    return np.sqrt(sum((x - y) ** 2))


def bipartite_match(true_centers, pred_centers, max_match_dist, dim_resolution=1): 
    """
    Match true and predicted markers using max-weight bipartite matching.
    This function is a wrapper around scipy.spatial.distance.cdist and scipy.optimize.linear_sum_assignment.
    40 to 100 times faster than networkx.
    
    Parameters
    ----------
    true_centers : numpy.array or pandas.DataFrame
      array of dim(n_cells, n_dim)

    pred_centers : numpy.array or pandas.DataFrame
      array of dim(n_pred_cells, n_dim)

    max_distance : float
      distance below which two markers are never matched. If dim_resolution is given this distance must be in that scale.

    dim_resolution: list of float
      resolution of input data (e.g. micron per voxel axis)

    return:
      DataFrame of matched and not matched centers.
    """
    if np.isscalar(dim_resolution):
        dim_resolution = (dim_resolution, ) * true_centers.shape[1]
    
    dim_resolution = np.array(dim_resolution)

    scaled_true = true_centers * dim_resolution
    scaled_pred = pred_centers * dim_resolution

    dist = sp.spatial.distance.cdist(scaled_true, scaled_pred, metric='euclidean')
    dist[dist >= max_match_dist] = 1e9

    true_idxs, pred_idxs = sp.optimize.linear_sum_assignment(dist, maximize=False)

    pred_TP = [pred_idxs[i] for i in range(len(true_idxs)) if dist[true_idxs[i], pred_idxs[i]] < max_match_dist]
    true_TP = [true_idxs[i] for i in range(len(true_idxs)) if dist[true_idxs[i], pred_idxs[i]] < max_match_dist]
    FP = [idx for idx in range(pred_centers.shape[0]) if idx not in pred_TP]
    FN = [idx for idx in range(true_centers.shape[0]) if idx not in true_TP]

    # create data frame of labeled centers
    colnames = ["x", "y", "z", "radius", "shape", "name", "comment", "R", "G", "B"]
    node_eval = []
    for i in pred_TP:
        x, y, z = pred_centers[i, 0], pred_centers[i, 1], pred_centers[i, 2]
        node_eval.append([x, y, z, 0, 1, 'TP', 'predicted', 0, 255, 0])

    for i in FP:
        x, y, z = pred_centers[i, 0], pred_centers[i, 1], pred_centers[i, 2]
        node_eval.append([x, y, z, 0, 1, 'FP', 'predicted', 255, 0, 0])

    for i in FN:
        x, y, z = true_centers[i, 0], true_centers[i, 1], true_centers[i, 2]
        node_eval.append([x, y, z, 0, 1, 'FN', 'true', 255, 128, 0])

    return pd.DataFrame(node_eval, columns=colnames)


def nx_bipartite_match(true_centers, pred_centers, max_distance, dim_resolution=1):
    """
    Match true and predicted markers using max-weight bipartite matching.
    This function is a wrapper around networkx.algorithms.matching.max_weight_matching
    
    Parameters
    ----------
    true_centers : numpy.array or pandas.DataFrame
      array of dim(n_cells, n_dim)

    pred_centers : numpy.array or pandas.DataFrame
      array of dim(n_pred_cells, n_dim)

    max_distance : float
      distance below which two markers are never matched. If dim_resolution is given this distance must be in that scale.

    dim_resolution: list of float
      resolution of input data (e.g. micron per voxel axis)

    return:
      DataFrame of matched and not matched centers.
    """
    if np.isscalar(dim_resolution):
        dim_resolution = (dim_resolution, ) * true_centers.shape[1]

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
