import os
import json
import numpy as np
import pandas as pd
from pathlib import Path


def vaa3d_to_numpy(marker_path):
    df = pd.read_csv(open(str(marker_path), "r"))[['#x', ' y', ' z']]
    df = df.dropna(axis=0)
    return df.to_numpy()


def slicer_to_numpy(marker_path):
    with open(marker_path, 'r') as f:
        markers = json.load(f)
    X = []
    control_points = markers['markups'][0]['controlPoints']
    for cp in control_points:
        X.append(cp['position'])
    return np.array(X)


def get_gt_as_numpy(marker_path):
    if isinstance(marker_path, Path):
        suffix = marker_path.suffix
    else:
        _, suffix = os.path.splitext(marker_path)
    
    if suffix == '.marker':
        gt = vaa3d_to_numpy(marker_path)
    elif suffix == '.json':
        gt = slicer_to_numpy(marker_path)
    else:
        raise ValueError('marker_path is incompatible with known formats: Vaa3d (.marker) or 3DSlicer (.json).')
    return gt[:, [2, 1, 0]] # transpose axis from [x, y, z] to [z, y, x]

