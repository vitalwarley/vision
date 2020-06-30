"""
Debug model output.
"""
import numpy as np
from tqdm import tqdm

from data import generate
from preprocess import prewhiten

def inspect_distances(model, x, y, batch_size):
    distances = []
    targets = []

    for (x_batch, y_batch) in tqdm(
            generate(x, y, batch_size=batch_size),
            total=len(y) // batch_size):
        dist = np.squeeze(model.predict_on_batch(x_batch))
        target = np.squeeze(y_batch)
        distances.append(dist)
        targets.append(target)

    distances = np.concatenate(distances)
    targets = np.concatenate(targets)

    return distances, targets


def inspect_distances_v2(model, x, y, batch_size):
    distances = []
    targets = []

    for (x_batch, y_batch) in tqdm(
            generate(x, y, batch_size=batch_size),
            total=len(y) // batch_size):
        x_batch[0] = prewhiten(x_batch[0])
        x_batch[1] = prewhiten(x_batch[1])
        dist = np.squeeze(model.predict_on_batch(x_batch))
        target = np.squeeze(y_batch)
        distances.append(dist)
        targets.append(target)

    distances = np.concatenate(distances)
    targets = np.concatenate(targets)

    return distances, targets

