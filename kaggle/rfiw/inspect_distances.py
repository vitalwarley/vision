"""Inspect distances.
"""
import os
import logging
import multiprocessing as mp

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from models.losses import contrastive_loss
from models.metrics import accuracy_for_distance
from models.siamese import create_siamese_network
from toys.mnist import load as load_mnist
from data import load as load_fiw
from utils import (
    set_gpu, inspect_distances, visualize_distance_distribution,
    run_and_release
)

logging.getLogger('tensorflow').setLevel(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def get_data(dataset_name, **kwargs):
    info = None
    if dataset_name == 'mnist':
        data = load_mnist(**kwargs)
    elif dataset_name == 'fiw':
        data, info = load_fiw(**kwargs)

    if data is None:
        return None

    (tr_pairs, tr_y), (te_pairs, te_y) = data

    tr_pairs = [tr_pairs[:, 0, ...], tr_pairs[:, 1, ...]]
    te_pairs = [te_pairs[:, 0, ...], te_pairs[:, 1, ...]]

    return ((tr_pairs, tr_y), (te_pairs, te_y)), info


def get_model(input_shape):
    model = create_siamese_network(input_shape)
    optimizer = optimizers.Adam()
    model.compile(loss=contrastive_loss, optimizer=optimizer, 
                  metrics=[accuracy_for_distance])
    return model


def run(output, dataset, input_shape, samples_per_class, **kwargs):
    set_gpu()
    kwargs.update(input_shape=input_shape, samples_per_class=samples_per_class)
    data, info = get_data(dataset_name=dataset, **kwargs)
    n_samples = samples_per_class if info is None else info[1]

    (tr_pairs, tr_y), (te_pairs, te_y) = data

    model = get_model(input_shape)
    model.fit(tr_pairs, tr_y, batch_size=32, epochs=10, verbose=0)
    loss, acc = model.evaluate(te_pairs, te_y, verbose=0)

    distances, targets = inspect_distances(model, te_pairs, te_y, 64)
    results = (n_samples, loss, acc, distances, targets)

    # mp shared variable
    output['results'] = results


def main(dataset):

    if dataset == 'mnist':
        input_shape = (28, 28, 1)
        n_classes = 10
        samples_per_class = [5, 10, 25, 50]  # , 100, 250, 500, 750]
        kwargs = dict(n_classes=n_classes,
                      input_shape=input_shape)
    elif dataset == 'fiw':
        input_shape = (64, 64, 3)
        n_classes = 10
        samples_per_class = [5, 10, 15, 20, 25, 30, 35, 40]
        kwargs = dict(n_classes=n_classes,
                      input_shape=input_shape,
                      build_if_exists=True)

    results = []
    for ics in samples_per_class:
        print(f"Starting with ~{ics} samples...")
        kwargs.update(samples_per_class=ics)

        # Sadly, I couldn't use a decorator like @run_and_release in gpu utils:
        # -> pickle.PicklingError: Can't pickle <function at ...>: ...
        with mp.Manager() as manager:
            output = manager.dict()
            args = (output, dataset)
            proc = mp.Process(target=run,
                              args=args,
                              kwargs=kwargs)
            proc.start()
            proc.join()
            results.append(output.get('results', []))

        print(f"Done with ~{ics} samples.")
        os.system('clear')
    
    return results


def visualize_distances(results, dataset):
    ncols = 4
    nrows = len(results) // ncols
    _, axs = plt.subplots(nrows, ncols, figsize=(20, 20), squeeze=False)

    for i, result in enumerate(results):
        n_samples, loss, acc, distances, targets = result
        ax = axs[i // ncols][i % ncols]
        for c in range(2):
            sns.kdeplot(distances[targets == c], label=str(c), ax=ax)
            ax.set_title(f"# samples per class: ~{n_samples}\n \
                          loss: {loss:.2f}, acc: {acc:.2f}")

    plt.suptitle(f"{dataset}: siamese net distances")
    plt.show()


if __name__ == '__main__':
    mp.set_start_method('spawn')

    dataset = 'fiw'
    # TODO: inspect distances with varying n_classes (which families to  choose?)
    results = main(dataset)
    visualize_distances(results, dataset)
