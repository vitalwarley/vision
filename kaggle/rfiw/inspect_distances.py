"""Inspect distances.
"""
import os
import logging
import pickle
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
from utils import set_gpu, inspect_distances, visualize_distances

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.basicConfig(format='%(asctime)s:%(levelname)s - %(message)s',
                    level=logging.INFO)

MNIST_TOTAL_CLASSES = 10
FIW_TOTAL_CLASSES = 470


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
    model.compile(loss=contrastive_loss,
                  optimizer=optimizer,
                  metrics=[accuracy_for_distance])
    return model


def run(output, dataset, input_shape, samples_per_class, **kwargs):
    set_gpu()
    kwargs.update(input_shape=input_shape, samples_per_class=samples_per_class)
    data, info = get_data(dataset_name=dataset, **kwargs)
    n_samples = (samples_per_class
                 if info is None else info['samples_per_class'])

    (tr_pairs, tr_y), (te_pairs, te_y) = data

    model = get_model(input_shape)
    logging.info('Fitting model.')
    model.fit(tr_pairs, tr_y, batch_size=32, epochs=10, verbose=1)
    logging.info('Evaluating model.')
    loss, acc = model.evaluate(te_pairs, te_y, verbose=1)

    info.update(dataset=dataset,
                samples_per_class=n_samples,
                loss=loss,
                acc=acc)

    logging.info('Inspecting distances.')
    distances, targets = inspect_distances(model, te_pairs, te_y, 64)
    results = (distances, targets, info)

    # mp shared variable
    output['results'] = results


def run_and_release(dataset, **kwargs):
    """Run model in a new process.
    """
    # Sadly, I couldn't use a decorator like @run_and_release in gpu utils:
    # -> pickle.PicklingError: Can't pickle <function at ...>: ...
    with mp.Manager() as manager:
        output = manager.dict()
        args = (output, dataset)
        proc = mp.Process(target=run, args=args, kwargs=kwargs)
        proc.start()
        proc.join()
        results = output.get('results', [])
    return results


def main(dataset, n_classes):

    if dataset == 'mnist':
        input_shape = (28, 28, 1)
        samples_per_class = [5, 10, 25, 50]  # , 100, 250, 500, 750]
        kwargs = dict(n_classes=n_classes, input_shape=input_shape)
    elif dataset == 'fiw':
        input_shape = (64, 64, 3)
        samples_per_class = [5, 10, 15, 20, 25, 30, 35, 40]
        kwargs = dict(n_classes=n_classes,
                      input_shape=input_shape,
                      build_if_exists=True)

    results = []
    if isinstance(n_classes, int):
        for ics in samples_per_class:
            logging.info(f"starting with ~{ics} samples...")
            kwargs.update(samples_per_class=ics)
            res = run_and_release(dataset, **kwargs)
            results.append(res)
            logging.info(f"done with ~{ics} samples.")
            # os.system('clear')
    elif isinstance(n_classes, list):
        for cls in n_classes:
            logging.info(f"starting with ~{cls} classes...")
            kwargs.update(n_classes=cls, samples_per_class=0)
            res = run_and_release(dataset, **kwargs)
            results.append(res)
            logging.info(f"done with ~{cls} classes.")
            # os.system('clear')

    return results


if __name__ == '__main__':
    mp.set_start_method('spawn')

    dataset = 'fiw'
    results = main(dataset, n_classes=[3, 6, 10, 100, 200, 470])
    with open('results.pkl', 'wb') as fp:
        pickle.dump(results, fp)
    visualize_distances(results, dataset)
