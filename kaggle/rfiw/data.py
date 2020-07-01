"""
Load data.
"""
import os
import logging
from pathlib import Path
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load(n_classes=0,
         samples_per_class=0,
         gap=5,
         input_shape=(64, 64, 3),
         s='train',
         return_np=True,
         build_if_exists=False):
    """Returns an array of pairs and labels or a train and validation dataset.
    """
    data = 'data'
    relations_file = os.path.join(data, 'train_relationships.csv')
    relations = pd.read_csv(relations_file)
    folder = os.path.join(data, s)

    families = os.listdir(folder)
    n_individuals_families = defaultdict(list)

    # dict(n_individuals : [families_with_n_individuals...])
    for fam in families:
        n_ind = len(os.listdir(f"{folder}/{fam}"))
        n_individuals_families[n_ind].append(fam)

    # Use all samples instead
    if not samples_per_class:
        fams = sorted(n_individuals_families,
                      key=lambda x: len(n_individuals_families[x]),
                      reverse=True)
        # Select most populated families
        families = [n_individuals_families[key] for key in fams]
        families = [fam for sublist in families for fam in sublist]
        families = families[:n_classes]
    else:
        # list(families with n_individuals in range(...))
        for n_ind in range(samples_per_class - gap, samples_per_class + gap):
            if n_ind in n_individuals_families:
                families.append(n_individuals_families[n_ind])
        # Select a few
        if len(families) >= n_classes:
            families = np.random.choice(families,
                                        size=n_classes,
                                        replace=False)
        # Not enough families,
        #  then add more families:
        #   start from those with the similar number of individuals
        else:
            # list(n_individuals...) <- families with more n_individuals)
            #  in decreasing order
            fams = sorted(n_individuals_families,
                          key=lambda x: len(n_individuals_families[x]),
                          reverse=True)
            # pivot similar families index (wrt to n_individuals_families)
            while len(families) < n_classes or not fams:
                # Index of most similar fams (in terms of n_individuals)
                pivot = np.argmin(
                    np.abs(np.array(fams.keys()) - samples_per_class))
                key = fams.pop(pivot)
                # Most similar fams
                fams_to_add = n_individuals_families[key]
                # Add what is needed
                need = n_classes - len(families)
                can_add = len(fams_to_add)
                will_add = need if need <= can_add else can_add
                fams_to_add = np.random.choice(fams_to_add, size=will_add)
                families.append(fams_to_add)

    # FIDX|FIDY|FIDZ|...
    fams = '|'.join(families)

    # Only those pairs within `families`
    relations = relations[relations.p1.str.contains(fams)
                          & relations.p2.str.contains(fams)]

    if os.path.exists('pairs.csv') and not build_if_exists:
        pairs = pd.read_csv('pairs.csv')
    else:
        logging.info('Building pairs relation...')
        try:
            pairs = build_pairs_relation(folder,
                                         relations,
                                         from_families=families)
        except (RuntimeError, ValueError, KeyError):
            # Couldn't build it
            return None
        pairs.to_csv('pairs.csv', index=False)

    if os.path.exists('image_pairs.npz') and not build_if_exists:
        loaded = np.load('image_pairs.npz')
        pairs, y = loaded['pairs'], loaded['y']
    else:
        logging.info('Building pairs image path...')
        pairs = build_pairs_path(folder, pairs)
        pairs.to_csv('pairs_path.csv', index=False)
        logging.info('Loading pairs images...')
        pairs, y = load_imgs(pairs, target_size=input_shape)
        np.savez_compressed('image_pairs', pairs=pairs, y=y)

    # Build info
    n_families = len(families)
    n_individuals_per_family = [
        len(os.listdir(f"{folder}/{fam}")) for fam in families
    ]
    samples_per_class = np.mean(n_individuals_per_family)
    n_individuals = sum(n_individuals_per_family)
    info = dict(n_families=n_families,
                n_individuals=n_individuals,
                samples_per_class=samples_per_class)

    logging.info(
        "%s individuals selected for %s families. "
        "~%s samples per class.", n_individuals, n_families,
        round(samples_per_class, 2))

    if return_np:
        x_train, x_test, y_train, y_test = train_test_split(pairs,
                                                            y,
                                                            test_size=0.2)
        return ((x_train, y_train), (x_test, y_test)), info

    train_dataset, val_dataset = build_datasets(pairs, y)

    return train_dataset, val_dataset


def build_pairs_relation(train_folder, relations, from_families):
    """Build pairs of individuals.

    Parameters
    ----------
    train_folder : str
        Path to train images.
    relations : pandas.DataFrame
        A DataFrame with postivei training pairs of individuals.
        Each individual is represented by FIDX/MIDY,
        where X is the family ID and Y is the individual ID.
    from_families : ndarray
        
    Returns
    -------
    pairs : pandas.DataFrame
        A DataFrame with pairs already in `relations` as well as
        negative pairs.
    """

    # Build list with all individuals
    individuals = []
    for family in tqdm(from_families, total=len(from_families)):
        for person in os.listdir(os.path.join(train_folder, family)):
            individuals.append(family + '/' + person)

    # Build a dataframe with all possible pairs of individuals
    possible_pairs = list(combinations(individuals, 2))
    possible_pairs = pd.DataFrame(possible_pairs, columns=['p1', 'p2'])

    # Filter same family pairs (some are negative, some are positive)
    # Positive pairs are in `relations`; we don't need the negatives there (?)
    nequal_fams = (
        possible_pairs.p1.str.split('/', expand=True)[0]  # get fid
        != possible_pairs.p2.str.split('/', expand=True)[0])
    negative_pairs = possible_pairs[nequal_fams]

    # Filter negatives to balance dataset
    n_positives = len(relations)
    # Sample some negatives...
    selected_negs_index = np.random.choice(negative_pairs.index,
                                           size=n_positives,
                                           replace=False)

    negative_pairs = negative_pairs.loc[selected_negs_index, :]

    negative_pairs['target'] = 0
    relations['target'] = 1
    pairs = relations.append(negative_pairs, ignore_index=True)

    logging.info('%s pairs selected.', pairs.shape[0])

    return pairs


def build_pairs_path(train_folder, pairs):
    """Build paths for image of each individual in `pairs``.

    This method just select a random image for each individual.

    Parameters
    ----------
    train_folder : str
        Path to train images.
    pairs : pandas.DataFrame
        A DataFrame with pairs of individuals and its labels.

    Returns
    -------
    pairs_path : pandas.DataFrame
        A DataFrame with pairs image path and its labels.
    """
    pairs_path = []

    def sample_img(train_folder, mid):
        folder = Path(train_folder) / mid
        path_imgs = list(folder.glob('*.jpg'))
        selected = None
        if path_imgs:
            selected = np.random.choice(path_imgs, size=1)[0]
        return selected

    for _, row in tqdm(pairs.iterrows(), total=pairs.shape[0]):
        mid1_img_path = sample_img(train_folder, row.p1)
        mid2_img_path = sample_img(train_folder, row.p2)
        if mid1_img_path and mid2_img_path:
            pairs_path.append(
                dict(p1=str(mid1_img_path),
                     p2=str(mid2_img_path),
                     target=str(row.target)))

    pairs_path = pd.DataFrame(pairs_path, columns=['p1', 'p2', 'target'])

    return pairs_path


def load_imgs(paths, target_size):
    """Load images from `paths`."""
    pairs = np.empty((len(paths), 2, *target_size), dtype=np.float32)
    for i, row in tqdm(paths.iterrows(), total=len(pairs)):
        img1 = img_to_array(load_img(row.p1, target_size=target_size)) / 255
        img2 = img_to_array(load_img(row.p2, target_size=target_size)) / 255
        pair = np.stack([img1, img2], axis=0)
        pairs[i, :] = pair

    y = paths.target.values.astype(np.uint8)

    return pairs, y


def generate(x, y, batch_size):
    """Return a batch generator."""
    n_samples = y.shape[0]
    n_batches = n_samples // batch_size
    n_batches += 1 if (n_batches * batch_size) < n_samples else 0
    if isinstance(x, list):
        x = np.stack(x, axis=1)

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = (1 + batch_idx) * batch_size
        x_batch = [
            x[batch_start:batch_end, 0, ...], x[batch_start:batch_end, 1, ...]
        ]
        y_batch = y[batch_start:batch_end]
        yield x_batch, y_batch


def build_datasets(pairs, y):
    """Return a `tf.data.Dataset` for training and validation."""
    x_train, x_val, y_train, y_val = train_test_split(pairs,
                                                      y,
                                                      test_size=0.2,
                                                      random_state=42)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        ((x_train[:, 0, ...], x_train[:, 1, ...]), y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices(
        ((x_val[:, 0, ...], x_val[:, 1, ...]), y_val))

    # repetitions = 1
    # train_dataset = (
    #     train_dataset.shuffle(100).batch(batch_size).repeat(repetitions)
    # )
    # val_dataset = (
    #     val_dataset.shuffle(100).batch(batch_size).repeat(repetitions)
    # )

    return train_dataset, val_dataset


def convert(pair, label):
    def _cvt(image):
        return tf.image.convert_image_dtype(image, tf.float32)

    image1, image2, = pair
    return (_cvt(image1), _cvt(image2)), label


def augment(pair, label):
    def augment_(image):
        image = tf.image.resize_with_crop_or_pad(image, 32, 32)
        image = tf.image.random_crop(image, size=[32, 32, 3])
        image = tf.image.random_brightness(image, max_delta=0.5)
        image = tf.image.resize(image, size=[64, 64])
        return image

    pair, label = convert(pair, label)
    image1, image2, = pair
    return (augment_(image1), augment_(image2)), label


def make_pair_input(
    pairs_path,
    cols,
    input_shape,
    batch_size,
    seed=None,
    data_gen_args=None,
):
    target_size = (input_shape[0], input_shape[1])
    seed = 42 if seed is None else seed

    datagen_p1 = ImageDataGenerator(**data_gen_args)
    datagen_p2 = ImageDataGenerator(**data_gen_args)

    if os.path.exists('image_pairs.npz'):
        loaded = np.load('image_pairs.npz')
        pairs, y = loaded['pairs'], loaded['y']
    else:
        pairs, y = load_imgs(pairs, target_size=input_shape)
        np.savez_compressed('image_pairs', pairs=pairs, y=y)

    datagen_p1.fit(pairs.reshape((-1, *input_shape)), augment=True, seed=seed)
    datagen_p2.fit(pairs.reshape((-1, *input_shape)), augment=True, seed=seed)

    flow_fd_args_train = dict(
        dataframe=pairs_path,
        target_size=target_size,
        batch_size=batch_size,
        directory=None,
        x_col=cols[0],
        y_col=cols[2],
        class_mode='binary',
        seed=seed,
        save_to_dir='data/augmented',
        subset='training',
        validate_filenames=False,
    )

    flow_fd_args_val = flow_fd_args_train.copy()

    gen_p1_train = datagen_p1.flow_from_dataframe(**flow_fd_args_train)

    flow_fd_args_train['x_col'] = cols[1]
    gen_p2_train = datagen_p2.flow_from_dataframe(**flow_fd_args_train)

    flow_fd_args_val['subset'] = 'validation'
    gen_p1_val = datagen_p1.flow_from_dataframe(**flow_fd_args_val)

    flow_fd_args_val['x_col'] = cols[1]
    gen_p2_val = datagen_p2.flow_from_dataframe(**flow_fd_args_val)

    train_sequence = PairInputSequence((gen_p1_train, gen_p2_train))
    val_sequence = PairInputSequence((gen_p1_val, gen_p2_val))

    return train_sequence, val_sequence


class PairInputSequence(utils.Sequence):
    def __init__(self, generators):
        self.gen_p1 = generators[0]
        self.gen_p2 = generators[1]

    def __len__(self):
        return self.gen_p1.__len__()

    def __getitem__(self, index):
        (p1_batch, y_batch) = self.gen_p1.__getitem__(index)
        (p2_batch, y_batch) = self.gen_p2.__getitem__(index)
        batch = ([p1_batch, p2_batch], y_batch)
        return batch
