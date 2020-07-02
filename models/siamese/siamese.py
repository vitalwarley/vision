"""Siamese net.
"""
from tensorflow.keras import layers
from tensorflow.keras import models

from models.distances import euclidean_distance, get_output_shape


def create_base_siamese(input_shape=None, **kwargs):
    # pylint: disable=unused-argument
    """Base siamese net.
    """
    inputs = layers.Input(shape=input_shape)
    out = layers.Conv2D(64, 10, activation='relu')(inputs)
    out = layers.MaxPooling2D()(out)
    out = layers.Conv2D(128, 7, activation='relu')(out)
    out = layers.MaxPool2D()(out)
    # x = layers.Conv2D(128, 4, activation='relu')(x)
    # x = layers.MaxPool2D()(x)
    # x = layers.Conv2D(256, 4, activation='relu')(x)
    out = layers.Flatten()(out)
    embeddings = layers.Dense(4096, activation='sigmoid')(out)
    model = models.Model(inputs=inputs, outputs=embeddings)
    return model


def create_siamese_network(input_shape=None, **kwargs):
    # pylint: disable=unused-argument
    """Siamese network for input pairs.
    """
    base_network = create_base_siamese(input_shape)

    left_input = layers.Input(shape=input_shape)
    right_input = layers.Input(shape=input_shape)

    left_embeddings = base_network(left_input)
    right_embeddings = base_network(right_input)
    embeddings = [left_embeddings, right_embeddings]

    distance = layers.Lambda(euclidean_distance,
                             output_shape=get_output_shape)(
                                 embeddings)

    model = models.Model([left_input, right_input], distance)

    return model

