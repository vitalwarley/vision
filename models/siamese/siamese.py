from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import models

from models.distances import euclidean_distance, eucl_dist_output_shape


def create_base_siamese(input_shape):
    input = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 10, activation='relu')(input)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 7, activation='relu')(x)
    x = layers.MaxPool2D()(x)
    # x = layers.Conv2D(128, 4, activation='relu')(x)
    # x = layers.MaxPool2D()(x)
    # x = layers.Conv2D(256, 4, activation='relu')(x)
    x = layers.Flatten()(x)
    embeddings = layers.Dense(4096, activation='sigmoid')(x)
    model = models.Model(inputs=input, outputs=embeddings)
    return model


def create_siamese_network(input_shape):
    base_network = create_base_siamese(input_shape)

    left_input = layers.Input(shape=input_shape)
    right_input = layers.Input(shape=input_shape)

    left_embeddings = base_network(left_input)
    right_embeddings = base_network(right_input)
    embeddings = [left_embeddings, right_embeddings]

    distance = layers.Lambda(euclidean_distance,
                             output_shape=eucl_dist_output_shape)(
                                 embeddings)

    # prediction = layers.Dense(1, activation='sigmoid')(distance)
    # model = models.Model([left_input, right_input], prediction)
    model = models.Model([left_input, right_input], distance)

    return model

