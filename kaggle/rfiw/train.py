"""Train in RFIW."""
import sys
import logging

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers

from models.losses import contrastive_loss
from models.metrics import accuracy_for_distance as accuracy
from models.siamese import create_siamese_network
from models.vggface import create_vggface_network
from models.facenet import create_facenet_network
from data import load, generate
from utils import set_gpu

set_gpu()

logging.basicConfig(
    format="%(asctime)s:%(levelname)s - %(message)s", level=logging.INFO
)
# pylint: disable=invalid-name

# Model parameters
input_shape = 48  # also defines image dimensions
model_name = "siamese"
# TODO: hyperparameters

# Training parameters
epochs = 10
batch_size = 32

# Data parameters
n_classes = 0
samples_per_class = 0
AUTOTUNE = tf.data.experimental.AUTOTUNE

fit_params = dict()

# Create model
if model_name == "siamese":
    input_shape = (input_shape, input_shape, 3)
    model = create_siamese_network(input_shape=input_shape)
    # Prepare data
    train_dataset, val_dataset = load(
        n_classes=n_classes,
        samples_per_class=samples_per_class,
        input_shape=input_shape,
        return_np=False,
    )

    train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(1000)
    val_dataset = val_dataset.shuffle(1000).batch(batch_size).prefetch(1000)
    fit_params = dict(x=train_dataset, epochs=epochs, validation_data=val_dataset)
elif model_name == "vggface":
    # vggface only supports shapes starting from 48
    if input_shape < 48:
        sys.exit(1)

    input_shape = (input_shape, input_shape, 3)
    model = create_vggface_network(input_shape=input_shape)
elif model_name == "facenet":
    # facenet only supports shapes starting from 160
    if input_shape < 160:
        sys.exit(1)
    input_shape = (input_shape, input_shape, 3)
    model = create_facenet_network(input_shape=input_shape)

if model_name in ["vggface", "facenet"]:  # keras vs tf iterator ndim error
    (train_dataset, val_dataset), _ = load(
        n_classes=n_classes,
        samples_per_class=samples_per_class,
        input_shape=input_shape,
    )
    train_steps = len(train_dataset[1]) // batch_size
    val_steps = len(val_dataset[1]) // batch_size
    train_dataset = generate(*train_dataset, batch_size)
    val_dataset = generate(*val_dataset, batch_size)
    fit_params = dict(
        x=train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
    )

# Compile model
optimizer = optimizers.Adam()
model.compile(
    optimizer=optimizer, loss=contrastive_loss, metrics=[accuracy],
)

tw = np.sum([K.count_params(w) for w in model.trainable_weights])
ntw = np.sum([K.count_params(w) for w in model.non_trainable_weights])
logging.info("parameters (trainable): %s", tw)
logging.info("parameters (non-trainable): %s", ntw)

# Fit model
model.fit(**fit_params)
