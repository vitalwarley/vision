import os
import time
from datetime import datetime

from tensorflow.keras import callbacks
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
import kerastuner as kt

from callbacks import LRTensorBoard
from optimizers import scheduler


class HyperMNIST(kt.HyperModel):

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def build(self, hp):
        model = models.Sequential()
        model.add(layers.Input(shape=(784)))
        for dim in range(hp.Int('num_layers', 2, 5)):
            model.add(layers.Dense(units=hp.Int('units',
                                         min_value=32,
                                         max_value=512,
                                         step=32),
                            activation='relu'))
        model.add(layers.Dense(units=10, activation='sigmoid'))

        # Compile model
        sgd = optimizers.SGD(lr=hp.Choice('learning_rate',
                                          values=[1e-2, 1e-3, 1e-4]))
        model.compile(optimizer=sgd,
                      loss='sparse_categorical_crossentropy',
                      metrics=['acc'])
        return model


def build_search():
    # Load model
    hypermodel = HyperMNIST(10)
    tuner = kt.RandomSearch(hypermodel, objective='val_acc',
                         max_trials=5, executions_per_trial=3,
                         directory='./tuner', project_name='mnist')

    tuner.search_space_summary()

    return tuner


def search(train, val):

    # TODO: how to discriminate between trials?
    log_dir = './logs/' + datetime.now().strftime('%Y%m%d-%H%M%S')
    cp_dir = './checkpoints/' + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    os.makedirs(cp_dir, exist_ok=True)

    tb_cb = LRTensorBoard(log_dir=log_dir)
    es_cb = callbacks.EarlyStopping(patience=2)
    mc_cb = callbacks.ModelCheckpoint(cp_dir,
                                      save_weights_only=True,
                                      save_best_only=True)
    lrs_cb = callbacks.LearningRateScheduler(scheduler, verbose=0)
    cbs = [
        #tb_cb,
        #es_cb,
        #mc_cb,
        lrs_cb,
    ]

    time.sleep(1)

    x_train, y_train = train

    tuner = build_search()
    tuner.search(x_train, y_train,
                 batch_size=64, epochs=10,
                 validation_data=val, callbacks=cbs,
                 verbose=0)

    return tuner

