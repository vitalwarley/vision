from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers


def dnn(input_shape, classes):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Dense(units=128,
                    activation='relu'))
    model.add(layers.Dense(units=classes, activation='sigmoid'))
    optimizer = optimizers.Adam(lr=1e-2)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    return model


def cnn(input_shape, classes):
    input = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 3, activation='relu', input_shape=input_shape)(input)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(classes, activation='sigmoid')(x)
    model = models.Model(input, x)
    # Compile model
    optimizer = optimizers.Adam(lr=1e-2)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    return model
