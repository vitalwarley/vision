"""
Source:
    - https://www.kaggle.com/ateplyuk/vggface-baseline-in-keras
"""
import keras

from models.distances import euclidean_distance, get_output_shape


def create_facenet_network(input_shape=None, **kwargs):
    """Facenet extractor.
    """
    # pylint: disable=unused-argument
    assert input_shape[0] >= 160
    assert input_shape[1] >= 160

    input_a = keras.layers.Input(shape=input_shape)
    input_b = keras.layers.Input(shape=input_shape)

    model_path = '/home/lativ/dev/vision/models/facenet/model/facenet_keras.h5'
    facenet = keras.models.load_model(model_path)

    for layer in facenet.layers[:-20]:
        layer.trainable = False

    last_layer = facenet.layers[-1].output
    # flatten = keras.layers.Flatten()(last_layer)
    # feature_extractor = keras.models.Model(facenet.input, flatten)
    feature_extractor = keras.models.Model(facenet.input, last_layer)

    features_a = feature_extractor(input_a)
    features_b = feature_extractor(input_b)
    embeddings = [features_a, features_b]

    distance = keras.layers.Lambda(euclidean_distance,
                                   output_shape=get_output_shape,
                                   name='distance')(embeddings)
    model = keras.models.Model([input_a, input_b], distance)

    return model
