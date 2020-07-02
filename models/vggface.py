"""
Source:
    - https://www.kaggle.com/ateplyuk/vggface-baseline-in-keras
"""
import keras
from keras_vggface.vggface import VGGFace

from models.distances import euclidean_distance, get_output_shape


def create_vggface_network(input_shape=None, **kwargs):
    """VGGFace extractor.
    """
    # pylint: disable=unused-argument
    input_a = keras.layers.Input(shape=input_shape)
    input_b = keras.layers.Input(shape=input_shape)

    vgg = VGGFace(include_top=False, input_shape=input_shape)

    for layer in vgg.layers:
        if not (layer.name.startswith('conv5') or layer.name == 'pool5'):
            layer.trainable = False

    last_layer = vgg.layers[-1].output
    flatten = keras.layers.Flatten()(last_layer)
    feature_extractor = keras.models.Model(vgg.input, flatten)

    features_a = feature_extractor(input_a)
    features_b = feature_extractor(input_b)
    embeddings = [features_a, features_b]

    distance = keras.layers.Lambda(euclidean_distance,
                                   output_shape=get_output_shape,
                                   name='distance')(embeddings)
    model = keras.models.Model([input_a, input_b], distance)

    return model
