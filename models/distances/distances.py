from keras import backend as K


def get_output_shape(shapes):
    """Utility for `euclidean_distance` and the like.

    When one uses `euclidean_distance` with `keras.layers.Lambda`,
    it is needed to set `output_shape=get_output_shape`.

    Parameters
    ----------
    shapes : tuple
        It contains two tuples, 
        one for each vector passed to `euclidean_distance`.

    Returns
    -------
    tuple
        Returns the output shape of the `euclidean_distance`.
    """
    shape1, _ = shapes
    return (shape1[0], 1)


def euclidean_distance(vects):
    """Computes the euclidean distance between two vectors.

    """
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))
