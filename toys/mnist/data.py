import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets.mnist import load_data as load_mnist


def create_pairs(x, digit_indices, num_classes, **kwargs):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = kwargs.get('samples_per_class')
    if n is None:
        n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = np.random.randint(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_sets(data, num_classes, **kwargs):
    # create training+test positive and negative pairs
    (x_train, y_train), (x_test, y_test) = data['train'], data['test']
    digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices, 10, **kwargs)

    digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    te_pairs, te_y = create_pairs(x_test, digit_indices, 10, **kwargs)
    
    return (tr_pairs, tr_y), (te_pairs, te_y)


def generate(x, y, batch_size, repetitions):
    n_samples = y.shape[0]
    n_batches = n_samples // batch_size
    n_batches += 1 if (n_batches * batch_size) < n_samples else 0
    for _ in range(repetitions):
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = (1 + batch_idx) * batch_size
            x_batch = [x[batch_start:batch_end, 0, ...],
                       x[batch_start:batch_end, 1, ...]]
            y_batch = y[batch_start:batch_end]
            yield x_batch, y_batch


def load(n_classes, input_shape, **kwargs):
    """Load mnist data."""
    # input_shape = kwargs.get('input_shape', (28, 28, 1))
    # n_classes = kwargs.get('n_classes', 0)
    print(f"Loading mnist with: shape={input_shape}, n_classes={n_classes}")

    (x_train, y_train), (x_test, y_test) = load_mnist()

    # Scale data
    x_train = x_train.astype(np.float64) / 255.
    x_test = x_test.astype(np.float64) / 255.

    # Reshape data
    input_shape = (-1,) + input_shape
    x_train = x_train.reshape(*input_shape)
    x_test = x_test.reshape(*input_shape)

    data = dict(train=(x_train, y_train), test=(x_test, y_test))

    return create_sets(data, n_classes, **kwargs)

