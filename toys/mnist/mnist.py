"""Playing with mnist."""

import os
import time
import logging

import hiplot as hip

from data import load_data
from hyperparams import search
from models.simple import cnn, dnn
from utils import set_gpu, fetch_my_experiment


logging.getLogger('tensorflow').setLevel(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--simple', action='store_true')
    parser.add_argument('--cnn', action='store_true')
    args = parser.parse_args()

    # input_shape = (28, 28, 1)
    input_shape = (784,)
    n_classes = 10
    data = load_data(input_shape)
    os.system('clear')

    if args.simple:
        model = dnn(input_shape, 10)
        x_train, y_train = data['train']
        val = data['val']
        model.fit(x_train, y_train,
                  batch_size=64, epochs=10,
                  validation_data=val)
    elif args.cnn:
        model = cnn(input_shape, 10)
        x_train, y_train = data['train']
        val = data['val']
        model.fit(x_train, y_train,
                  batch_size=64, epochs=10,
                  validation_data=val)

    else:
        _ = search(data['train'], data['val'])
        hip.run_server([fetch_my_experiment])


if __name__ == '__main__':
    set_gpu()
    main() 
