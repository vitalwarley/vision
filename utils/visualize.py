import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_history(history):
    loss, acc, val_loss, val_acc = history.history.keys()
    #  "Accuracy"
    plt.plot(history.history[acc])
    plt.plot(history.history[val_acc])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history[loss])
    plt.plot(history.history[val_loss])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def imshow(img):
    plt.imshow(img)
    plt.show()


def visualize_pairs(pairs, y):

    n_pairs = len(pairs)
    _, axarr = plt.subplots(nrows=2, ncols=n_pairs, figsize=(20, 20),
                            squeeze=False)

    for a in range(n_pairs):
        axarr[0, a].set_title("Family : " + str(y[a]))
        axarr[0, a].imshow(pairs[a, 0, ...])
        axarr[1, a].imshow(pairs[a, 1, ...])
        axarr[0, a].xaxis.set_visible(False)
        axarr[0, a].yaxis.set_visible(False)
        axarr[1, a].xaxis.set_visible(False)
        axarr[1, a].yaxis.set_visible(False)
    plt.show()



def visualize_distance_distribution(distances, targets, n_classes):
    for c in range(n_classes):
        sns.kdeplot(distances[targets == c], label=str(c))
    plt.show()


def plot_augmented_pairs(datagen, pairs):
    """Augment and plot a given set of pairs.

    Parameters
    ---------
    datagen : ImageDataGenerator
    pairs : pandas.DataFrame
    """
    pass
    # pairs['target'] = pairs.target.astype(str)
    # data_gen_args = dict(featurewise_center=True,
    #                      featurewise_std_normalization=True,
    #                      rotation_range=90,
    #                      width_shift_range=0.1,
    #                      height_shift_range=0.1,
    #                      zoom_range=0.2,
    #                      validation_split=0.2)

    # train_sequence, val_sequence = make_pair_input(
    #     pairs,
    #     pairs.columns,
    #     input_shape,
    #     BATCH_SIZE,
    #     seed=42,
    #     data_gen_args=data_gen_args,
    # )

if __name__ == '__main__':
    loaded = np.load('image_pairs.npz')
    pairs, y = loaded['pairs'], loaded['y']
    idxs = np.random.choice(np.arange(len(y)), 6, replace=False)
    visualize_pairs(pairs[idxs, ...], y[idxs, ...])

