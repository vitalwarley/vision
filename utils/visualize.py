import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')

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
    _, axarr = plt.subplots(nrows=2,
                            ncols=n_pairs,
                            figsize=(20, 20),
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


def visualize_distances(results, dataset, model, save_name='results.png'):
    ncols = 4
    nrows = (len(results) // ncols) + 1
    _, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(15, 8),
        squeeze=False,
        constrained_layout=True
    )

    for i, result in enumerate(results):
        distances, targets, info = result
        n_samples, n_classes, n_individuals, loss, acc = (
            info['samples_per_class'], info['n_individuals'],
            info['n_families'], info['loss'], info['acc'])
        ax = axs[i // ncols][i % ncols]
        ylim = 0
        annotations = (f"Families: {n_classes}\n"
                       f"Individuals: {n_individuals}\n"
                       f"Samples per class: ~{n_samples:.2f}\n"
                       f"Loss: {loss:.2f}, Acc: {acc:.2f}")

        for c in range(2):
            try:
                sns.distplot(distances[targets == c],
                             kde=True,
                             label=str(c),
                             ax=ax)
            except (ValueError, RuntimeError):
                pass

            if c == 1:
                _, ylim_ = ax.get_ylim()
                if ylim_ > ylim:
                    ylim = ylim_

                y = ylim - (0.15 * ylim)
                ax.annotate(annotations, xy=(-0.9, y), fontsize='x-small')

            _, ylim = ax.get_ylim()

        ax.set_xlim((-1, 2))
        ax.set_title(i)

    plt.suptitle(f"{dataset}: {model} net distances")
    if save_name:
        plt.savefig(save_name, dpi=300)
    else:
        plt.show()


def plot_augmented_pairs(datagen, pairs):
    """Augment and plot a given set of pairs.

    Parameters
    ---------
    datagen : ImageDataGenerator
    pairs : pandas.DataFrame
    """
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
    pass


def visualize(original, augmented):
    _ = plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original image')
    plt.imshow(original)
    plt.subplot(1, 2, 2)
    plt.title('Augmented image')
    plt.imshow(augmented)
    plt.show()
