import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread
import seaborn as sns

raw_image = imread('../photo_data/raw/BG-11C-6625-01-14-16-07-36.jpg')


def show_channels():
    """
    Function to show channels of test image separately
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].imshow(raw_image)
    axs[0, 0].set_title('3 channels image')

    axs[0, 1].imshow(np.expand_dims(raw_image[:, :, 0] / 255, 2), cmap='Greys')
    axs[0, 1].set_title('Red [0] channel')

    axs[1, 0].imshow(np.expand_dims(raw_image[:, :, 1] / 255, 2), cmap='Greys')
    axs[1, 0].set_title('Green [1] channel')

    axs[1, 1].imshow(np.expand_dims(raw_image[:, :, 2] / 255, 2), cmap='Greys')
    axs[1, 1].set_title('Blue [2] channel')

    plt.tight_layout()
    plt.show()


def show_channels_distribution():
    """
    Function to show values distribution of channels of test image separately
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].imshow(raw_image)
    axs[0, 0].set_title('3 channels image')

    sns.distplot(raw_image[:, :, 0] / 255, hist=True, ax=axs[0, 1])
    axs[0, 1].set_title('Red [0] channel')

    sns.distplot(raw_image[:, :, 1] / 255, hist=True, ax=axs[1, 0])
    axs[1, 0].set_title('Green [1] channel')

    sns.distplot(raw_image[:, :, 2] / 255, hist=True, ax=axs[1, 1])
    axs[1, 1].set_title('Blue [2] channel')

    plt.tight_layout()
    plt.show()
