"""
Visualization of a given model
"""
import argparse

import dataset

import cv2
import imageio
import keras
import matplotlib.pyplot as plt
import numpy as np
import vis.utils.utils as vutils
import vis.visualization as vvis


def import_model(model_path):
    """
    Utility method for importing a model given its path.
    :param model_path: path to model file.
    :return: Keras model
    """
    return keras.models.load_model(
        model_path,
        compile=False
    )


def get_seed_image(bpart, img_size, img_path):
    """
    Utility method for getting a seeding image to visualize attention.
    Pick a random image from validation set, unless img_path is specified.
    :param bpart: Body part to pick.
    :param img_size: Image size to reshape to.
    :param img_path: path to image file.
    :return: Keras model
    """
    if not img_path:
        _, valid_labeled, _, valid_path = dataset.load_dataframe()
        df_valid = dataset.build_dataframe(valid_labeled, valid_path)
        df_valid = df_valid[df_valid["body_part"] == bpart]
        img_path = df_valid["path"].sample(1).iloc[0]

    img = imageio.imread(img_path)
    img = dataset.grayscale(img)
    img = dataset.zero_pad(img)
    return cv2.resize(img, (img_size, img_size)).reshape((img_size, img_size, 1))


def plt_saliency(model, img, ax, idx):
    """
    Plot saliency graph, which generates an image that represents
    the highest activation based on a seeding image;

    Reference: https://arxiv.org/pdf/1312.6034v2.pdf

    Args:
        model: Model to plot.
        img: Seed image.
        ax: Matplotlib axis.
        idx: Index of the plot to be shown on the axis.

    Returns: None

    """
    pred_layer_idx = vutils.find_layer_idx(model, "predictions")

    sal = vvis.visualize_saliency(
        model, pred_layer_idx, filter_indices=None, seed_input=img
    )

    ax[idx].imshow(sal, cmap='jet')
    ax[idx].set_title("Saliency")


def plt_cam(model, img, ax, idx, layer_idx=None):
    """
    Plot Class Activation Map(CAM), which represents the activation
    at the end of all convolutional layer;

    Reference: https://arxiv.org/pdf/1610.02391v1.pdf

    Args:
        model: Model to plot.
        img: Seed image.
        ax: Matplotlib axis.
        idx: Index of the plot to be shown on the axis.
        layer_idx: Index of the layer to plot Grad-CAM. Optional.

    Returns: None

    """
    pred_layer_idx = vutils.find_layer_idx(model, "predictions")
    hmap = vvis.visualize_cam(
        model, pred_layer_idx, filter_indices=None, seed_input=img
    )
    ax[idx].imshow(vvis.overlay(hmap, np.stack((img.reshape(img.shape[0:2]),)*3, -1)))
    ax[idx].set_title("Heatmap")


def plt_attention(model_path, img_path, bpart, img_size, **kwargs):
    """
    Plot attention graph, including saliency and CAM.

    :param model_path: Path to the model
    :param img_path: Path to a validation image. Optional
    :param bpart: Body part to pick if img_path not given
    :param img_size: Size of the image to reshape to
    :param kwargs:
    :return:
    """
    model = import_model(model_path)
    img = get_seed_image(bpart, img_size, img_path)

    f, ax = plt.subplots(1, 3)

    # plot input image
    ax[0].imshow(img.reshape(img.shape[0:2]), cmap='gray')
    ax[0].set_title("Input")

    # plot saliency
    plt_saliency(model, img, ax, 1)

    # plot heatmap
    plt_cam(model, img, ax, 2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Define argument parser so that the script can be executed directly
    # from console.
    ARG_PARSER = argparse.ArgumentParser("VGGNet model")
    PARENT_PARSER = argparse.ArgumentParser(add_help=False)
    SUBPARSER = ARG_PARSER.add_subparsers(help='sub-command help')

    # Shared arguments
    PARENT_PARSER.add_argument(
        "-m", "--model_path", type=str, required=True, help="path to model file."
    )

    PARENT_PARSER.add_argument(
        "-i", "--img_path", type=str, default=None,
        help="path to image file. If set, use given image instead "
             "of a random on from validation set"
    )

    PARENT_PARSER.add_argument(
        "-is", "--img_size", type=int, default=512, help="image size to reshape to"
    )

    PARENT_PARSER.add_argument(
        "-bp", "--bpart", type=str, default="all",
        help="body part to use for training and prediction"
    )

    # Arguments for plotting saliency
    SALIENCY_PARSER = SUBPARSER.add_parser("attention", parents=[PARENT_PARSER])
    SALIENCY_PARSER.set_defaults(func=plt_attention)

    # parse argument
    ARGS = ARG_PARSER.parse_args()
    ARGS.func(**vars(ARGS))
