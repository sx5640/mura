"""
A standard DenseNet169 model
"""
import os
import sys

CURRENT_PATH = os.path.abspath(__file__)
CURRENT_PATH = os.path.dirname(CURRENT_PATH)
ROOT_PATH = os.path.dirname(CURRENT_PATH)
ROOT_PATH = os.path.dirname(ROOT_PATH)
sys.path.append(ROOT_PATH)

import keras
from keras.applications import imagenet_utils

import mura_model


class MobileNet(mura_model.MuraModel):
    """
    A 16 layers VGGNet Model object that designed to work with MURA dataset.
    """

    def __init__(self, resize=False, weights=None, grayscale=False, **kwargs):
        super(MobileNet, self).__init__(CURRENT_PATH, resize=resize, grayscale=grayscale)
        self.img_size_vgg = 224
        self.img_size = self.img_size_vgg if resize else self.img_size_origin
        self.model = self.build_model(weights)

    def build_model(self, weights):
        """
        Laying out the VGGNet model.
        :param weights: pretrained weight to import.
            Notes: if weight == "imagenet", will set image size to 224 and color to 3
        :return: keras model
        """
        print("****** Building Model")

        inputs = keras.layers.Input(
            (self.img_size, self.img_size, self.color_channel)
        )

        custom_weights = weights and os.path.isfile(weights)
        preload_weights = None
        if not custom_weights:
            preload_weights = weights

        preload_model = keras.applications.MobileNet(
            input_tensor=inputs,
            weights=preload_weights,
            include_top=False,
            pooling="avg"
        )

        output = keras.layers.Dense(
            1,
            activation="sigmoid",
            name="predictions"
        )(preload_model.output)

        model = keras.models.Model(
            inputs=inputs,
            outputs=[output],
            name="MobileNet"
        )

        if custom_weights:
            model.load_weights(weights, by_name=True)

        return model

    def load_and_process_image(self, path, imggen=None):
        """
        Load and preprocess a single image
        Args:
            path: path to image file.
            imggen: Image Generator for performing image perturbation

        Returns:
            image in ndarray
        """
        img = super().load_and_process_image(path, imggen)
        imagenet_utils.preprocess_input(img)
        return img


if __name__ == "__main__":
    MobileNet.train_from_cli()
