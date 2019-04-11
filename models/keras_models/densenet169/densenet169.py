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

from models.keras_models import abs_model


class DenseNet169(abs_model.KerasModel):
    """
    A 16 layers VGGNet Model object that designed to work with MURA dataset.
    """

    def __init__(self, img_size=224, weight=None, grayscale=False, **kwargs):
        super(DenseNet169, self).__init__(CURRENT_PATH, img_size=img_size, grayscale=grayscale)
        self.img_size_vgg = 224
        self.img_size = img_size
        self.model = self.build_model(weight)

    def build_model(self, weight):
        """
        Laying out the VGGNet model.
        :param weight: pretrained weight to import.
            Notes: if weight == "imagenet", will set image size to 224 and color to 3
        :return: keras model
        """
        print("****** Building Model")

        inputs = keras.layers.Input(
            (self.img_size, self.img_size, self.color_channel)
        )
        preload_model = keras.applications.DenseNet169(
            input_tensor=inputs,
            weights=weight,
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
            name="DenseNet169"
        )
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
    DenseNet169.train_from_cli()
