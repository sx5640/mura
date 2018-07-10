"""

"""
import os
import sys

ROOT_PATH = os.path.abspath(__file__)
ROOT_PATH = os.path.dirname(ROOT_PATH)
ROOT_PATH = os.path.dirname(ROOT_PATH)
ROOT_PATH = os.path.dirname(ROOT_PATH)
sys.path.append(ROOT_PATH)

import keras

import mura_model


class VGGNet16(mura_model.MuraModel):
    """
    A 16 layers VGGNet Model object that designed to work with MURA dataset.
    """

    def __init__(self, resize=True, weight=None, grayscale=False, l1=0.0, l2=0.0, **kwargs):
        super(VGGNet16, self).__init__("vggnet16", resize=resize, grayscale=grayscale)
        self.img_size_vgg = 224
        self.img_size = self.img_size_vgg if resize else self.img_size_origin
        self.model = self.build_model(self.img_size, self.color_channel, l1, l2)

    @classmethod
    def train_from_cli(cls):
        args = cls.ARG_PARSER.parse_args()
        arg_dict = {k: v for k, v in vars(args).items() if v is not None}
        model = cls(**arg_dict)
        model.train(**arg_dict)

    @staticmethod
    def conv_block(tensor, depth, num_layers, filter_size, activation, l1, l2):
        """
        Define a convolution block that can be recycled.
        :param tensor: input tensor.
        :param depth: number of filters in each layer.
        :param num_layers: number of convolutional layers in the block.
        :param filter_size: size of the filter in each layer.
        :param activation: activation function in each layer.
        :param l1: L1 regularization applied to each convolutional layer.
        :param l2: L2 regularization applied to each convolutional layer.
        :return: output tensor
        """

        reg = keras.regularizers.l1_l2(l1, l2)
        for _ in range(num_layers):
            tensor = keras.layers.Conv2D(
                depth,
                (filter_size, filter_size),
                padding="same",
                activation=activation,
                kernel_regularizer=reg
            )(tensor)
        return keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(tensor)

    def build_model(self, img_size, color_channel, l1, l2):
        """
        Laying out the VGGNet model.
        :param img_size: size of the input image.
        :param color_channel: number of colour channels
        :param l1: L1 regularization applied to each convolutional layer.
        :param l2: L2 regularization applied to each convolutional layer.
        :return: keras model
        """
        print("****** Building Model")
        img_tensor = keras.layers.Input((img_size, img_size, color_channel))

        tensor = self.conv_block(img_tensor, 64, 2, 3, "relu", l1, l2)
        tensor = self.conv_block(tensor, 128, 2, 3, "relu", l1, l2)
        tensor = self.conv_block(tensor, 256, 3, 3, "relu", l1, l2)
        tensor = self.conv_block(tensor, 512, 3, 3, "relu", l1, l2)
        tensor = self.conv_block(tensor, 512, 3, 3, "relu", l1, l2)
        tensor = keras.layers.Flatten(name="flatten")(tensor)
        tensor = keras.layers.Dense(256, activation="relu")(tensor)
        tensor = keras.layers.Dropout(0.5)(tensor)
        tensor = keras.layers.Dense(1, activation="sigmoid", name="predictions")(tensor)

        return keras.models.Model(inputs=img_tensor, outputs=tensor, name="VGG")


if __name__ == "__main__":
    VGGNet16.train_from_cli()