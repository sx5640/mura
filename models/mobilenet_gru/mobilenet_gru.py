"""
A model that replace the top of DenseNet169 with GRU units
"""
import os
import sys

CURRENT_PATH = os.path.abspath(__file__)
CURRENT_PATH = os.path.dirname(CURRENT_PATH)
ROOT_PATH = os.path.dirname(CURRENT_PATH)
ROOT_PATH = os.path.dirname(ROOT_PATH)
sys.path.append(ROOT_PATH)

import datetime
import math

import keras
from keras.applications import imagenet_utils
import numpy as np

import mura_model
import util


class MobileNetGRU(mura_model.MuraModel):
    """
    A 16 layers VGGNet Model object that designed to work with MURA dataset.
    """

    def __init__(self, resize=False, weights=None, grayscale=False, lock_bottom=False, **kwargs):
        super(MobileNetGRU, self).__init__(CURRENT_PATH, resize=resize, grayscale=grayscale)
        self.img_size_vgg = 224
        self.img_size = self.img_size_vgg if resize else self.img_size_origin
        self.model = self.build_model(weights, lock_bottom)

    def build_model(self, weights, lock_bottom):
        """
        Laying out the VGGNet model.
        :param weights: pretrained weight to import.
            Notes: if weight == "imagenet", will set image size to 224 and color to 3
        :param lock_bottom: lock bottom layers weights.
        :return: keras model
        """
        print("****** Building Model")

        inputs = keras.layers.Input(
            (11, self.img_size, self.img_size, self.color_channel)
        )

        custom_weights = weights and weights != "imagenet"
        preload_weights = weights
        if custom_weights:
            preload_weights = None

        preload_model = keras.applications.MobileNet(
            input_shape=(self.img_size, self.img_size, self.color_channel),
            weights=preload_weights,
            include_top=False,
            pooling="avg"
        )

        if lock_bottom:
            for layer in preload_model.layers:
                layer.trainable = False

        output = keras.layers.TimeDistributed(preload_model)(inputs)

        output = keras.layers.CuDNNGRU(12)(output)

        output = keras.layers.Dense(
            1,
            activation="sigmoid",
            name="predictions"
        )(output)

        model = keras.models.Model(
            inputs=inputs,
            outputs=[output],
            name="DenseNet169GRU"
        )

        if custom_weights:
            model.load_weights(weights, by_name=True, skip_mismatch=True)

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

    def input_generator(self, df, batch_size, is_train_data):
        """
        Generator that yields a batch of images with their labels
        Args:
            is_train_data: if the generator is used to generate training data.
                if True, will shuffle df each epoch and add image processing.
            df: Dataframe that contains all the images need to be loaded
            batch_size: Maximum number of images in each batch

        Yields: List of training images and labels in a batch

        """
        df = df.groupby("study").agg({
            "path": lambda x: x.tolist(),
            "label": np.prod}
        )

        imggen = None
        if is_train_data:
            print("****** Preparing Training Image Generator")
            imggen = self.prepare_imggen(df)

        while True:
            # shuffle the training set
            if is_train_data:
                df = df.sample(frac=1).reset_index(drop=True)
            for g, batch in df.groupby(np.arange(len(df)) // batch_size):
                # reset the index so that the index is based on position inside batch
                batch.reset_index(inplace=True)
                # seq_len = batch["path"].str.len().max()
                seq_len = 11
                effective_batch_size = min(batch_size, batch.shape[0])
                imgs = np.zeros((
                    effective_batch_size,
                    seq_len,
                    self.img_size,
                    self.img_size,
                    self.color_channel
                ))
                labels = []
                for index, row in batch.iterrows():
                    labels.append(row["label"])
                    paths = row["path"]
                    for i in range(len(paths)):
                        img = self.load_and_process_image(paths[i], imggen)
                        imgs[index, i, :, :, :] = img.reshape(1, 1, *img.shape)
                yield np.asarray(imgs), np.asarray(labels)

    def calc_steps(self, df, batch_size):
        """
        Calculate number of steps needed iterate over the dataframe
        via generator in each epoch given the batch size.
        Args:
            df: dataframe to iterate
            batch_size: batch size

        Returns: number of steps

        """
        return math.ceil(len(df["study"].unique()) / batch_size)

    def write_prediction(self, valid_df, batch_size):
        """
        Run prediction using given model on a list of images,
        and write the result to a csv file.
        :param batch_size: number of inputs in each batch.
        :param valid_df: validation dataset table
        :return:
            path to result result csv
        """
        predictions = self.model.predict_generator(
            self.img_generator(valid_df, batch_size, False),
            steps=math.ceil(valid_df.shape[0] / batch_size)
        )

        studies = valid_df["study"].unique()
        util.create_dir(self.result_path)
        for i in range(len(studies)):
            valid_df.loc[valid_df["study"] == studies[i], "prediction"] = predictions[i]

        result_path = os.path.join(self.result_path, "{}_{:%Y-%m-%d-%H%M}.csv".format(
            self.__class__.__name__, datetime.datetime.now()
        )
                                   )
        valid_df.to_csv(result_path)

        return valid_df


if __name__ == "__main__":
    MobileNetGRU.TRAIN_PARSER.add_argument("--lock_bottom", action="store_true",
                                           help="lock bottom layers weights.")
    MobileNetGRU.train_from_cli()
