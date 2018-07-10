"""
An abstract Model object that designed to work with MURA dataset.
"""
import abc
import argparse
import datetime
import math
import os
import pickle

import keras
import numpy as np
import pandas as pd

import dataset
import metric
import util


class MuraModel(abc.ABC):
    """
    An abstract Model object that designed to work with MURA dataset.
    """
    @classmethod
    @abc.abstractmethod
    def train_from_cli(cls, **kwargs):
        pass

    # Define argument parser so that the script can be executed directly
    # from console.
    ARG_PARSER = argparse.ArgumentParser("VGGNet model")
    SUBPARSER = ARG_PARSER.add_subparsers(help='sub-command help')
    # Arguments for training
    TRAIN_PARSER = SUBPARSER.add_parser("train")
    TRAIN_PARSER.set_defaults(func=train_from_cli)
    TRAIN_PARSER.add_argument("--resize", action="store_true",
                              help="resize image to original ImageNet size")

    TRAIN_PARSER.add_argument("--grayscale", action="store_true",
                              help="load image as grayscale instead of RGB")

    TRAIN_PARSER.add_argument("--load_param", action="store_true",
                              help="load parameters from pretrained model")

    TRAIN_PARSER.add_argument("-e", "--epochs", type=int,
                              help="number of epochs to run")

    TRAIN_PARSER.add_argument("-b", "--batch_size", type=int,
                              help="batch size")

    TRAIN_PARSER.add_argument("-d", "--decay", type=float,
                              help="decay per epoch")

    TRAIN_PARSER.add_argument("-w", "--weights", type=str,
                              help="path to pretrained weights to import")

    TRAIN_PARSER.add_argument("-l1", "--l1", type=float,
                              help="L1 regularization applied to each convolutional layer")

    TRAIN_PARSER.add_argument("-l2", "--l2", type=float,
                              help="L2 regularization applied to each convolutional layer")

    TRAIN_PARSER.add_argument("-bp", "--bpart", type=str,
                              help="body part to use for training and prediction")

    TRAIN_PARSER.add_argument("-np", "--num_pick", type=int,
                              help="number of images to pick from each patient")

    TRAIN_PARSER.add_argument("-lr", "--learning_rate", type=float,
                              help="learning rate")

    # Global Configs
    ROOT_PATH = os.path.abspath(__file__)  # ?/mura_model.py
    ROOT_PATH = os.path.dirname(ROOT_PATH)  # ?/

    def __init__(self, model_name, resize=True, grayscale=False, **kwargs):
        self.img_size_origin = 512
        self.img_size = None
        self.img_resized = resize
        self.img_grayscale = grayscale
        self.color_channel = 1 if grayscale else 3
        self.model_root_path = os.path.join(self.ROOT_PATH, "models", model_name)   # ?/models/{model_name}
        self.weight_path = os.path.join(self.ROOT_PATH, "weights")                  # ?/weights/
        self.model_save_path = os.path.join(self.model_root_path, "saved_models")   # ?/models/{model_name}/saved_models
        self.result_path = os.path.join(self.model_root_path, "results")            # ?/models/{model_name}/results
        self.cache_path = os.path.join(self.ROOT_PATH, "cache")                     # ?/cache
        self.log_path = os.path.join(self.model_root_path, "logs", model_name)      # /models/{model_name}/logs
        self.model = None

    def load_imgs(self, df):
        """
        Generator that loads all the images from the dataframe and and return them
        as ndarrays
        Args:
            df: Dataframe that contains all the images need to be loaded

        Yields: list of resized images in ndarray, list of labels, list of path

        """
        imgs = []
        labels = []
        paths = []
        for index, row in df.iterrows():
            img = dataset.load_image(row["path"], self.img_grayscale)
            if self.img_resized:
                img = dataset.resize_img(img, self.img_size)
            imgs.append(img)
            labels.append(row["label"])
            paths.append(row["path"])

        return np.asarray(imgs), np.asarray(labels), np.asarray(paths)

    def load_resources(self, bpart, num_pick):
        """
        Utility method that load all the resources needed for training.
        Will use csv/pickle/recreated images as cache to avoid recomputation.
        Args:
            bpart: Body part to pick
            num_pick: Number of images to pick from each study in training set

        Returns: train_df, valid_df, img_valid, label_valid, path_valid, flow_dir

        """
        train_table_path = os.path.join(
            self.cache_path,
            "training_table_{}_{}.csv".format(bpart, num_pick)
        )

        valid_table_path = os.path.join(
            self.cache_path,
            "valid_table_{}.csv".format(bpart)
        )

        # Load datasets from csvs. If not exist, recreate from dataset.py and save to csvs
        try:
            train_df = pd.read_csv(train_table_path, index_col=0)
            valid_df = pd.read_csv(valid_table_path, index_col=0)

        except FileNotFoundError:
            util.create_dir(self.cache_path)

            train_df, valid_df = dataset.preprocess()

            if bpart != "all":
                train_df = dataset.pick_bpart(train_df, bpart)
                valid_df = dataset.pick_bpart(valid_df, bpart)

            if num_pick > 0:
                train_df = dataset.pick_n_per_patient(train_df, num_pick)

            train_df.to_csv(train_table_path)
            valid_df.to_csv(valid_table_path)

        return train_df, valid_df

    def input_generator(self, df, batch_size, imggen=None):
        """
        Generator that yields a batch of training images with their labels
        Args:
            df: Dataframe that contains all the images need to be loaded
            batch_size: Maximum number of images in each batch
            imggen: ImageDataGenerator used to apply perturbation. If None,
                then no perturbation is applied.

        Yields: List of training images and labels in a batch

        """
        while True:
            # loop once per epoch
            df = df.sample(frac=1).reset_index(drop=True)
            for g, batch in df.groupby(np.arange(len(df)) // batch_size):
                imgs = []
                labels = []
                for index, row in batch.iterrows():
                    img = dataset.load_image(row["path"], self.img_grayscale)
                    if self.img_resized:
                        img = dataset.resize_img(img, self.img_size)
                    if imggen:
                        img = imggen.random_transform(img)
                    imgs.append(img)
                    labels.append(row["label"])

                yield np.asarray(imgs), np.asarray(labels)

    def img_generator(self, df, batch_size):
        """
        Generator that yields a batch of validation images with their labels
        Args:
            df: Dataframe that contains all the images need to be loaded
            batch_size: Maximum number of images in each batch

        Yields: List of training images and labels in a batch

        """
        while True:
            # loop once per epoch
            for g, batch in df.groupby(np.arange(len(df)) // batch_size):
                imgs = []
                for index, row in batch.iterrows():
                    img = dataset.load_image(row["path"], self.img_grayscale)
                    if self.img_resized:
                        img = dataset.resize_img(img, self.img_size)
                    imgs.append(img)

                yield np.asarray(imgs)

    def load_validation(self, valid_df, bpart):
        """
        Load Validation images into memory.
        TODO: Remove once the bug with validation_data=generator is resolved.
        Args:
            valid_df:
            bpart:

        Returns:

        """
        color_mode = "grayscale" if self.img_grayscale else "rgb"
        valid_pickle_path = os.path.join(
            self.cache_path,
            "valid_images_{}_{}_{}.pickle".format(
                bpart, self.img_size, color_mode
            )
        )
        try:
            with open(valid_pickle_path, "rb") as file:
                img_valid, label_valid, path_valid = pickle.load(file)
        except FileNotFoundError:
            img_valid, label_valid, path_valid = self.load_imgs(valid_df)
            with open(valid_pickle_path, "wb") as file:
                pickle.dump(
                    [img_valid, label_valid, path_valid],
                    file,
                    protocol=4
                )

        return img_valid, label_valid, path_valid

    def train(self, weight=None, bpart="all", num_pick=0,
              batch_size=32, epochs=50, learning_rate=0.0001, decay=0,
              l1=0.0, l2=0.0, **kwargs):
        """
        Build and train a VGGNet model.
        :param weight: path to pretrained weights.
        :param bpart: Body part to train on.
        :param num_pick: Number of images to pick per patient.
        :param batch_size: number of inputs in each batch.
        :param epochs: number of epochs to run before ending.
        :param learning_rate: initial learning rate.
        :param l1: L1 regularization applied to each convolutional layer.
        :param l2: L2 regularization applied to each convolutional layer.
        :param decay: decay per epoch, which learning rate is is calculated by
            lr *= (1. / (1. + decay * iterations))
        :param kwargs: extra parameters
        :return:
            A History object. Its History.history attribute is a record of training
            loss values and metrics values at successive epochs, as well as validation loss
            values and validation metrics values (if applicable).

            Path to where the model is been saved

            Path to where the result csv is been saved
        """
        # use binary_crossentropy loss and adam optimizer, same as MURA baseline model
        adam = keras.optimizers.Adam(
            lr=learning_rate, beta_1=0.9, beta_2=0.999,
            epsilon=None, decay=decay, amsgrad=False
        )

        global_recall = metric.BinaryRecall()
        global_kappa = metric.BinaryKappa()

        self.model.compile(
            loss='binary_crossentropy',
            optimizer=adam,
            metrics=[keras.metrics.binary_accuracy, metric.batch_recall, global_recall, global_kappa]
        )

        print("****** Preparing Input")
        train_df, valid_df = self.load_resources(bpart, num_pick)

        print("****** Preparing Training Image Generator")
        imggen_args = dict(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=30,
            fill_mode="constant",
            cval=0,
            horizontal_flip=True
        )
        input_img_gen = keras.preprocessing.image.ImageDataGenerator(**imggen_args)
        samples, _, _ = self.load_imgs(train_df.sample(1000))
        input_img_gen.fit(np.asarray(samples))

        # TODO: Remove this once the bug with `validation_data=input_generator` is resolved.
        print("****** Loading Validation Inputs")
        valid_imgs, valid_labels, _ = self.load_validation(valid_df, bpart)

        # log training time
        start_time = datetime.datetime.now()
        print("****** Starting Training: {:%H-%M-%S}".format(start_time))

        # Initiate TensorBoard callback
        log_path = os.path.join(self.log_path, "log_{}_{}_{}_{:%H-%M-%S}".format(
            bpart, num_pick, self.img_size, start_time
        ))
        util.create_dir(log_path)
        tfboard = keras.callbacks.TensorBoard(log_dir=log_path, write_grads=True)
        history = self.model.fit_generator(
            self.input_generator(train_df, batch_size, input_img_gen),
            steps_per_epoch=math.ceil(train_df.shape[0] / batch_size),
            epochs=epochs,
            verbose=2,
            validation_data=(valid_imgs, valid_labels),
            # validation_data=input_generator(valid_df, batch_size, img_size, grayscale),
            # validation_steps=math.ceil(valid_df.shape[0]/batch_size),
            callbacks=[tfboard],
            workers=4
        )
        print('****** Training time: %s' % (datetime.datetime.now() - start_time))

        print("****** Saving Model")
        # save model after success training
        util.create_dir(self.model_save_path)
        model_path = os.path.join(self.model_save_path, "vgg_{}_{}_{}_{:%Y-%m-%d-%H%M}.h5".format(
            bpart, num_pick, self.img_size, datetime.datetime.now()
        )
                                  )
        keras.models.save_model(
            self.model,
            model_path
        )

        print("****** Writing Predictions")
        # run prediction on validation set and save result in csv
        result_path = self.write_prediction(valid_df, batch_size)

        return history, model_path, result_path

    def write_prediction(self,  valid_df, batch_size,):
        """
        Run prediction using given model on a list of images,
        and write the result to a csv file.
        :param batch_size: number of inputs in each batch.
        :param valid_df: validation dataset table
        :return:
            path to result result csv
        """
        predictions = self.model.predict_generator(
            self.img_generator(valid_df, batch_size),
            steps=math.ceil(valid_df.shape[0] / batch_size)
        )
        util.create_dir(self.result_path)
        for i in range(len(predictions)):
            valid_df.at[i, "prediction"] = predictions[i]

        result_path = os.path.join(self.result_path, "vgg_{:%Y-%m-%d-%H%M}.csv".format(
            datetime.datetime.now()
        )
                                   )
        valid_df.to_csv(result_path)

        return result_path
