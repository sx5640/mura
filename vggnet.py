"""
A simple implementation of VGGNet.
"""
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


IMG_SIZE = 512
VGG_INPUT_SIZE = 224
WEIGHT_PATH = os.path.abspath(__file__)                                     # ?/vggnet.py
ROOT_PATH = os.path.dirname(WEIGHT_PATH)                                    # ?/
WEIGHT_PATH = os.path.join(ROOT_PATH, "weights")                            # ?/weights/
WEIGHT_PATH_16 = os.path.join(WEIGHT_PATH, "vgg16_weights.h5")              # ?/weights/vgg16_weights.h5
MODEL_PATH = os.path.join(ROOT_PATH, "models", "vgg")                       # ?/models/vgg
RESULT_PATH = os.path.join(ROOT_PATH, "results", "vgg")                     # ?/results/vgg
DATA_CACHE_PATH = os.path.join(ROOT_PATH, "cache")                          # ?/cache
DATA_LOG_PATH = os.path.join(ROOT_PATH, "logs", "vgg")                      # ?/log/vgg


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


def build_model(img_size, color_channel, l1, l2):
    """
    Laying out the VGGNet model.
    :param img_size: size of the input image.
    :param color_channel: number of colour channels
    :param l1: L1 regularization applied to each convolutional layer.
    :param l2: L2 regularization applied to each convolutional layer.
    :return: keras model
    """
    img_tensor = keras.layers.Input((img_size, img_size, color_channel))

    tensor = conv_block(img_tensor, 64, 2, 3, "relu", l1, l2)
    tensor = conv_block(tensor, 128, 2, 3, "relu", l1, l2)
    tensor = conv_block(tensor, 256, 3, 3, "relu", l1, l2)
    tensor = conv_block(tensor, 512, 3, 3, "relu", l1, l2)
    tensor = conv_block(tensor, 512, 3, 3, "relu", l1, l2)
    tensor = keras.layers.Flatten(name="flatten")(tensor)
    tensor = keras.layers.Dense(256, activation="relu")(tensor)
    tensor = keras.layers.Dropout(0.5)(tensor)
    tensor = keras.layers.Dense(1, activation="sigmoid", name="predictions")(tensor)

    model = keras.models.Model(inputs=img_tensor, outputs=tensor, name="VGG")

    return model


def load_imgs(df, grayscale, img_size):
    """
    Generator that loads all the images from the dataframe and and return them
    as ndarrays
    Args:
        df: Dataframe that contains all the images need to be loaded
        grayscale: Import image as grayscale or RGB
        img_size: Image size to resize the image to if not equals IMG_SIZE

    Yields: list of resized images in ndarray, list of labels, list of path

    """
    imgs = []
    labels = []
    paths = []
    for index, row in df.iterrows():
        img = dataset.load_image(row["path"], grayscale)
        if img_size != IMG_SIZE and img_size > 0:
            img = dataset.resize_img(img, img_size)
        imgs.append(img)
        labels.append(row["label"])
        paths.append(row["path"])

    return np.asarray(imgs), np.asarray(labels), np.asarray(paths)


def load_resources(bpart, num_pick):
    """
    Utility method that load all the resources needed for training.
    Will use csv/pickle/recreated images as cache to avoid recomputation.
    Args:
        bpart: Body part to pick
        num_pick: Number of images to pick from each study in training set

    Returns: train_df, valid_df, img_valid, label_valid, path_valid, flow_dir

    """
    train_table_path = os.path.join(
        DATA_CACHE_PATH,
        "training_table_{}_{}.csv".format(bpart, num_pick)
    )

    valid_table_path = os.path.join(
        DATA_CACHE_PATH,
        "valid_table_{}.csv".format(bpart)
    )

    # Load datasets from csvs. If not exist, recreate from dataset.py and save to csvs
    try:
        train_df = pd.read_csv(train_table_path, index_col=0)
        valid_df = pd.read_csv(valid_table_path, index_col=0)

    except FileNotFoundError:
        util.create_dir(DATA_CACHE_PATH)

        train_df, valid_df = dataset.preprocess()

        if bpart != "all":
            train_df = dataset.pick_bpart(train_df, bpart)
            valid_df = dataset.pick_bpart(valid_df, bpart)

        if num_pick > 0:
            train_df = dataset.pick_n_per_patient(train_df, num_pick)

        train_df.to_csv(train_table_path)
        valid_df.to_csv(valid_table_path)

    return train_df, valid_df


def input_generator(df, batch_size, img_size, grayscale, imggen=None):
    """
    Generator that yields a batch of training images with their labels
    Args:
        df: Dataframe that contains all the images need to be loaded
        batch_size: Maximum number of images in each batch
        grayscale: Import image as grayscale or RGB
        img_size: Image size to resize the image to if not equals IMG_SIZE
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
                img = dataset.load_image(row["path"], grayscale)
                if img_size != IMG_SIZE and img_size > 0:
                    img = dataset.resize_img(img, img_size)
                if imggen:
                    img = imggen.random_transform(img)
                imgs.append(img)
                labels.append(row["label"])

            yield np.asarray(imgs), np.asarray(labels)


def img_generator(df, batch_size, img_size, grayscale):
    """
    Generator that yields a batch of validation images with their labels
    Args:
        df: Dataframe that contains all the images need to be loaded
        batch_size: Maximum number of images in each batch
        grayscale: Import image as grayscale or RGB
        img_size: Image size to resize the image to if not equals IMG_SIZE

    Yields: List of training images and labels in a batch

    """
    while True:
        # loop once per epoch
        for g, batch in df.groupby(np.arange(len(df)) // batch_size):
            imgs = []
            for index, row in batch.iterrows():
                img = dataset.load_image(row["path"], grayscale)
                if img_size != IMG_SIZE and img_size > 0:
                    img = dataset.resize_img(img, img_size)
                imgs.append(img)

            yield np.asarray(imgs)


def load_validation(valid_df, bpart, grayscale, img_size):
    """
    Load Validation images into memory. To be removed once the bug with
    validation_data=generator is resolved.
    Args:
        valid_df:
        bpart:
        grayscale:
        img_size:

    Returns:

    """
    color_mode = "grayscale" if grayscale else "rgb"
    valid_pickle_path = os.path.join(
        DATA_CACHE_PATH,
        "valid_images_{}_{}_{}.pickle".format(
            bpart, img_size, color_mode
        )
    )
    try:
        with open(valid_pickle_path, "rb") as file:
            img_valid, label_valid, path_valid = pickle.load(file)
    except FileNotFoundError:
        img_valid, label_valid, path_valid = load_imgs(valid_df, color_mode, img_size)
        with open(valid_pickle_path, "wb") as file:
            pickle.dump(
                [img_valid, label_valid, path_valid],
                file,
                protocol=4
            )

    return img_valid, label_valid, path_valid


def train(resize=True, load_param=False, grayscale=False, bpart="all", num_pick=0,
          batch_size=32, epochs=50, learning_rate=0.0001, decay=0,
          l1=0.0, l2=0.0, **kwargs):
    """
    Build and train a VGGNet model.
    :param resize: resize image to original ImageNet size.
    :param load_param: load parameters from pretrained model.
    :param grayscale: Import image as grayscale or RGB
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
    # prepare training params
    img_size = VGG_INPUT_SIZE if resize else IMG_SIZE

    color_channel = 1 if grayscale else 3

    print("****** Building Model")
    model = build_model(img_size, color_channel, l1, l2)

    # use binary_crossentropy loss and adam optimizer, same as MURA baseline model
    adam = keras.optimizers.Adam(
        lr=learning_rate, beta_1=0.9, beta_2=0.999,
        epsilon=None, decay=decay, amsgrad=False
    )

    global_recall = metric.BinaryRecall()
    global_kappa = metric.BinaryKappa()

    model.compile(
        loss='binary_crossentropy',
        optimizer=adam,
        metrics=[keras.metrics.binary_accuracy, metric.batch_recall, global_recall, global_kappa]
    )

    print("****** Preparing Input")
    train_df, valid_df = load_resources(bpart, num_pick)

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
    samples, _, _ = load_imgs(train_df.sample(1000), grayscale, img_size)
    input_img_gen.fit(np.asarray(samples))

    # TODO: Remove this once the bug with `validation_data=input_generator` is resolved.
    print("****** Loading Validation Inputs")
    valid_imgs, valid_labels, _ = load_validation(valid_df, bpart, grayscale, img_size)

    # log training time
    start_time = datetime.datetime.now()
    print("****** Starting Training: {:%H-%M-%S}".format(start_time))

    # Initiate TensorBoard callback
    log_path = os.path.join(DATA_LOG_PATH, "log_{}_{}_{}_{:%H-%M-%S}".format(
        bpart, num_pick, img_size, start_time
    ))
    util.create_dir(log_path)
    tfboard = keras.callbacks.TensorBoard(log_dir=log_path, write_grads=True)
    history = model.fit_generator(
        input_generator(train_df, batch_size, img_size, grayscale, input_img_gen),
        steps_per_epoch=math.ceil(train_df.shape[0]/batch_size),
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
    util.create_dir(MODEL_PATH)
    model_path = os.path.join(MODEL_PATH, "vgg_{}_{}_{}_{:%Y-%m-%d-%H%M}.h5".format(
            bpart, num_pick, img_size, datetime.datetime.now()
        )
    )
    keras.models.save_model(
        model,
        model_path
    )

    print("****** Writing Predictions")
    # run prediction on validation set and save result in csv
    result_path = write_prediction(model, valid_df, batch_size, img_size, grayscale)

    return history, model_path, result_path


def write_prediction(model, valid_df, batch_size, img_size, grayscale):
    """
    Run prediction using given model on a list of images,
    and write the result to a csv file.
    :param model: model to run prediction on
    :param batch_size: number of inputs in each batch.
    :param valid_df: validation dataset table
    :param img_size: Image size to resize the image to if not equals IMG_SIZE
    :param grayscale: Import image as grayscale or RGB
    :return:
        path to result result csv
    """
    predictions = model.predict_generator(
        img_generator(valid_df, batch_size, img_size, grayscale),
        steps=math.ceil(valid_df.shape[0]/batch_size)
    )
    util.create_dir(RESULT_PATH)
    for i in range(len(predictions)):
        valid_df.at[i, "prediction"] = predictions[i]

    result_path = os.path.join(RESULT_PATH, "vgg_{:%Y-%m-%d-%H%M}.csv".format(
            datetime.datetime.now()
        )
    )
    valid_df.to_csv(result_path)

    return result_path


if __name__ == "__main__":
    # Define argument parser so that the script can be executed directly
    # from console.
    ARG_PARSER = argparse.ArgumentParser("VGGNet model")
    SUBPARSER = ARG_PARSER.add_subparsers(help='sub-command help')
    # Arguments for training
    TRAIN_PARSER = SUBPARSER.add_parser("train")
    TRAIN_PARSER.set_defaults(func=train)
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

    # parse argument
    ARGS = ARG_PARSER.parse_args()
    ARG_DICT = {k: v for k, v in vars(ARGS).items() if v is not None}
    ARGS.func(**ARG_DICT)
