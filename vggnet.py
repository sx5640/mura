"""
A simple implementation of VGGNet.
"""
import argparse
import csv
import datetime
import os

import keras as K
import numpy as np

import dataset
import util


IMG_SIZE = 512
VGG_INPUT_SIZE = 224
WEIGHT_PATH = os.path.abspath(__file__)                                     # ?/vggnet.py
ROOT_PATH = os.path.dirname(WEIGHT_PATH)                                    # ?/
WEIGHT_PATH = os.path.join(ROOT_PATH, "weights")                            # ?/weights/
WEIGHT_PATH_16 = os.path.join(WEIGHT_PATH, "vgg16_weights.h5")              # ?/weights/vgg16_weights.h5
MODEL_PATH = os.path.join(ROOT_PATH, "models", "vgg")                       # ?/models/vgg
RESULT_PATH = os.path.join(ROOT_PATH, "results", "vgg")                     # ?/results/vgg
DATA_CACHE_PATH = os.path.join(ROOT_PATH, "cache", "vgg")                   # ?/cache/vgg
DATA_CACHE_FILE_RS = os.path.join(DATA_CACHE_PATH, "vggnet-resized.npy")    # ?/cache/vgg/vggnet-resized.npy
DATA_CACHE_FILE = os.path.join(DATA_CACHE_PATH, "vggnet-resized.npy")       # ?/cache/vgg/vggnet.npy
DATA_LOG_PATH = os.path.join(ROOT_PATH, "logs", "vgg")                      # ?/log/vgg


def conv_block(tensor, depth, num_layers, filter_size, activation):
    """
    Define a convolution block that can be recycled.
    :param tensor: input tensor.
    :param depth: number of filters in each layer.
    :param num_layers: number of convolutional layers in the block.
    :param filter_size: size of the filter in each layer.
    :param activation: activation function in each layer.
    :return: output tensor
    """
    for _ in range(num_layers):
        tensor = K.layers.Conv2D(
            depth,
            (filter_size, filter_size),
            padding="same",
            activation=activation
        )(tensor)
    return K.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(tensor)


def build_model(img_size):
    """
    Laying out the VGGNet model.
    :param img_size: size of the input image.
    :return: keras model
    """
    img_tensor = K.layers.Input((img_size, img_size, 1))

    tensor = conv_block(img_tensor, 64, 2, 3, "relu")
    tensor = conv_block(tensor, 128, 2, 3, "relu")
    tensor = conv_block(tensor, 256, 3, 3, "relu")
    tensor = conv_block(tensor, 512, 3, 3, "relu")
    tensor = conv_block(tensor, 512, 3, 3, "relu")
    tensor = K.layers.Flatten()(tensor)
    tensor = K.layers.Dense(256, activation="relu")(tensor)
    tensor = K.layers.Dropout(0.5)(tensor)
    tensor = K.layers.Dense(1, activation="sigmoid")(tensor)

    model = K.models.Model(inputs=img_tensor, outputs=tensor, name="VGG")

    return model


def select_input(table):
    """
    Select and prepare input for given dataset.
    :param table: dataset to process.
    :return: list of image, labels, path
    """
    result_table = dataset.pick_bpart(table, "elbow")
    result_table = dataset.pick_n_per_patient(result_table, 0)

    return dataset.load_images(result_table)


def prepare_input():
    """
    Prepare input data for both training and validation dataset.
    :return:
    """
    train_table, valid_table = dataset.preprocess()

    img_train, label_train, path_train = select_input(train_table)

    img_valid, label_valid,  path_valid = select_input(valid_table)

    return img_train, label_train, path_train, \
        img_valid, label_valid, path_valid


def train(resize, load_param, batch_size, epochs, learning_rate, **kwargs):
    """
    Build and train a VGGNet model.
    :param resize: resize image to original ImageNet size.
    :param load_param: load parameters from pretrained model.
    :param batch_size: number of inputs in each batch.
    :param epochs: number of epochs to run before ending.
    :param learning_rate: initial learning rate.
    :param kwargs: extra parameters
    :return:
    """
    print("****** Compiling Data")
    img_size = VGG_INPUT_SIZE if resize else IMG_SIZE
    cache_file = DATA_CACHE_FILE_RS if resize else DATA_CACHE_FILE
    model = build_model(img_size)

    # use binary_crossentropy loss and adam optimizer, same as MURA baseline model
    adam = K.optimizers.Adam(
        lr=learning_rate, beta_1=0.9, beta_2=0.999,
        epsilon=None, decay=0.0, amsgrad=False
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=adam,
        metrics=[K.metrics.binary_accuracy])

    # Cache inputs and labels so that preprocess won't be called in every single training
    if HAS_CACHE:

        img_train, label_train, path_train, \
            img_valid, label_valid, path_valid = np.load(cache_file)

    else:
        util.create_dir(DATA_CACHE_PATH)

        print("****** Preparing Input")

        img_train, label_train, path_train,\
            img_valid, label_valid, path_valid = prepare_input()
        print("****** Resizing Image")

        if resize:
            img_train = dataset.resize_img(img_train, img_size)
            img_valid = dataset.resize_img(img_valid, img_size)

        np.save(
            cache_file,
            [
                img_train, label_train, path_train,
                img_valid, label_valid, path_valid
            ]
        )

    # log training time
    start_time = datetime.datetime.now()
    print("****** Starting Training: {:%H-%M-%S}".format(start_time))

    # Initiate TensorBoard callback
    util.create_dir(DATA_LOG_PATH)
    tfboard = K.callbacks.TensorBoard(log_dir=DATA_LOG_PATH, write_grads=True)

    model.fit(
        img_train, label_train, batch_size=batch_size, epochs=epochs, verbose=1,
        validation_data=(img_valid, label_valid),
        callbacks=[tfboard]
    )
    print('****** Training time: %s' % (datetime.datetime.now() - start_time))

    print("****** Saving Model")
    # save model after success training
    util.create_dir(MODEL_PATH)
    K.models.save_model(
        model,
        os.path.join(MODEL_PATH, "vgg_{:%Y-%m-%d-%H%M}.h5".format(
            datetime.datetime.now()
        ))
    )

    print("****** Writing Predictions")
    # run prediction on validation set and save result in csv
    write_prediction(model, img_valid, path_valid, batch_size, label_valid)


def write_prediction(model, imgs, paths, batch_size, labels=None):
    """
    Run prediction using given model on a list of images,
    and write the result to a csv file.
    :param model: model to run prediction on
    :param imgs: images to predict
    :param paths: list of image paths
    :param batch_size: number of inputs in each batch.
    :param labels: actual labels of the images
    :return:
    """
    predictions = model.predict(imgs, batch_size=batch_size)
    util.create_dir(RESULT_PATH)
    with open(
        os.path.join(RESULT_PATH, "vgg_{:%Y-%m-%d-%H%M}.csv".format(
            datetime.datetime.now()
        )),
        "w",
        newline=''
    ) as csvfile:
        fieldnames = ['path', 'prediction', "label"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(predictions)):
            writer.writerow(
                {
                    'path': paths[i],
                    'prediction': predictions[i],
                    "label": labels[i] if labels is not None else None
                }
            )


if __name__ == "__main__":
    HAS_CACHE = os.path.exists(DATA_CACHE_FILE)
    # Define argument parser so that the script can be executed directly
    # from console.
    ARG_PARSER = argparse.ArgumentParser("VGGNet model")
    SUBPARSER = ARG_PARSER.add_subparsers(help='sub-command help')
    # Arguments for training
    TRAIN_PARSER = SUBPARSER.add_parser("train")
    TRAIN_PARSER.set_defaults(func=train)
    TRAIN_PARSER.add_argument("--resize", action="store_true",
                              help="resize image to original ImageNet size")

    TRAIN_PARSER.add_argument("--load_param", action="store_true",
                              help="load parameters from pretrained model")

    TRAIN_PARSER.add_argument("-e", "--epochs", type=int, default=50,
                              help="number of epochs to run")

    TRAIN_PARSER.add_argument("-b", "--batch_size", type=int, default=8,
                              help="batch size")

    TRAIN_PARSER.add_argument("-lr", "--learning_rate", type=float, default=0.0001,
                              help="learning rate")

    # parse argument
    ARGS = ARG_PARSER.parse_args()
    ARGS.func(**vars(ARGS))
