"""

"""
import argparse
import csv
import datetime
import os

import keras as K
import mnist
import numpy as np

import util
import vggnet


DATA_DIR = os.path.join(vggnet.ROOT_PATH, "dataset", "mnist")
MODEL_PATH = os.path.join(vggnet.ROOT_PATH, "models", "mnist")           # ?/models/mnist
RESULT_PATH = os.path.join(vggnet.ROOT_PATH, "results", "mnist")         # ?/results/mnist

IMG_SIZE = 28
NUM_CLASS = 10


def one_hot_encode(labels):
    """
    One Hot Encode a list of labels.
    :param labels: ndarray of unencoded labels
    :return: ndarray of encoded labels
    """
    result = np.zeros((len(labels), NUM_CLASS))
    result[range(len(labels)), labels] = 1
    return result


def build_model(img_size):
    """
    Laying out the VGGNet model.
    :param img_size: size of the input image.
    :return: keras model
    """
    img_tensor = K.layers.Input((img_size, img_size, 1))

    tensor = vggnet.conv_block(img_tensor, 64, 2, 3, "relu")
    # tensor = vggnet.conv_block(tensor, 128, 2, 3, "relu")
    # tensor = vggnet.conv_block(tensor, 256, 3, 3, "relu")
    # tensor = vggnet.conv_block(tensor, 512, 3, 3, "relu")
    # tensor = vggnet.conv_block(tensor, 512, 3, 3, "relu")
    tensor = K.layers.Flatten()(tensor)
    tensor = K.layers.Dense(256, activation="relu")(tensor)
    tensor = K.layers.Dropout(0.5)(tensor)
    tensor = K.layers.Dense(NUM_CLASS, activation="softmax")(tensor)

    model = K.models.Model(inputs=img_tensor, outputs=tensor, name="VGG")

    return model


def train(batch_size, epochs, learning_rate, **kwargs):
    """
    Build and train a VGGNet model.
    :param batch_size: number of inputs in each batch.
    :param epochs: number of epochs to run before ending.
    :param learning_rate: initial learning rate.
    :param kwargs: extra parameters
    :return:
    """
    model = build_model(IMG_SIZE)

    # use categorical_crossentropy loss and adam optimizer, same as MURA baseline model
    adam = K.optimizers.Adam(
        lr=learning_rate, beta_1=0.9, beta_2=0.999,
        epsilon=None, decay=0.0, amsgrad=False
    )

    model.compile(
        loss='categorical_crossentropy',
        optimizer=adam,
        metrics=["acc"])

    # log training time
    start_time = datetime.datetime.now()
    model.fit(
        IMGS_TRAIN, LABELS_TRAIN, batch_size=batch_size, epochs=epochs, verbose=1,
        validation_data=(IMGS_VALID, LABELS_VALID))
    print('Training time: %s' % (datetime.datetime.now() - start_time))

    # save model after success training
    util.create_dir(MODEL_PATH)

    K.models.save_model(
        model,
        os.path.join(MODEL_PATH, "vgg_{:%Y-%m-%d-%H%M}.h5".format(
            datetime.datetime.now()
        ))
    )

    # run prediction on validation set and save result in csv
    write_prediction(model, IMGS_VALID, batch_size, LABELS_VALID)


def write_prediction(model, imgs, batch_size, labels=None):
    """
    Run prediction using given model on a list of images,
    and write the result to a csv file.
    :param model: model to run prediction on
    :param imgs: images to predict
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
        fieldnames = ['prediction', "label"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(predictions)):
            writer.writerow(
                {
                    'prediction': predictions[i],
                    "label": labels[i] if labels is not None else None
                }
            )


if __name__ == "__main__":
    # Load dataset
    MNDATA = mnist.MNIST(DATA_DIR, return_type="numpy")

    IMGS_TRAIN, LABELS_TRAIN = MNDATA.load_training()
    IMGS_VALID, LABELS_VALID = MNDATA.load_testing()

    # Data preprocessing
    IMGS_TRAIN = IMGS_TRAIN.reshape((IMGS_TRAIN.shape[0], IMG_SIZE, IMG_SIZE, 1))
    LABELS_TRAIN = one_hot_encode(LABELS_TRAIN)
    IMGS_VALID = IMGS_VALID.reshape((IMGS_VALID.shape[0], IMG_SIZE, IMG_SIZE, 1))
    LABELS_VALID = one_hot_encode(LABELS_VALID)

    # Define argument parser so that the script can be executed directly
    # from console.
    ARG_PARSER = argparse.ArgumentParser("VGGNet model")
    SUBPARSER = ARG_PARSER.add_subparsers(help='sub-command help')
    # Arguments for training
    TRAIN_PARSER = SUBPARSER.add_parser("train")
    TRAIN_PARSER.set_defaults(func=train)

    TRAIN_PARSER.add_argument("-e", "--epochs", type=int, default=50,
                              help="number of epochs to run")

    TRAIN_PARSER.add_argument("-b", "--batch_size", type=int, default=8,
                              help="batch size")

    TRAIN_PARSER.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                              help="learning rate")

    # parse argument
    ARGS = ARG_PARSER.parse_args()
    ARGS.func(**vars(ARGS))
