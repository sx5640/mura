"""
Import data from dataset, and preprocess it.

To use take advantage of the preprocess in this file, simply import this file to
your python code and call `train_table, valid_table = dataset.preprocess()` to get
all images in one table with their labels and metadata.

Once you have the training and validation table, you pass them into the utility
methods to select the dataset you need, and use `dataset.load_images(result_table)`
to output the actual images and labels as ndarray. You can also use
`dataset.resize_img(imgs, img_size)` to resize your images to desired image size.
"""
import imghdr
import math
import os
import re

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import util


IMG_SIZE = 512

DATA_VIR = "1.1"

DATA_DIR = os.path.abspath(__file__)                            # ?/dataset.py
ROOT_DIR = os.path.dirname(DATA_DIR)                            # ?/
DATA_DIR = os.path.join(
    ROOT_DIR,
    "dataset",
    "MURA-v" + DATA_VIR,
)                                                               # ?/dataset/MURA-v1.1/

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "valid")

BPARTS = ["elbow", "finger", "forearm", "hand", "humerus", "shoulder", "wrist"]


def load_dataframe():
    """
     Import csv files into Dataframes.
    :return:
    """
    train_labeled = pd.read_csv(
        os.path.join(DATA_DIR, "train_labeled_studies.csv"),
        names=["patient", "label"]
    )

    valid_labeled = pd.read_csv(
        os.path.join(DATA_DIR, "valid_labeled_studies.csv"),
        names=["patient", "label"]
    )

    # import image paths
    train_path = pd.read_csv(
        os.path.join(DATA_DIR, "train_image_paths.csv"),
        names=["path"]
    )

    valid_path = pd.read_csv(
        os.path.join(DATA_DIR, "valid_image_paths.csv"),
        names=["path"]
    )

    return train_labeled, valid_labeled, train_path, valid_path


def classify_bpart(data):
    """
    Divide TRAIN_LABELED into sub-sets based on the body parts in the image.
    Also add body part as a new feature of the dataset.
    :param data: dataset to process.
    :return:
    """
    for bpart in BPARTS:
        data.loc[data["path"].str.contains(bpart.upper()), "body_part"] = bpart


def complete_path(data, column):
    """
    Convert relative image path to absolute path so that the execution does not depend
    on working directory. Also clean up the patient name
    :param data: dataset to process.
    :param column: column to perform the operation.
    :return:
    """
    data[column] = np.where(
        data[column].str.startswith("MURA-v" + DATA_VIR),
        data[column].str.replace("MURA-v" + DATA_VIR, DATA_DIR),
        data[column]
    )


def extract_study(row):
    """
    Callback function to generate a column for unique patient-study combo.
    :param row: a row from processing table
    :return:
    """
    match = re.search("study\d+", row["path"])
    if match:
        study = match.group()
        return "{}-{}-{}".format(row["patient"], row["body_part"], study)
    else:
        raise ValueError("study not found in " + row["path"])


def get_patient(row):
    """
    Call back function to check if the image column is a valid path,
    and grab the parent directory if it is.
    :param row: a row from processing table
    :return:
    """
    try:
        img_type = imghdr.what(row["path"])
    except IsADirectoryError:
        img_type = None

    if img_type:
        return os.path.dirname(row["path"]) + "/"
    return row["patient"]


def build_dataframe(df_label, df_path):
    """
    Build datasets by combining image paths with labels, so that we have a dataframe
    where each row is an image and has the patient it belongs to, as well as the label
    :param df_label: labeled dataset.
    :param df_path: image paths.
    :return: training table, validation table
    """
    df_label = df_label.copy(deep=True)
    df_path = df_path.copy(deep=True)

    complete_path(df_path, "path")
    complete_path(df_label, "patient")

    # Apply a transformation over each row to save image directory as a new column
    df_path["patient"] = df_path.apply(get_patient, axis=1)

    # Merge two table on patient column
    result = df_path.merge(df_label, on="patient")

    classify_bpart(result)

    # change .../patient00001/... to patient00001
    result["patient"] = result["patient"].str.extract("(patient\d{5})")

    # Apply a transformation over each row to create a column for unique
    # patient-bpart-study combo
    result["study"] = result.apply(extract_study, axis=1)
    return result


def preprocess():
    """
    Preprocess datasets.
    :return: training set, validation set
    """
    train_labeled, valid_labeled, train_path, valid_path = load_dataframe()
    df_train = build_dataframe(train_labeled, train_path)
    df_valid = build_dataframe(valid_labeled, valid_path)

    return df_train, df_valid


#################################
#       Utility Fnctions        #
#################################

def pick_bpart(df, bpart):
    """
    Create a sub dataset of particular body part.
    :param df: dataframe to process
    :param bpart: body part to extract
    :return: trimmed dataframe
    """
    if bpart == "all":
        return df
    return df[df["body_part"] == bpart].reset_index()


def pick_n_per_patient(df, num):
    """
    Create a sub dataset that pick first n images from each patient. Will return error
    if num is greater than the minial count
    :param df: dataframe to process
    :param num: number of images to pick from each patient. if set to 0, then pick all.
    :return: trimmed dataframe
    """
    if num == 0:
        return df
    min_count = df.groupby("study")["path"].count().min()

    if num > min_count:
        raise ValueError("num is greater than minimum count of images per patient: {}".format(
            min_count
        ))

    result = pd.DataFrame()
    for study in df["study"].unique():
        result = result.append(df[df["study"] == study][:num])

    return result.reset_index()


def zero_pad(img):
    """
    Add black padding to the image.

    for each side of the image, each colour channel shall be padded with 0s of size
    (512 - image_width/height)/2 on each end, so that the image stays in the center,
    and is surrounded with black.

    :param img: Image to process in nparray.
    :return: Processed image.
    """
    result = np.zeros((IMG_SIZE, IMG_SIZE, img.shape[2]))
    horz_start = int((IMG_SIZE - img.shape[0]) / 2)
    horz_cord = range(horz_start, horz_start + img.shape[0])

    vert_start = int((IMG_SIZE - img.shape[1]) / 2)
    vert_cord = range(vert_start, vert_start + img.shape[1])

    result[np.ix_(horz_cord, vert_cord, range(img.shape[2]))] = img.reshape(
            (img.shape[0], img.shape[1], img.shape[2])
        )
    return result


def load_image(img_path, is_grayscale=False):
    """
    Load a single image into a ndarray.
    Args:
        img_path:
            path to the image

        is_grayscale:
            if load the image to grayscale or RGB

    Returns: image as ndarray

    """
    im = keras.preprocessing.image.load_img(img_path, grayscale=is_grayscale)
    im = keras.preprocessing.image.img_to_array(im)     # converts image to numpy array
    return zero_pad(im)


def plot_first_n_img(imgs, num=9):
    """
    Plot first n images from the given list.
    :param imgs: ndarry of images
    :param num: number of images to show
    :return:
    """
    n_row = int(math.sqrt(num))
    n_col = math.ceil(math.sqrt(num))
    plt.figure(1)
    plt.tight_layout()
    for i in range(num):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(imgs[i, :, :, 0], cmap='gray')


def save_img(img, filename):
    """
    Utility method that convert a ndarray into image and save to a image file.
    Args:
        img: image in ndarray
        filename: target filename including path

    Returns:

    """
    img = keras.preprocessing.image.array_to_img(img)
    try:
        img.save(filename)
    except FileNotFoundError:
        util.create_dir(os.path.dirname(filename))
        img.save(filename)


def resize_img(img, size):
    """
    Given a list of images in ndarray, resize them into target size.
    Args:
        img: Input image in ndarray
        size: Target image size

    Returns: Resized images in ndarray

    """

    img = cv2.resize(img, (size, size))
    if len(img.shape) == 2:
        img = img.reshape((size, size, 1))
    return img

