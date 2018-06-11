"""
Import data from dataset, and preprocess it.
"""
import imghdr
import os
import re

import numpy as np
import pandas as pd
import imageio


IMG_SIZE = 512

DATA_VIR = "1.1"

DATA_DIR = os.path.abspath(__file__)        # ?/dataset.py
DATA_DIR = os.path.dirname(DATA_DIR)        # ?/
DATA_DIR = os.path.join(
    DATA_DIR,
    "dataset",
    "MURA-v" + DATA_VIR,
)                                           # ?/dataset/MURA-v1.1/

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "valid")

# import labeled dataset into Dataframe
TRAIN_LABELED = pd.read_csv(
    os.path.join(DATA_DIR, "train_labeled_studies.csv"),
    names=["patient", "label"]
)

VALID_LABELED = pd.read_csv(
    os.path.join(DATA_DIR, "valid_labeled_studies.csv"),
    names=["patient", "label"]
)

# import image paths
TRAIN_PATH = pd.read_csv(
    os.path.join(DATA_DIR, "train_image_paths.csv"),
    names=["path"]
)

VALID_PATH = pd.read_csv(
    os.path.join(DATA_DIR, "valid_image_paths.csv"),
    names=["path"]
)

BPARTS = ["elbow", "finger", "forearm", "hand", "humerus", "shoulder", "wrist"]


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
    df_train = build_dataframe(TRAIN_LABELED, TRAIN_PATH)
    df_valid = build_dataframe(VALID_LABELED, VALID_PATH)

    return df_train, df_valid


def load_images(df):
    """
    Load all images in the dataframe in to a nparray. Will add padding to make
    a square image. Image will be grayscaled if it not already.
    :param df: dataframe to load from.
    :return: nparray of shape (num_images, img_size, img_size), corresponding labels,
    image path
    """
    num_images = df.shape[0]
    imgs = np.zeros((num_images, IMG_SIZE, IMG_SIZE))
    labels = np.zeros(num_images)
    path = [""] * num_images
    for idx, row in df.iterrows():
        # load image from path into nparray
        img = imageio.imread(row["path"])

        # gray scale image if it is not already gray scaled
        if len(img.shape) == 3:
            img = np.tensordot(img, [0.2, 0.5, 0.3], axes=(-1, -1))

        # add padding to the image matrix
        # for each side of the image, each colour channel shall be padded with 0s of size
        # (512 - image_width/height)/2 on each end, so that the image stays in the center,
        # and is surrounded with black.
        horz_start = int((IMG_SIZE - img.shape[0]) / 2)
        horz_cord = range(horz_start, horz_start + img.shape[0])

        vert_start = int((IMG_SIZE - img.shape[1]) / 2)
        vert_cord = range(vert_start, vert_start + img.shape[1])

        imgs[np.ix_([idx], horz_cord, vert_cord)] = img
        labels[idx] = row["label"]
        path[idx] = row["path"]

    return imgs, labels, path


def pick_bpart(df, bpart):
    """
    Create a sub dataset of particular body part.
    :param df: dataframe to process
    :param bpart: body part to extract
    :return: trimmed dataframe
    """
    return df[df["body_part"] == bpart].reset_index()


def pick_n_per_patient(df, num):
    """
    Create a sub dataset that pick first n images from each patient. Will return error
    if num is greater than the minial count
    :param df: dataframe to process
    :param num: number of images to pick from each patient
    :return: trimmed dataframe
    """
    min_count = df.groupby("study")["path"].count().min()

    if num > min_count:
        raise ValueError("num is greater than minimum count of images per patient: {}".format(
            min_count
        ))

    result = pd.DataFrame()
    for study in df["study"].unique():
        result = result.append(df[df["study"] == study][:num])

    return result.reset_index()
