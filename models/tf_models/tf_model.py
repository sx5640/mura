"""
An abstract Model object that designed to work with MURA dataset with Tensorflow backend.
"""
import os

import pandas as pd

import dataset
from models.keras_models import util

ROOT_PATH = os.path.abspath(__file__)  # ?/abs_model.py
ROOT_PATH = os.path.dirname(ROOT_PATH)  # ?/


def load_resources(bpart, num_pick):
    """
    Utility method that load all the resources needed for training.
    Will use csv/pickle/recreated images as cache to avoid recomputation.
    Args:
        bpart: Body part to pick
        num_pick: Number of images to pick from each study in training set

    Returns: train_df, valid_df, img_valid, label_valid, path_valid, flow_dir

    """
    cache_path = os.path.join(ROOT_PATH, "cache")  # ?/cache

    train_table_path = os.path.join(
        cache_path,
        f"training_table_{bpart}_{num_pick}.csv"
    )

    valid_table_path = os.path.join(
        cache_path,
        "valid_table_{}.csv".format(bpart)
    )

    # Load datasets from csvs. If not exist, recreate from dataset.py and save to csvs
    try:
        train_df = pd.read_csv(train_table_path, index_col=0)
        valid_df = pd.read_csv(valid_table_path, index_col=0)

    except FileNotFoundError:
        util.create_dir(cache_path)

        train_df, valid_df = dataset.preprocess()

        if bpart != "all":
            train_df = dataset.pick_bpart(train_df, bpart)
            valid_df = dataset.pick_bpart(valid_df, bpart)

        if num_pick > 0:
            train_df = dataset.pick_n_per_patient(train_df, num_pick)

        train_df.to_csv(train_table_path)
        valid_df.to_csv(valid_table_path)

    return train_df, valid_df

