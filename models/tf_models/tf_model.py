"""
An abstract Model object that designed to work with MURA dataset with Tensorflow backend.
"""
import os
import sys

import pandas as pd

ROOT_PATH = os.path.abspath(__file__)  # ?/models/tf_models/tf_model.py
ROOT_PATH = os.path.abspath(os.path.join(ROOT_PATH, os.pardir, os.pardir, os.pardir))  # ?/
sys.path.append(ROOT_PATH)

import dataset
import models.abs_model
import util


class TFModel(models.abs_model):

