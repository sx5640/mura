"""
General purposed utility methods shared by all models.

To use, simply import the file and start making calls.
"""
import os
import tempfile

import keras
import numpy as np


def get_model_memory_usage(batch_size, model):
    """
    Estimate how much memory the model will take, assuming all parameters is in float32
    and float32 takes 4 bytes (32 bits) in memory.
    :param batch_size:
    :param model:
    :return:
    """
    # Calculate the total number of outputs from all layers
    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem
    # Calculate the total number of trainable parameters
    trainable_count = np.sum(
        [keras.backend.count_params(p) for p in set(model.trainable_weights)]
    )
    # Calculate the total number of non trainable parameters
    non_trainable_count = np.sum(
        [keras.backend.count_params(p) for p in set(model.non_trainable_weights)]
    )
    # total memory = 4 bytes * total number of numbers in each run * number of images in each run
    total_memory = 4.0 * batch_size * (shapes_mem_count + trainable_count + non_trainable_count)
    # convert to GB
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes


def print_weights(file):
    """
    Load a model file and print is weights
    :param file: path to model
    :return:
    """
    model = keras.models.load_model(file)
    for layer in model.layers:
        print("Layer: {}, weights: \n{}".format(
            layer.name,
            layer.get_weights()
        ))


def create_dir(path):
    """
    Recursively create the directory and all its parent directories.
    :param path: directory path
    :return:
    """
    if not (os.path.exists(path)):
        # create the directory you want to save to
        create_dir(os.path.dirname(path))
        os.mkdir(path)


def reload_model(model):
    """
    Reload a given model by saving it to a temporary file and reload.
    :param model: Model to reload.
    :return: reloaded model.
    """
    model_path = os.path.join(tempfile.gettempdir(), "temp.h5")
    try:
        model.save(model_path)
        model = keras.models.load_model(model_path)
    finally:
        os.remove(model_path)

    return model
