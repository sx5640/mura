"""
General purposed utility methods shared by all models.

To use, simply import the file and start making calls.
"""
import os

import numpy as np
import keras


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


#################################
#          Metrics              #
#################################
def batch_recall(y_true, y_pred):
    """Recall metric. THIS IS A REUSE OF REMOVED KERAS CODE:
    https://github.com/keras-team/keras/commit/a56b1a55182acf061b1eb2e2c86b48193a0e88f7

    Only computes a batchwise average of recall.

    Computes the recall, a metric for multilabel classification of
    how many relevant items are selected.
    """
    true_positives = keras.backend.sum(
        keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1))
    )
    possible_positives = keras.backend.sum(
        keras.backend.round(keras.backend.clip(y_true, 0, 1))
    )
    return true_positives / (possible_positives + keras.backend.epsilon())


class BinaryRecall(keras.layers.Layer):
    """Stateful Metric to calculate the recall over all batches.
    Assumes predictions and targets of shape `(samples, 1)`.

    Reference:
        https://github.com/keras-team/keras/blob/master/tests/keras/metrics_test.py

    # Arguments
        name: String, name for the metric.
    """

    def __init__(self, name='global_recall', **kwargs):
        super(BinaryRecall, self).__init__(name=name, **kwargs)
        self.stateful = True
        self.true_positives = keras.backend.variable(value=0, dtype='int32')
        self.possible_positives = keras.backend.variable(value=0, dtype='int32')

    def reset_states(self):
        keras.backend.set_value(self.true_positives, 0)
        keras.backend.set_value(self.possible_positives, 0)

    def __call__(self, y_true, y_pred):
        """Computes the recall in a batch.
        # Arguments
            y_true: Tensor, batch_wise labels
            y_pred: Tensor, batch_wise predictions
        # Returns
            The overall recall seen this epoch at the completion of the batch.
        """
        y_true = keras.backend.cast(y_true, 'int32')
        y_pred = keras.backend.cast(keras.backend.round(y_pred), 'int32')
        correct_preds = keras.backend.cast(keras.backend.equal(y_pred, y_true), 'int32')
        true_pos = keras.backend.cast(keras.backend.sum(correct_preds * y_true), 'int32')
        poss_pos = keras.backend.cast(keras.backend.sum(y_true), 'int32')
        current_true_pos = self.true_positives * 1
        current_poss_pos = self.possible_positives * 1
        self.add_update(
            keras.backend.update_add(self.true_positives, true_pos),
            inputs=[y_true, y_pred]
        )
        self.add_update(
            keras.backend.update_add(self.possible_positives, poss_pos),
            inputs=[y_true, y_pred]
        )
        return (current_true_pos + true_pos) / (current_poss_pos + poss_pos)


class BinaryKappa(keras.layers.Layer):
    """Stateful Metric to calculate kappa over all batches.
    Assumes predictions and targets of shape `(samples, 1)`.

    Reference:
        https://github.com/keras-team/keras/blob/master/tests/keras/metrics_test.py

    # Arguments
        name: String, name for the metric.
    """

    def __init__(self, name='global_kappa', **kwargs):
        super(BinaryKappa, self).__init__(name=name, **kwargs)
        self.stateful = True
        self.true_positives = keras.backend.variable(value=0, dtype='int32')
        self.true_negative = keras.backend.variable(value=0, dtype='int32')
        self.false_positives = keras.backend.variable(value=0, dtype='int32')
        self.false_negative = keras.backend.variable(value=0, dtype='int32')

    def reset_states(self):
        keras.backend.set_value(self.true_positives, 0)
        keras.backend.set_value(self.true_negative, 0)
        keras.backend.set_value(self.false_positives, 0)
        keras.backend.set_value(self.false_negative, 0)

    def __call__(self, y_true, y_pred):
        """Computes the kappa in a batch.
        # Arguments
            y_true: Tensor, batch_wise labels
            y_pred: Tensor, batch_wise predictions
        # Returns
            The kappa seen this epoch at the completion of the batch.
        """
        y_true = keras.backend.cast(y_true, 'int32')
        y_pred = keras.backend.cast(keras.backend.round(y_pred), 'int32')
        correct_preds = keras.backend.cast(keras.backend.equal(y_pred, y_true), 'int32')
        true_pos = keras.backend.cast(
            keras.backend.sum(correct_preds * y_true),
            'int32'
        )
        true_neg = keras.backend.cast(
            keras.backend.sum(correct_preds * (1 - y_true)),
            'int32'
        )
        false_pos = keras.backend.cast(
            keras.backend.sum((1 - correct_preds) * (1 - y_true)),
            'int32'
        )
        false_neg = keras.backend.cast(
            keras.backend.sum((1 - correct_preds) * y_true),
            'int32'
        )

        self.add_update(
            keras.backend.update_add(self.true_positives, true_pos),
            inputs=[y_true, y_pred]
        )
        self.add_update(
            keras.backend.update_add(self.true_negative, true_neg),
            inputs=[y_true, y_pred]
        )
        self.add_update(
            keras.backend.update_add(self.false_positives, false_pos),
            inputs=[y_true, y_pred]
        )
        self.add_update(
            keras.backend.update_add(self.false_negative, false_neg),
            inputs=[y_true, y_pred]
        )
        true_pos += self.true_positives * 1
        true_neg += self.true_negative * 1
        false_pos += self.false_positives * 1
        false_neg += self.false_negative * 1
        sm = true_pos + true_neg + false_pos + false_neg
        obs_agree = (true_pos + true_neg) / sm
        poss_pos = (true_pos + false_neg) * (true_pos + false_pos) / sm**2
        poss_neg = (true_neg + false_neg) * (true_neg + false_pos) / sm**2
        poss_agree = poss_pos + poss_neg
        return (obs_agree - poss_agree) / (1 - obs_agree)
