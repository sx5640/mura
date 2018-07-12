"""
Metrics for model and result evaluation
"""
import keras
import keras.backend as K
import numpy as np


#################################
#        Model Metrics          #
#################################

def batch_recall(y_true, y_pred):
    """Recall metric. THIS IS A REUSE OF REMOVED KERAS CODE:
    https://github.com/keras-team/keras/commit/a56b1a55182acf061b1eb2e2c86b48193a0e88f7

    Only computes a batchwise average of recall.

    Computes the recall, a metric for multilabel classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(
        K.round(K.clip(y_true * y_pred, 0, 1))
    )
    possible_positives = K.sum(
        K.round(K.clip(y_true, 0, 1))
    )
    return true_positives / (possible_positives + K.epsilon())


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
        self.true_positives = K.variable(value=0, dtype='int32')
        self.possible_positives = K.variable(value=0, dtype='int32')

    def reset_states(self):
        K.set_value(self.true_positives, 0)
        K.set_value(self.possible_positives, 0)

    def __call__(self, y_true, y_pred):
        """Computes the recall in a batch.
        # Arguments
            y_true: Tensor, batch_wise labels
            y_pred: Tensor, batch_wise predictions
        # Returns
            The overall recall seen this epoch at the completion of the batch.
        """
        y_true = K.cast(y_true, 'int32')
        y_pred = K.cast(K.round(y_pred), 'int32')
        true_pos = K.cast(K.sum(y_pred * y_true), 'int32')
        poss_pos = K.cast(K.sum(y_true), 'int32')

        self.add_update(
            K.update_add(self.true_positives, true_pos),
            inputs=[y_true, y_pred]
        )
        self.add_update(
            K.update_add(self.possible_positives, poss_pos),
            inputs=[y_true, y_pred]
        )
        true_pos = K.cast(self.true_positives * 1, "float32")
        poss_pos = K.cast(self.possible_positives * 1, "float32")
        return true_pos / (K.epsilon() + poss_pos)


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
        self.true_positives = K.variable(value=0, dtype='int32')
        self.true_negative = K.variable(value=0, dtype='int32')
        self.false_positives = K.variable(value=0, dtype='int32')
        self.false_negative = K.variable(value=0, dtype='int32')

    def reset_states(self):
        K.set_value(self.true_positives, 0)
        K.set_value(self.true_negative, 0)
        K.set_value(self.false_positives, 0)
        K.set_value(self.false_negative, 0)

    def __call__(self, y_true, y_pred):
        """Computes the kappa in a batch.
        # Arguments
            y_true: Tensor, batch_wise labels
            y_pred: Tensor, batch_wise predictions
        # Returns
            The kappa seen this epoch at the completion of the batch.
        """
        y_true = K.cast(y_true, 'int32')
        y_pred = K.cast(K.round(y_pred), 'int32')
        true_pos = K.cast(
            K.sum(y_pred * y_true),
            'int32'
        )
        true_neg = K.cast(
            K.sum((1 - y_pred) * (1 - y_true)),
            'int32'
        )
        false_pos = K.cast(
            K.sum(y_pred * (1 - y_true)),
            'int32'
        )
        false_neg = K.cast(
            K.sum((1 - y_pred) * y_true),
            'int32'
        )

        self.add_update(
            K.update_add(self.true_positives, true_pos),
            inputs=[y_true, y_pred]
        )
        self.add_update(
            K.update_add(self.true_negative, true_neg),
            inputs=[y_true, y_pred]
        )
        self.add_update(
            K.update_add(self.false_positives, false_pos),
            inputs=[y_true, y_pred]
        )
        self.add_update(
            K.update_add(self.false_negative, false_neg),
            inputs=[y_true, y_pred]
        )

        true_pos = K.cast(self.true_positives * 1, "float32")
        true_neg = K.cast(self.true_negative * 1, "float32")
        false_pos = K.cast(self.false_positives * 1, "float32")
        false_neg = K.cast(self.false_negative * 1, "float32")

        sm = true_pos + true_neg + false_pos + false_neg
        obs_agree = (true_pos + true_neg) / sm
        poss_pos = (true_pos + false_neg) * (true_pos + false_pos) / (sm**2)
        poss_neg = (true_neg + false_neg) * (true_neg + false_pos) / (sm**2)
        poss_agree = poss_pos + poss_neg
        return (obs_agree - poss_agree) / (1 - poss_agree + K.epsilon())


#################################
#       Dataframe Metrics       #
#################################

def basic_metrics(predict, label):
    """
    Methods that returns:
        true positive
        true negative
        false positive
        false negative

    Args:
        predict: prediction
        label: labels

    Returns:
        true_pos, true_neg, false_pos, false_neg, sum
    """
    true_pos = int(sum(np.round(predict) * label))
    true_neg = int(sum(-1 * np.round(predict - 1) * -1 * (label - 1)))
    false_pos = int(sum(np.round(predict) * -1 * (label - 1)))
    false_neg = int(sum((-1 * np.round(predict - 1) * label)))
    sm = len(predict)
    return true_pos, true_neg, false_pos, false_neg, sm


def kappa(predict, label):
    """
    Methods for calculating Cohen's kappa.

    https://en.wikipedia.org/wiki/Cohen%27s_kappa
    Args:
        predict: prediction
        label: labels

    Returns:
        kappa
    """
    true_pos, true_neg, false_pos, false_neg, sm = basic_metrics(predict, label)

    obs_agree = (true_pos + true_neg) / sm
    poss_pos = (true_pos + false_neg) * (true_pos + false_pos) / (sm ** 2)
    poss_neg = (true_neg + false_neg) * (true_neg + false_pos) / (sm ** 2)
    poss_agree = poss_pos + poss_neg
    return (obs_agree - poss_agree) / (1 - poss_agree + np.finfo(np.float).eps)


def recall(predict, label):
    """
    Methods for calculating recall.

    https://en.wikipedia.org/wiki/Precision_and_recall
    Args:
        predict: prediction
        label: labels

    Returns:
        recall
    """
    true_pos, true_neg, false_pos, false_neg, sm = basic_metrics(predict, label)
    return true_pos / (true_pos + false_neg + np.finfo(np.float).eps)


def precision(predict, label):
    """
    Methods for calculating precision.

    https://en.wikipedia.org/wiki/Precision_and_recall
    Args:
        predict: prediction
        label: labels

    Returns:
        precision
    """
    true_pos, true_neg, false_pos, false_neg, sm = basic_metrics(predict, label)
    return true_pos / (true_pos + false_pos + np.finfo(np.float).eps)


def accuracy(predict, label):
    """
    Methods for calculating accuracy.

    Args:
        predict: prediction
        label: labels

    Returns:
        accuracy
    """
    true_pos, true_neg, false_pos, false_neg, sm = basic_metrics(predict, label)
    return (true_pos + true_neg) / (sm + np.finfo(np.float).eps)
