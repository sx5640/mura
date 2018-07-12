"""
Define Custom losses
"""
import keras.backend as K
import tensorflow as tf


class WeightedCrossEntropy:
    """
    Implement weighted cross entropy which was used in the baseline model
    """
    def __init__(self, df):
        num_pos = tf.constant(float(len(df[df["label"] == 1])), name="PositiveCount")
        num_neg = tf.constant(float(len(df[df["label"] == 0])), name="NegativeCount")
        self.pos_weight = num_neg / (num_neg + num_pos)
        self.neg_weight = num_pos / (num_pos + num_neg)
        self.__name__ = "weighted_cross_entropy_loss"

    def __call__(self, y_true, y_predict):
        return K.mean(
            -self.pos_weight * y_true * K.log(y_predict) -
            self.neg_weight * (1 - y_true) * K.log(1 - y_predict)
        )
