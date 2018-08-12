"""
Training callback functions
"""
import keras
import numpy as np


class ReloadBest(keras.callbacks.Callback):
    """
    Callback function for reloading best weight at end of each epoch
    """
    def __init__(self, reload_path, patient=10, monitor="val_loss", mode="min"):
        super(ReloadBest, self).__init__()
        self.reload_path = reload_path
        self.patient = patient
        self.num_epoch_after_plateau = 0
        self.monitor = monitor
        self.mode = mode
        if self.mode not in ["max", "min"]:
            raise ValueError("Unexpected Value for mode: %s" % self.mode)
        self.best = np.Inf if self.mode == "min" else -np.Inf

    def on_epoch_begin(self, epoch, logs=None):
        self.num_epoch_after_plateau += 1

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if self.mode == "min":
            update = current < self.best
        else:
            update = current > self.best
        update = update or self.num_epoch_after_plateau >= self.patient

        if update:
            self.num_epoch_after_plateau = 0
        elif self.num_epoch_after_plateau >= self.patient:
            print("****** Reload weight from %s:" % self.reload_path)
            self.model.load_weights(self.reload_path)
