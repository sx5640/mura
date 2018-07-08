import numpy as np

import metric


def print_evl(predict, label):
    """
    Utility method that prints the evaluation results:
        accuracy
        recall
        precision
        kappa
        Contingency table
    Args:
        predict: prediction
        label: labels

    Returns:

    """
    print("Accuracy:  {}".format(metric.accuracy(predict, label)))
    print("Recall:    {}".format(metric.recall(predict, label)))
    print("Precision: {}".format(metric.precision(predict, label)))
    print("Kappa:     {}".format(metric.kappa(predict, label)))
    print(
        "Contingency Table:\n"
        "{0:5d}          {2:5d}\n"
        "{3:5d}          {1:5d}\n"
        "Number of Inputs: {4}"
        .format(*metric.basic_metrics(predict, label))
    )


def evl_result(val_df):
    """
    Utility method that analyse the result of from a model.
    Will print:
        accuracy, recall, precision, kappa, Contingency table
        evaluated:
            per image
            per study using highest score
            per study using lowest score
            per study using average score

    Args:
        val_df: result dataframe

    Returns:
        result loaded into dataframe
    """
    print("****** Evaluation per Image")
    print_evl(
        np.asarray(val_df["prediction"].tolist()),
        np.asarray(val_df["label"].tolist())
    )

    print("****** Evaluation per Study Using Highest Score")
    print_evl(
        np.asarray(val_df.groupby("study")["prediction"].max().tolist()),
        np.asarray(val_df.groupby("study")["label"].mean().tolist())
    )

    print("****** Evaluation per Study Using Lowest Score")
    print_evl(
        np.asarray(val_df.groupby("study")["prediction"].min().tolist()),
        np.asarray(val_df.groupby("study")["label"].mean().tolist())
    )

    print("****** Evaluation per Study Using Average Score")
    print_evl(
        np.asarray(val_df.groupby("study")["prediction"].mean().tolist()),
        np.asarray(val_df.groupby("study")["label"].mean().tolist())
    )
