"""
General purposed utility methods shared by all models.

To use, simply import the file and start making calls.
"""

import os


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