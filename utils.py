#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some utils for training is implemented here.
Thanks to Harvard Annoated Transformer in http://nlp.seas.harvard.edu/2018/04/03/attention.html

@author: Ma (Ma787639046@outlook.com)
@data: 2020/12/24

"""
import os
import logging

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to {log_path}.

    !!! Note that old file of {log_path} will be overwriten !!!

    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) file path to where to log
    """
    # if os.path.exists(log_path) is True:
    #     os.remove(log_path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s : %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s : %(message)s'))
        logger.addHandler(stream_handler)
