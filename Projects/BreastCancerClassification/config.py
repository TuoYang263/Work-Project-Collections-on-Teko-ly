# -*- coding: utf-8 -*-
"""
This file holds configuration need for building the dataset and training the model

@author: Tuo Yang
"""
import os

class Config():
    # the path to the input dataset
    INPUT_DATASET = "datasets/original"
    # the path for the new directory 
    BASE_PATH = "datasets/idc"
    # the paths for the training,validation,and testing directories using the
    # bat
    TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
    VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
    TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])
    
    # declare 80% of the entire dataset is used for training, of that, 10% will be used for
    # validation
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1

