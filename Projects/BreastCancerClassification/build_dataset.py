# -*- coding: utf-8 -*-
"""
Split the dataset into training, validation, and testing sets in
the ratio - 80% of the entire dataset will be used for training, 
             and of that, 10% will be used for validation

@author: Tuo Yang
"""
# Module not found error, solution: 
# import sys
# sys.path.append()
# watch the path used for the interaction with the Python interpreter
from config import Config
from imutils import paths
import random, shutil, os, shutil

# put all images' paths into a list 
original_paths = list(paths.list_images(Config.INPUT_DATASET))
# set up the random seed
random.seed(7)
# shuffle the order of all image paths
random.shuffle(original_paths)

# settle down the index used for splitting 80% of the dataset
split_index = int(len(original_paths)*Config.TRAIN_SPLIT)
# save the first 80% images' paths to the variable train_paths
train_paths = original_paths[:split_index]
# save the another 20% images' paths to the variable test_paths
test_paths = original_paths[split_index:]

# of the training dataset, keep the first 10% as the validation set
split_index = int(len(train_paths)*Config.VAL_SPLIT)
val_paths = train_paths[:split_index]
train_paths = train_paths[split_index:]

# create a list for traversing in the loop
datasets = [("training", train_paths, Config.TRAIN_PATH),
            ("validation", val_paths, Config.VAL_PATH),
            ("testing", test_paths, Config.TEST_PATH)]

# traverse the list dataset in the loop to build various sets
for (set_type, original_path, base_path) in datasets:
    # f means str.format: format string, {} includes the format
    print(f'Building {set_type} set')
    
    # if the set's path does not exist, creating the path
    if not os.path.exists(base_path):
        print(f'Building directory {base_path}')
        os.makedirs(base_path)
        
    for path in original_path:
        # acquire the file name of each image from the last element of
        # the string path, the function split use '\' as the seperator
        file_name = path.split(os.path.sep)[-1]
        # acquire the label of each image, it could be two digits: 0 and 1
        # 0 represents benign, 1 represents malignant
        label = file_name[-5:-4]
        
        # create the label's path, if it does not exist, 
        # its directories will be created
        label_path = os.path.sep.join([base_path, label])
        if not os.path.exists(label_path):
            print(f'Building directory {label_path}')
            os.makedirs(label_path)
        
        # move all image files to the new created paths
        new_path = os.path.sep.join([label_path, file_name])
        # shutils.copy2(), move source files to their target directories
        shutil.copy2(path, new_path)

"""
program outputs in the console:
Building training set
Building directory datasets/idc\training
Building directory datasets/idc\training\0
Building directory datasets/idc\training\1
Building validation set
Building directory datasets/idc\validation
Building directory datasets/idc\validation\1
Building directory datasets/idc\validation\0
Building testing set
Building directory datasets/idc\testing
Building directory datasets/idc\testing\1
Building directory datasets/idc\testing\0
"""
        
    






