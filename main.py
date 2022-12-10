from dotenv import load_dotenv

load_dotenv()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Flatten
from keras.layers import Dropout
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import random
import requests
from PIL import Image
import cv2
import pickle
import pandas as pd
import ntpath
import os

DATA_DIR = os.getenv('DATA_DIR')
FOLDERS_TO_TRAIN = "wnids.txt"
VALIDATION_TEXT = os.getenv('VALIDATION_TEXT')


def preprocess_img(img):
    path = path_leaf(img)
    img = mpimg.imread(img)
    if path in val_crop:
        x_min, y_min, x_max, y_max = val_crop[path][0], val_crop[path][1], val_crop[path][2], val_crop[path][3]
        img = img[y_min:y_max, x_min:x_max]
    else:
        train_crop[path] = list(map(int, train_crop[path]))
        x_min, y_min, x_max, y_max = train_crop[path][0], train_crop[path][1], train_crop[path][2], train_crop[path][3]
        img = img[y_min:y_max, x_min:x_max]
    return img


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail


def import_training_data():
    training_paths = []
    training_labels = []
    training_crop = {}
    with open(DATA_DIR + FOLDERS_TO_TRAIN) as f:
        folders = list(f.read().splitlines())
    for folder in folders:
        img_folder = DATA_DIR + "train//" + folder + "//images//"
        for paths in os.listdir(img_folder):
            training_paths.append(img_folder + paths)
            training_labels.append(folder)
    for folder in folders:
        img_box = DATA_DIR + "train//" + folder + "//" + folder + "_boxes.txt"
        with open(img_box) as i:
            temp = i.readlines()
            for info in temp:
                formatted_info = info.split()
                training_crop[formatted_info[0]] = list(formatted_info[1:])
    return training_paths, training_labels, training_crop


def import_validation_data():
    validation_paths = []
    validation_labels = []
    validation_crop = {}
    with open(DATA_DIR + VALIDATION_TEXT) as f:
        temp = f.readlines()
        for info in temp:
            formatted_info = info.split()
            validation_paths.append(DATA_DIR + "val//images//" + formatted_info[0])
            validation_labels.append(formatted_info[1])
            validation_crop[formatted_info[0]] = list(map(int, formatted_info[2:]))
    return validation_paths, validation_labels, validation_crop


x_train, y_train, train_crop = import_training_data()

x_val, y_val, val_crop = import_validation_data()

x_train = list(map(preprocess_img, x_train))
x_val = list(map(preprocess_img, x_val))
plt.imshow(x_train[0])
plt.show()
