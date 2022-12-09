import numpy as np
import matplotlib.pyplot as plt
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

DATA_DIR = "C://Users//PC//Downloads//tiny-imagenet-200//tiny-imagenet-200//"
FOLDERS_TO_TRAIN = "wnids.txt"
VALIDATION_TEXT = "val_annotations"


# def region_of_interest(image):


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail


def import_training_data():
    training_paths = []
    training_labels = []
    with open(DATA_DIR + FOLDERS_TO_TRAIN) as f:
        folders = list(f.read().splitlines())
    for folder in folders:
        img_folder = DATA_DIR + "train//" + folder + "//images//"
        for paths in os.listdir(img_folder):
            training_paths.append(img_folder + paths)
            training_labels.append(folder)
    return training_paths, training_labels


def import_validation_data():
    validation_paths = []
    validation_labels=[]
    with open(DATA_DIR+)


# x_train, y_train = import_training_data()

# print(x_train)