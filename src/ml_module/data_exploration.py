from matplotlib import pyplot as plt
import numpy as np
import os

def plot_dataset(data, dataset_type):
    temp = set(data)
    temp_dict = {}
    for i, val in enumerate(temp):
        temp_dict[val] = i
    nums = [0 for _ in range(200)]
    for val in data:
        nums[temp_dict[val]] += 1
    plt.figure(figsize=(12, 4))
    plt.bar(range(0, 200), nums)
    plt.title("Distribution of the {} set".format(dataset_type))
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.show()

def plot_average_box_area(dataset_type, y, crops, labels_indexes):
    nums = [0 for _ in range(200)]
    for i, (x_min, y_min, x_max, y_max) in enumerate(crops):
        area = abs(x_min - x_max) * abs(y_min - y_max)
        nums_index = labels_indexes[y[i]]
        nums[nums_index] += area
    nums = [num / 200 for num in nums]
    plt.figure(figsize=(12, 4))
    plt.bar(range(0, 200), nums)
    plt.title("Distribution of bounded box area for {} set".format(dataset_type))
    plt.xlabel("Class number")
    plt.ylabel("Average box area")
    plt.show()

def plot_classes_with_zero_area(dataset_type, y, crops, labels_indexes):
    nums = [0 for _ in range(200)]
    for i, (x_min, y_min, x_max, y_max) in enumerate(crops):
        if x_min - x_max == 0 or y_min - y_max == 0:
            nums_index = labels_indexes[y[i]]
            nums[nums_index] += 1
    plt.figure(figsize=(12, 4))
    plt.bar(range(0, 200), nums)
    plt.title("Number of images with no objects for {} set".format(dataset_type))
    plt.xlabel("Class number")
    plt.ylabel("Number of objectless images")
    plt.show()

def explore_data(x_train, y_train, x_val, y_val, x_test, train_crop, val_crop, labels_indexes):

    # We can see that both the training dataset and the validation dataset have equal distributions
    # for all classes in terms of the number of images. 500 training images per class and 50 validation
    # images per class.
    plot_dataset(y_train, "training")
    plot_dataset(y_val, "validation")

    # Hence the total images (equal distributions for all classes).
    plot_dataset(y_train + y_val, "combined")

    # However, given the bounded box coordinates for the training and validation datasets,
    # we would like to know the distribution of object sizes for each classes. To measure
    # this, we compute the average area of the bounded box for each classes. We then compare
    # the distribution between the training and the validation datasets.

    # From plotting the 2 graphs, it seems that they have the same distribution
    # for box areas.
    plot_average_box_area("training", y_train, train_crop, labels_indexes)
    plot_average_box_area("validation", y_val, val_crop, labels_indexes)

    # One thing that we would also like to know is whether some images have 0 bounded box
    # area, meaning, no objects in the image. 

    # From the graphs plotted by the below functions, we can see that some images have
    # no objects in them.
    plot_classes_with_zero_area("training", y_train, train_crop, labels_indexes)
    plot_classes_with_zero_area("validation", y_val, val_crop, labels_indexes)

    # Because of this fact, we have a feature within our preprocessing component to discard
    # images in the training dataset that have no objects to see if discard such images
    # help to boost validation accuracy.
