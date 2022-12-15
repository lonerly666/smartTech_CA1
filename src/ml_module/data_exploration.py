from matplotlib import pyplot as plt
import numpy as np
import random

def plot_training_data(x_train, y_train):
    num_of_samples = []
    cols = 5
    num_classes = 200
    fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 10))
    fig.tight_layout()
    for i in range(cols):
        for j in range(num_classes):
            x_selected = x_train[y_train == j]
            axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("gray"))
            axs[j][i].axis("off")
            if i == 2:
                num_of_samples.append(len(x_selected))

    plt.figure(figsize=(12, 4))
    plt.bar(range(0, num_classes), num_of_samples)
    plt.title("Distribution of the training dataset")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.show()


def explore_data(x_train, y_train, x_val, y_val):
    plot_training_data(x_train, y_train)