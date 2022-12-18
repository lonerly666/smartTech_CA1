from matplotlib import pyplot as plt


def plot_total_images(y_train, y_val):
    temp = set(y_train)
    temp_dict = {}
    for i, val in enumerate(temp):
        temp_dict[val] = i
    nums = [i for i in range(200)]
    for val in y_train:
        nums[temp_dict[val]] += 1
    for val in y_val:
        nums[temp_dict[val]] += 1
    plt.figure(figsize=(12, 4))
    plt.bar(range(0, 200), nums)
    plt.title("Total distribution of the data")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.show()

def plot_training_data(y_train):
    temp = set(y_train)
    temp_dict = {}
    for i, val in enumerate(temp):
        temp_dict[val] = i
    nums = [i for i in range(200)]
    for val in y_train:
        nums[temp_dict[val]] += 1
    plt.figure(figsize=(12, 4))
    plt.bar(range(0, 200), nums)
    plt.title("Distribution of the training set")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.show()

def plot_val_data(y_val):
    temp = set(y_val)
    temp_dict = {}
    for i, val in enumerate(temp):
        temp_dict[val] = i
    nums = [i for i in range(200)]
    for val in y_val:
        nums[temp_dict[val]] += 1
    plt.figure(figsize=(12, 4))
    plt.bar(range(0, 200), nums)
    plt.title("Distribution of the validation set")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.show()


def explore_data(y_train, y_val):
    plot_training_data(y_train)
    plot_val_data(y_val)
    plot_total_images(y_train, y_val)