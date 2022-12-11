import io
import zipfile
import requests
import os
from dotenv import load_dotenv

load_dotenv()


# Path relative to the root directory of this project.
ROOT_DIR = os.getenv('ROOT_DIR')
DATA_DIR = ROOT_DIR + '//tiny-imagenet-200//'

FOLDERS_TO_TRAIN = 'wnids.txt'

VALIDATION_TEXT = 'val/val_annotations.txt'

# Download tiny-imagenet-200 zip file and unzip it if it doesn't exist yet within the designated dir.


def download():
    if os.path.isdir(DATA_DIR):
        print('Dataset already downloaded')
        return

    r = requests.get(
        'http://cs231n.stanford.edu/tiny-imagenet-200.zip', stream=True)
    if not r.ok:
        print('Failed to download the dataset')
        return

    print('Downloading images...')
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(ROOT_DIR)
    z.close()
    print('Finished downloading images')


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
            validation_paths.append(
                DATA_DIR + "val//images//" + formatted_info[0])
            validation_labels.append(formatted_info[1])
            validation_crop[formatted_info[0]] = list(
                map(int, formatted_info[2:]))
    return validation_paths, validation_labels, validation_crop


def extract_data():
    download()

    x_train, y_train, train_crop = import_training_data()

    x_val, y_val, val_crop = import_validation_data()

    return (x_train, y_train, train_crop), (x_val, y_val, val_crop)
