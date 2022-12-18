import io
import zipfile
import requests
import os
import matplotlib.image as mpimg
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
	training_images = []
	training_labels = []
    
	with open(DATA_DIR + FOLDERS_TO_TRAIN) as f:
		folders = [label.strip() for label in f.readlines()]
	
	index = 0
	image_name_index = {}
	for folder in folders:
		img_folder = DATA_DIR + "train//" + folder + "//images//"
		for img_name in os.listdir(img_folder):
			training_images.append(mpimg.imread(img_folder + img_name))
			training_labels.append(folder)
			image_name_index[img_name] = index
			index += 1
	
	training_crop = [None for _ in range(len(training_images))]
	for folder in folders:
		img_box = DATA_DIR + "train//" + folder + "//" + folder + "_boxes.txt"
		with open(img_box) as i:
			temp = i.readlines()
			for info in temp:
				formatted_info = info.split()
				training_crop[image_name_index[formatted_info[0]]] = list(map(int, formatted_info[1:]))
	return training_images, training_labels, training_crop


def import_validation_data():
	validation_images = []
	validation_labels = []
	validation_crop = []	

	with open(DATA_DIR + VALIDATION_TEXT) as f:
		temp = f.readlines()
		for info in temp:
			formatted_info = info.split()
			validation_images.append(mpimg.imread(DATA_DIR + "val//images//" + formatted_info[0]))
			validation_labels.append(formatted_info[1])
			validation_crop.append(list(map(int, formatted_info[2:])))
	return validation_images, validation_labels, validation_crop

def import_test_data():
	test_images = []
	test_images_folder = DATA_DIR + "test//images//"
	for img_name in os.listdir(test_images_folder):
		test_images.append(mpimg.imread(test_images_folder + img_name))
	return test_images



def extract_data():
    download()

    x_train, y_train, train_crop = import_training_data()

    x_val, y_val, val_crop = import_validation_data()

    x_test = import_test_data()

    # maps each label to a specific index in an array of length 200 (the number of classes for tiny imagenet).
	# this is to ensure consistency between each component in the pipeline.
    with open(DATA_DIR + 'wnids.txt') as f:
        labels = [label.strip() for label in f.readlines()]
    labels_indexes = {label:index for index, label in enumerate(labels)}

    # maps each label to its english word
    with open(DATA_DIR + 'words.txt') as f:
        lines = [label.strip().split() for label in f.readlines()]
        labels_english = dict()
        for line in lines:
            if line[0] in labels:
                labels_english[line[0]] = ' '.join(line[1:])

    return (x_train, y_train, train_crop), (x_val, y_val, val_crop), x_test, labels_indexes, labels_english
