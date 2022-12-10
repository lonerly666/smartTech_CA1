from dotenv import load_dotenv

load_dotenv()

import os
import requests
import zipfile
import io
import ntpath
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

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
	
	r = requests.get('http://cs231n.stanford.edu/tiny-imagenet-200.zip', stream=True)
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
			validation_paths.append(DATA_DIR + "val//images//" + formatted_info[0])
			validation_labels.append(formatted_info[1])
			validation_crop[formatted_info[0]] = list(map(int, formatted_info[2:]))
	return validation_paths, validation_labels, validation_crop

# Apply bounded box (the values are provided by the dataset)
def crop(img, train_crop, val_crop):
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
	_, tail = ntpath.split(path)
	return tail

def extract_data():
	download()
	
	x_train, y_train, train_crop = import_training_data()

	x_val, y_val, val_crop = import_validation_data()
	
	print('Applying bounded box to the training and validation images')
	x_train = [crop(x, train_crop, val_crop) for x in x_train]
	x_val = [crop(x, train_crop, val_crop) for x in x_val]
		
	plt.imshow(x_train[0])
	plt.show()
	
	return (x_train, y_train), (x_val, y_val)
