import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import ntpath
import enum
import cv2

BOX_APPLY_TYPE_NONE = "None"
BOX_APPLY_TYPE_CROP = "Crop"
BOX_APPLY_TYPE_ZEROES_OUT = "ZeroesOut"
BOX_APPLY_TYPES = [BOX_APPLY_TYPE_NONE, BOX_APPLY_TYPE_CROP, BOX_APPLY_TYPE_ZEROES_OUT]

# This function crops the images
def crop(img, x_min, y_min, x_max, y_max):
    return img[y_min:y_max, x_min:x_max]

# Instead of cropping the images, this will zeroes out the areas outside of the bounding box.
# TODO: implement this function.
def zeroes_outside_boundedbox(img, x_min, y_min, x_max, y_max):
	pass

# Given bounded boxes points for the training and the validation dataset, transform the images in such a way that makes
# the bounded box area more significant or remove the noise around them.
# If crop is set to false, the function will call the zeroes_outside_boundedbox function.	
def apply_bounded_box(img, crop_points, box_apply_type):
	if box_apply_type not in BOX_APPLY_TYPES:
		raise Exception("Passed box_apply_type value that doesn't exist.")
	
	[x_min, y_min, x_max, y_max] = crop_points
	
	# Apply bounded box based on box_apply_type.
	if box_apply_type == BOX_APPLY_TYPE_CROP:
		return crop(img, x_min, y_min, x_max, y_max)
	elif box_apply_type == BOX_APPLY_TYPE_ZEROES_OUT:
		return zeroes_outside_boundedbox(img, x_min, y_min, x_max, y_max)
	return img

def resize_img(img, width, height):
	if width > img.shape[1] or height > img.shape[0]:
		raise Exception("Upscaling image is not allowed.")
	if width == img.shape[1] and height == img.shape[0]:
		return img
	return cv2.resize(img, (width, height))

def preprocess(x_train, x_val, train_crop, val_crop, box_apply_type="None"):
	def preprocess_pipeline(img_index, img, dataset_type):
		# apply bounded box
		if dataset_type == "train":
			img = apply_bounded_box(img, train_crop[img_index], box_apply_type)
		else:
			img = apply_bounded_box(img, val_crop[img_index], box_apply_type)

		# grayscale
		if len(img.shape) == 3 and img.shape[2] == 3:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		# equalize histogram
		img = cv2.equalizeHist(img)

		# resize image
		new_width, new_height = 64, 64
		img = resize_img(img, new_width, new_height)

		# normalize
		img = img/255
		return img

	x_train = [preprocess_pipeline(i, img, "train") for i, img in enumerate(x_train)]
	x_val = [preprocess_pipeline(i, img, "val") for i, img in enumerate(x_val)]

	return x_train, x_val
