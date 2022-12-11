import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import ntpath


def path_leaf(path):
    _, tail = ntpath.split(path)
    return tail

# Apply bounded box (the values are provided by the dataset)
def crop(img, train_crop, val_crop):
    path = path_leaf(img)
    img = mpimg.imread(img)
    if path in val_crop:
        x_min, y_min, x_max, y_max = val_crop[path][0], val_crop[path][1], val_crop[path][2], val_crop[path][3]
        img = img[y_min:y_max, x_min:x_max]
    else:
        train_crop[path] = list(map(int, train_crop[path]))
        x_min, y_min, x_max, y_max = train_crop[path][0], train_crop[
            path][1], train_crop[path][2], train_crop[path][3]
        img = img[y_min:y_max, x_min:x_max]
    return img


def preprocess(x_train, x_val, train_crop, val_crop):
    print("Applying bounded box to the images...")
    x_train = [crop(img, train_crop, val_crop) for img in x_train]
    x_val = [crop(img, train_crop, val_crop) for img in x_val]
    return x_train, x_val
