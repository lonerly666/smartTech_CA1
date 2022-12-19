from ml_module.data_extraction import extract_data
from ml_module.data_preprocess import preprocess, preprocess_labels
from ml_module.data_exploration import explore_data
from ml_module.models.classifier_model_1 import Classifier_Model_1
from ml_module.models.classifier_model_3 import Classifier_Model_3
from ml_module.models.classifier_model_2 import Classifier_Model_2
from matplotlib import pyplot as plt
from utils import translate_predictions
import os
import numpy as np
import matplotlib.image as mpimg
from keras.utils.np_utils import to_categorical

(x_train, y_train, train_crop), (x_val, y_val,
                                 val_crop), x_test, labels_indexes, labels_english, test_names_indexes = extract_data()

explore_data(x_train, y_train, x_val, y_val, x_test, train_crop, val_crop, labels_indexes)

#LeNet model
m1_x_train, m1_y_train, m1_x_val, m1_y_val, _ = preprocess(
    x_train,
    x_val,
    x_test,
    train_crop,
    val_crop,
    y_train,
    y_val,
    labels_indexes,
    box_apply_type="None",
    color_channel="GRAY",
    blur=True,
    hist_equalization=True,
    resize_to=(32, 32))
lenet_model = Classifier_Model_1()
lenet_model.train(m1_x_train, m1_y_train, m1_x_val, m1_y_val)
print(lenet_model.summary())
lenet_model.save("./src/ml_module/saved/lenet.h5")


# Modified LeNet model
m2_x_train, m2_y_train, m2_x_val, m2_y_val, _ = preprocess(
    x_train,
    x_val,
    x_test,
    train_crop,
    val_crop,
    y_train,
    y_val,
    labels_indexes,
    box_apply_type="DiscardZeroArea",
    color_channel="GRAY",
    blur=True,
    hist_equalization=True,
    resize_to=(64, 64))
modified_lenet_model = Classifier_Model_3()
modified_lenet_model.train(m2_x_train, m2_y_train, m2_x_val, m2_y_val)
print(modified_lenet_model.summary())
modified_lenet_model.save("./src/ml_module/saved/modified_lenet.h5")

# Modified AlexNet model
m3_x_train, m3_y_train, m3_x_val, m3_y_val, _ = preprocess(
    x_train,
    x_val,
    x_test,
    train_crop,
    val_crop,
    y_train,
    y_val,
    labels_indexes,
    box_apply_type="None",
    color_channel="RGB",
    blur=False,
    hist_equalization=True,
    resize_to=(64, 64))
alexnet_model = Classifier_Model_2()
alexnet_model.build()
alexnet_model.train(m3_x_train, m3_y_train, m3_x_val, m3_y_val)
print(alexnet_model.summary())
alexnet_model.save("./src/ml_module/saved/modified_alexnet.h5")

# Chosen AlexNet model.
_, _, _, _, preprocessed_x_test = preprocess(
    x_train,
    x_val,
    x_test,
    train_crop,
    val_crop,
    y_train,
    y_val,
    labels_indexes,
    box_apply_type="None",
    color_channel="RGB",
    blur=False,
    hist_equalization=True,
    resize_to=(64, 64))

chosen_alexnet_model = Classifier_Model_2()
chosen_alexnet_model.load("./src/ml_module/saved/modified_alexnet.h5")

# Predict the test dataset using the chosen alexnet model.
with open('./predictions.txt', 'w') as f:
    predictions = chosen_alexnet_model.predict(preprocessed_x_test)
    translated_predictions = translate_predictions(
        predictions, labels_indexes, labels_english, test_names_indexes)
    for prediction in translated_predictions:
        f.write("{} {} {}\n".format(
            prediction[0], prediction[1], prediction[2]))

# Predict random images from the test dataset.
random_sample = np.asarray([preprocessed_x_test[13],preprocessed_x_test[9]])
prediction = chosen_alexnet_model.predict(random_sample)
plt.imshow(random_sample[0])
plt.show()
plt.imshow(random_sample[1])
plt.show()
print("Predicted: {}".format(translate_predictions(prediction, labels_indexes, labels_english)))


# Predict accuracy for the annotated test dataset (test_label.txt).
test_data = []
test_result = []
ROOT_DIR = os.getenv('ROOT_DIR')
with open('./test_label.txt') as f:
    test_labels = f.readlines()
    for data in test_labels:
        temp = data.split()
        test_data.append(mpimg.imread(
            ROOT_DIR+"//tiny-imagenet-200//test//images//"+temp[0]))
        test_result.append(temp[1])

test_result, _ = preprocess_labels(test_result, test_result, labels_indexes)
_, _, _, _, preprocessed_test_data = preprocess(
    x_train,
    x_val,
    test_data,
    train_crop,
    val_crop,
    y_train,
    y_val,
    labels_indexes,
    box_apply_type="None",
    color_channel="RGB",
    blur=False,
    hist_equalization=True,
    resize_to=(64, 64))

# Loss: 5.12, Accuracy: 5.55%
print(chosen_alexnet_model.evaluate(preprocessed_test_data, test_result))
