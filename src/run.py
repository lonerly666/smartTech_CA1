from ml_module.data_extraction import extract_data
from ml_module.data_preprocess import preprocess
from ml_module.data_exploration import explore_data
from ml_module.models.classifier_model_1 import Classifier_Model_1
from ml_module.models.classifier_model_3 import Classifier_Model_3
from ml_module.models.classifier_model_2 import Classifier_Model_2


(x_train, y_train, train_crop), (x_val, y_val, val_crop) = extract_data()


#LeNet model
m1_x_train, m1_y_train, m1_x_val, m1_y_val = preprocess(
    x_train,
    x_val,
    train_crop,
    val_crop,
    y_train,
    y_val,
    box_apply_type="None",
    color_channel="GRAY",
    blur=True,
    hist_equalization=True,
    resize_to=(64, 64))
explore_data(m1_y_train, m1_y_val)
lenet_model = Classifier_Model_1()
lenet_model.train(m1_x_train, m1_y_train, m1_x_val, m1_y_val)
print(lenet_model.summary())
lenet_model.save("./src/ml_module/saved/model1.h5")


# Modified LeNet model
m2_x_train, m2_y_train, m2_x_val, m2_y_val = preprocess(
    x_train,
    x_val,
    train_crop,
    val_crop,
    y_train,
    y_val,
    box_apply_type="DiscardZeroArea",
    color_channel="GRAY",
    blur=True,
    hist_equalization=True,
    resize_to=(32, 32))
explore_data(m2_y_train, m2_y_val)
modified_lenet_model = Classifier_Model_3()
modified_lenet_model.train(m2_x_train, m2_y_train, m2_x_val, m2_y_val)
print(modified_lenet_model.summary())
modified_lenet_model.save("./src/ml_module/saved/model2.h5")

# AlexNet model
m3_x_train, m3_y_train, m3_x_val, m3_y_val = preprocess(
    x_train,
    x_val,
    train_crop,
    val_crop,
    y_train,
    y_val,
    box_apply_type="None",
    color_channel="RGB",
    blur=False,
    hist_equalization=True,
    resize_to=(64, 64))
explore_data(m3_y_train, m3_y_val)
alexnet_model = Classifier_Model_2()
alexnet_model.train(m3_x_train, m3_y_train, m3_x_val, m3_y_val)
print(alexnet_model.summary())
alexnet_model.save("./src/ml_module/saved/model3.h5")
