from ml_module.data_extraction import extract_data
from ml_module.data_preprocess import preprocess
from ml_module.models.classifier_model_1 import Classifier_Model_1
from ml_module.models.classifier_model_3 import Classifier_Model_3



(x_train, y_train, train_crop), (x_val, y_val, val_crop) = extract_data()

x_train, y_train, x_val, y_val = preprocess(
    x_train, x_val, train_crop, val_crop, y_train, y_val)


x_train = x_train.reshape(len(x_train), 64, 64, 1)
x_val = x_val.reshape(len(x_val), 64, 64, 1)
model_a = Classifier_Model_3()
model_a.train(x_train, y_train, x_val, y_val)
del model_a
