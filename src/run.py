from ml_module.data_extraction import extract_data
from ml_module.data_preprocess import preprocess
from ml_module.data_exploration import explore_data
from ml_module.models.classifier_model_1 import Classifier_Model_1
from ml_module.models.classifier_model_3 import Classifier_Model_3
from ml_module.models.classifier_model_2 import Classifier_Model_2


(x_train, y_train, train_crop), (x_val, y_val, val_crop) = extract_data()

x_train, y_train, x_val, y_val = preprocess(x_train, x_val, train_crop, val_crop, y_train, y_val)


model1 = Classifier_Model_1()
model1.train(x_train, y_train, x_val, y_val)