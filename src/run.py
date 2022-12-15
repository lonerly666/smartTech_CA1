from ml_module.data_extraction import extract_data
from ml_module.data_preprocess import preprocess
import matplotlib.pyplot as plt

(x_train, y_train, train_crop), (x_val, y_val, val_crop) = extract_data()

x_train, y_train, x_val, y_val = preprocess(x_train, x_val, train_crop, val_crop, y_train, y_val)

plt.imshow(x_train[0])
plt.show()
