from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import ZeroPadding2D
from keras.optimizers import SGD, Adam
from matplotlib import pyplot as plt
from keras.layers import PReLU, Dense
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow import keras

"""
CNN inspired by http://vision.stanford.edu/teaching/cs231n/reports/2015/pdfs/lucash_final.pdf

# iteration 2
Was able to get 23% validation accuracy with the following conditions:
1. No resizing (64, 64)
2. No grayscaling (since AlexNet is good for colored images)
3. No blurring
4. No histogram equalization
5. 20 epoch
a bit overfitting.

# iteration 3
19% for training and validation accuracy (not overfitting) - still need to do more epochs
1. histogram equalization
2. L2 regularization to further combat overfitting
3. add vertical flip param to imagedatagen
4. 50 epoch
5. Dropout layer to 0.7
"""

# iteration 4
# achieves the same result as the previous iteration (the only difference is it used 80 epochs.)

class Classifier_Model_2:
    def __init__(self):
        self.model = None

    def build(self):
        regularizer = l2(0.01)
        self.model = Sequential()
        self.model.add(Conv2D(96, (3, 3), activation=PReLU(), input_shape=(64, 64, 3), kernel_regularizer=regularizer))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(256, (5, 5), activation=PReLU(), padding='same', kernel_regularizer=regularizer))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(384, (3, 3), activation=PReLU(), padding='same', kernel_regularizer=regularizer))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(384, (3, 3), activation=PReLU()))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(256, (3, 3), activation=PReLU(), padding='same', kernel_regularizer=regularizer))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(4096, activation=PReLU()))
        self.model.add(Dropout(0.7))
        self.model.add(Dense(4096, activation=PReLU()))
        self.model.add(Dropout(0.7))
        self.model.add(Dense(200, activation='softmax'))
        self.model.compile(
            SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, x_train, y_train, x_val, y_val):
        datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, shear_range=0.1, horizontal_flip=True, vertical_flip=True)
        # set batch_size to 1 since we want to use SGD.
        history = self.model.fit(datagen.flow(x_train, y_train, batch_size=50), validation_data=(
            x_val, y_val), epochs=80, verbose=1, shuffle=1)
        # Check if overfitting
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['training', 'validation'])
        plt.title('Loss')
        plt.xlabel('epoch')
        plt.show()

    def predict(self, x_test):
        return self.model.predict(x_test)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def save(self, path):
        self.model.save(path)

    def summary(self):
        return self.model.summary()

    def load(self, path):
        self.model = keras.models.load_model(path)
