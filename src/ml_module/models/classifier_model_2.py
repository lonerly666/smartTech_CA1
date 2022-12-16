from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import ZeroPadding2D
from keras.optimizers import SGD
from matplotlib import pyplot as plt
from tensorflow.keras.layers import PReLU, Dense
from keras.preprocessing.image import ImageDataGenerator

# CNN inspired by http://vision.stanford.edu/teaching/cs231n/reports/2015/pdfs/lucash_final.pdf

class Classifier_Model_2:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv2D(96, (3, 3), activation=PReLU(), input_shape=(64, 64, 1)))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(ZeroPadding2D(padding=(2, 2)))
        self.model.add(Conv2D(256, (5, 5), activation=PReLU()))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(ZeroPadding2D(padding=(1, 1)))
        self.model.add(Conv2D(384, (3, 3), activation=PReLU()))
        self.model.add(ZeroPadding2D(padding=(1, 1)))
        self.model.add(Conv2D(384, (3, 3), activation=PReLU()))
        self.model.add(ZeroPadding2D(padding=(1, 1)))
        self.model.add(Conv2D(256, (3, 3), activation=PReLU()))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(4096, activation=PReLU()))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096, activation=PReLU()))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(200, activation='softmax'))
        self.model.compile(
            SGD(learning_rate=0.01, momentum=0.9, decay=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, x_train, y_train, x_val, y_val):
        datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, shear_range=0.1, rotation_range=5)
        # set batch_size to 1 since we want to use SGD.
        history = self.model.fit(datagen.flow(x_train, y_train, batch_size=1), validation_data=(
            x_val, y_val), epochs=20, verbose=1, shuffle=1)
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
        self.model.save(path+"model2.h5")

    def load(self, path):
        pass
