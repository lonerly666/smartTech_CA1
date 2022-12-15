from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.optimizers import Adam


class Classifier_Model_1:
    def __init__(self):
        self.model = Sequential()
        self.model.add(
            Conv2D(60, (5, 5), activation='relu', input_shape=(64, 64, 1)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(30, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(500, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(43, activation='softmax'))
        self.model.compile(
            Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, x_train, y_train, x_val, y_val):
        self.model.fit(x_train, y_train, validation_data=(
            x_val, y_val), epochs=20, batch_size=50, verbose=1, shuffle=1)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def save(self, path):
        self.model.save(path+"model1.h5")

    def load(self, path):
        pass
