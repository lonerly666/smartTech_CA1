from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.optimizers import Adam
from matplotlib import pyplot as plt

"""
# LeNet model
1st Iteration:
- overfitting
- accuracy ~ 10% 
- grayscale, size 64, 64, applied gaussian blur, histogram equalization

2nd iteration:
- used tanh as activation function
- validation accuracy: 10%
- fixed overfitting
- grayscale, resized to 32, 32, applied gaussian blur, histogram equalization

"""
class Classifier_Model_1:
    def __init__(self):
        self.model = Sequential(
            [
                Conv2D(60, (5, 5), activation='tanh', input_shape=(32, 32, 1)),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(30, (3, 3), activation='tanh'),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(500, activation='tanh'),
                Dropout(0.8),
                Dense(200, activation='softmax'),
            ]
        )
        self.model.compile(
            Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, x_train, y_train, x_val, y_val):
        history = self.model.fit(x_train, y_train, batch_size=50, validation_data=(
            x_val, y_val), epochs=15, verbose=1, shuffle=1)
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
        pass
