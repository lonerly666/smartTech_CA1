from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

"""
Modified LeNet model

1st iteration:
- added more convolutional layers
- tweaked the learning rate to 0.0001
- increase the epochs to 80 
- used ImageDataGenerator to augment the data
- accuracy ~12%

2nd iteration:
- increase the epochs to 120 since its not overfitting
- accuracy still low ~16%

"""


class Classifier_Model_3:
    def __init__(self):
        self.model = Sequential([
            Conv2D(120, (3, 3), activation='relu', input_shape=(64, 64, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(60, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(30, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(15, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(500, activation='relu'),
            Dropout(0.5),
            Dense(200, activation='softmax')
        ])
        self.model.compile(
            Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, x_train, y_train, x_val, y_val):
        # Data augmentation
        datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                                     zoom_range=0.1, shear_range=0.1, horizontal_flip=True)
        history = self.model.fit(datagen.flow(x_train, y_train, batch_size=50), validation_data=(
            x_val, y_val), epochs=30, batch_size=50, verbose=1, shuffle=1)
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
        self.model.save(path+"model1.h5")
    
    def summary(self):
        return self.model.summary()

    def load(self, path):
        pass
