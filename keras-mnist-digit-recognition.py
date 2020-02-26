import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

#variables
num_classes = 10
img_rows, img_cols = 28, 28

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# #obtaining and preprocessing data
# #number of rows in dataset
num_train_images = X_train.shape[0]
num_test_images = X_test.shape[0]

X_train = X_train.reshape(num_train_images, img_rows, img_cols, 1) / 255
X_test = X_test.reshape(num_test_images, img_rows, img_cols, 1) / 255
# #use onehotencoding on output
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

#creating the NN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

classifier = Sequential()
classifier.add(Conv2D(32, kernel_size = (3, 3), activation='relu',
                      input_shape = (img_rows, img_cols, 1)))
classifier.add(Dropout(0.5))
classifier.add(Conv2D(64, kernel_size = (3, 3), activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Flatten())
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(num_classes, activation='softmax'))

#fitting and training model
classifier.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
classifier.fit(X_train, y_train,
              batch_size=200,
              epochs=3,
              validation_data=(X_test, y_test))
