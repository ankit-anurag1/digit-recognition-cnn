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

classifier.summary()
classifier.save('digit_recognizer.h5')

#performing digit recognition on external handwritten digits
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

#read image file
img = cv2.imread("./Number.jpeg")
if img is None:
    print("No such file exists")
plt.imshow(img, cmap='gray')
plt.show()

#convert to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)
plt.imshow(img, cmap='gray')
plt.show()

#threshholding - digits2black
_, thresh = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY_INV)
plt.imshow(thresh, cmap='gray')
plt.show()

#find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_copy = img.copy()
digits = []
for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)
    cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 3)
    digit = img[y:y+h, x:x+w]
    digit = np.pad(cv2.resize(digit, (18, 18)), ((5, 5), (5, 5)), 'constant', constant_values=255)
    _, digit = cv2.threshold(digit, 100, 255, cv2.THRESH_BINARY_INV)
    digits.append(digit)
    
#show image with bounding rectangle
plt.imshow(img_copy, cmap='gray')
plt.show()

#Shaping the 
#processing the digits
X_pred = np.array(digits) / 255
for X in X_pred:
    print('----------\n----------')
    prediction = classifier_loaded(X.reshape(1, img_rows, img_cols, 1))
    plt.imshow(X.reshape(img_rows, img_cols), cmap='gray')
    plt.show()
    print(np.argmax(prediction, axis=1))
