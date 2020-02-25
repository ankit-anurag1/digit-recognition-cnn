# import numpy as np
# import pandas as pd
# from tensorflow import keras 

# #variables
# num_classes = 10
# img_rows, img_cols = 28, 28

# #obtaining and preprocessing data
# input_path = './input/'
# dataset = pd.read_csv(input_path + 'train.csv')
# #number of rows in dataset
# num_images = dataset.shape[0]
# X = dataset.values[:,1:].reshape(num_images, img_rows, img_cols, 1) / 255
# #use onehotencoding on output
# y = keras.utils.to_categorical(dataset.label, num_classes)

# #organising test data
# test_dataset = pd.read_csv(input_path + 'test.csv')
# test_num_images = test_dataset.shape[0]
# test_X = test_dataset.values[:, :].reshape(test_num_images, img_rows, img_cols, 1) / 255

# #creating the CNN model
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

# classifier = Sequential()
# classifier.add(Conv2D(32, kernel_size = (3, 3), activation='relu',
#                       input_shape = (img_rows, img_cols, 1)))
# classifier.add(Dropout(0.5))
# classifier.add(Conv2D(64, kernel_size = (3, 3), activation='relu'))
# classifier.add(Dropout(0.5))
# classifier.add(Flatten())
# classifier.add(Dense(128, activation='relu'))
# classifier.add(Dense(num_classes, activation='softmax'))

# #fitting and training model
# classifier.compile(optimizer='adam', loss='categorical_crossentropy',
#                   metrics=['accuracy'])
# classifier.fit(X, y,
#               batch_size=128,
#               epochs=2,
#               validation_split=0.2)

# #prediction on test data and formatting
# preds = classifier.predict(test_X)
# results = np.argmax(preds, axis=1)