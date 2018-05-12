import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, Dropout, Cropping2D
from keras.layers.pooling import MaxPooling2D
#from keras.models import model_from_json

lines = []
with open('W://Udacity/Behavior_training3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
correction = 0.2
for line in lines:
    center_path = line[0]
    left_path = line[1]
    right_path = line[2]
    image_c = cv2.imread(center_path)
    images.append(image_c)
    image_l = cv2.imread(left_path)
    images.append(image_l)
    image_r = cv2.imread(right_path)
    images.append(image_r)
    measurement = float(line[3])
    measurements.append(measurement)
    measurement_l = measurement + correction
    measurement_r = measurement - correction
    measurements.append(measurement_l)
    measurements.append(measurement_r)
    
    
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*(-1.0))

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
#model.add(Dropout(rate = 0.2))
model.add(Dense(50))
#model.add(Dropout(rate = 0.2))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)

model.save('model.h5')