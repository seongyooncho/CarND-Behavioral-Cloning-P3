import csv
import cv2
import numpy as np
import gc
from sklearn.utils import shuffle
from sklear.model_selection import train_test_split

samples = []
with open('./data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    line.append(1) #flag for augmentation
    samples.append(line)

def generator(samples, batch_size=32):
  num_samples = len(samples)
  while 1:
    shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]

      images = []
      angles = []

      for batch_sample in batch_samples:
        name = './data/IMG/' + batch_sample[0].split('/')[-1]
        center_image = cv2.imread(name)
        center_angle = float(batch_sample[3])
        images.append(center_image)
        angles.append(center_angle)

      X_train = np.array(images)
      y_train = np.array(angles)
      yield shuffle(X_train, y_train)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

#augmented_images = []
#augmented_measurements = []
#for image, measurement in zip(images, measurements):
#  augmented_images.append(image)
#  augmented_measurements.append(measurement)
#  augmented_images.append(np.fliplr(image))
#  augmented_measurements.append(-measurement)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((75, 20), (0, 0))))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10, verbose=1)
history_object = model.fit_generator(
        train_generator, 
        len(train_samples), 
        validation_data=validation_generator, 
        validation_steps=len(validation_samples),
        epochs=3)

model.save('model.h5')

gc.collect()

print(history_object.history.keys())

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('history.png')
