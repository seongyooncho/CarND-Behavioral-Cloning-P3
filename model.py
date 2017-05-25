import csv
import cv2
import numpy as np
import gc
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

BATCH_SIZE = 32

samples = []
with open('./data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    line.append(1)
    samples.append(line)
    line[-1] = -1
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
        name = './data/IMG/'+batch_sample[0].split('/')[-1]
        center_image = cv2.imread(name)
        center_angle = float(batch_sample[3])
        if (batch_sample[-1] == -1):
          center_image = np.fliplr(center_image)
          center_angle = - center_angle
        images.append(center_image)
        angles.append(center_angle)

      X_train = np.array(images)
      y_train = np.array(angles)
      yield shuffle(X_train, y_train)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, batch_size = BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size = BATCH_SIZE)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((75, 20), (0, 0))))
model.add(Conv2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(
        train_generator, 
        steps_per_epoch = len(train_samples) // BATCH_SIZE + 1, 
        validation_data = validation_generator, 
        validation_steps = len(validation_samples) // BATCH_SIZE + 1,
        epochs=10)

model.save('model.h5')

gc.collect()

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
