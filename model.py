import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Define constants used for training
EPOCHS = 30
BATCH_SIZE = 32

# Select recorded data type
# DATA_TYPE = ['track1_normal1', 'track1_normal2', 'track1_backward', 'track2_normal1', 'track2_backward']
DATA_TYPE = ['track1_normal1', 'track1_normal2', 'track1_backward']

# Select model: LeNet or NVIDIA
MODEL = 'NVIDIA'

# Angle correction for left or right camera image
CORRECTION = 0.2

# Load selected data from data folder. Prepare for augmentation (flip, left, right)
samples = []
for datum in DATA_TYPE:
  with open('./data/'+datum+'/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
      #Load center, normal image
      line[0] = './data/'+datum+'/IMG/'+line[0].split('/')[-1]
      line.append(1) #This flag is indicating normal image
      samples.append(line.copy())
      #Load center, flipped image
      line[-1] = -1
      samples.append(line.copy())
      #Load left, normal image
      line[0] = './data/'+datum+'/IMG/'+line[1].split('/')[-1]
      line[3] = float(line[3]) + CORRECTION
      line[-1] = 1
      samples.append(line.copy())
      #Load left, flipped image
      line[-1] = -1
      samples.append(line.copy())
      #Load right, normal image
      line[0] = './data/'+datum+'/IMG/'+line[2].split('/')[-1]
      line[3] = line[3] - CORRECTION * 2
      line[-1] = 1
      samples.append(line.copy())
      #Load left, flipped image
      line[-1] = -1
      samples.append(line.copy())

# Define generator function for training
def generator(samples, batch_size=32):
  num_samples = len(samples)
  while 1:
    shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]

      images = []
      angles = []

      for batch_sample in batch_samples:
        image = cv2.imread(batch_sample[0])
        angle = float(batch_sample[3])
        if (batch_sample[-1] == -1):
          image = np.fliplr(image)
          angle = -angle
        images.append(image)
        angles.append(angle)

      X_train = np.array(images)
      y_train = np.array(angles)
      yield shuffle(X_train, y_train)

# Split samples into train and validation set
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Set generators
train_generator = generator(train_samples, batch_size = BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size = BATCH_SIZE)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from keras import backend as K

# Define Keras model
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# LeNet model
if (MODEL == 'LeNet'):
  model.add(Conv2D(6, (5, 5), activation='relu'))
  model.add(MaxPooling2D())
  model.add(Conv2D(6, (5, 5), activation='relu'))
  model.add(MaxPooling2D())
  model.add(Flatten())
  model.add(Dense(120))
  model.add(Dense(84))

# NVIDIA model
elif (MODEL == 'NVIDIA'):
  model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
  model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
  model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(Flatten())
  model.add(Dense(100))
  model.add(Dense(50))
  model.add(Dense(10))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

# Plot model
plot_model(model, show_shapes=True, to_file='examples/model.png')

# Run generator
history_object = model.fit_generator(
        train_generator, 
        steps_per_epoch = len(train_samples) // BATCH_SIZE + 1, 
        validation_data = validation_generator, 
        validation_steps = len(validation_samples) // BATCH_SIZE + 1,
        epochs=EPOCHS,
        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')])

# Save model to file
model_filename = 'model_' + MODEL + '.h5'
model.save(model_filename)
print('Model saved to ' + model_filename)

# Clear session
K.clear_session()

# Plot and save history image using matplotlib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('examples/history.png')
