import keras
import tensorflow as tf
from keras import layers, optimizers, losses
import numpy as np
from sklearn.model_selection import train_test_split
import os

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

x=np.load('training_data/data.npy')
y=np.load('training_data/target.npy')

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.1)
x_train=x_train.reshape(-1, 125, 125, 1)
x_test=x_test.reshape(-1, 125, 125, 1)

model=keras.Sequential([
    layers.Conv2D(128, (3, 3), activation='relu', input_shape=(125, 125, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(1024, activation='relu'),
    layers.Dense(2, activation='softmax')

])

model.compile(
    loss=losses.SparseCategoricalCrossentropy(),
    optimizer=optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)
model.fit(x_train, y_train, batch_size=5, epochs=10, verbose=2)
model.evaluate(x_test, y_test, batch_size=5)
model.save('mask 2.model')
