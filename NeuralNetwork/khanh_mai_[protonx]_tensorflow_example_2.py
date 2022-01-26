# -*- coding: utf-8 -*-

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
import matplotlib.pyplot as plt
plt.imshow(train_images[0])

train_labels[0]

train_images = train_images / 255.0 # Normalize # Standardize

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

train_images.shape

# 1. Tạo mạng Sequential
model = Sequential()
# 2. Thêm lớp Flatten() # 28 * 28 thành vector có chiều (784,)
model.add(Flatten(input_shape=(28, 28)))
# 3. Thu về miền của nhãn
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['acc'])

train_labels[0]

train_images.shape[0] / 32

model.fit(train_images, train_labels, epochs=15, batch_size=64)

test_images = test_images / 255.0

model.evaluate(test_images, test_labels)

test_images[:1, :].shape

model.predict(test_images[:1, :])

import numpy as np
np.sum(model.predict(test_images[:1, :]))

np.argmax(model.predict(test_images[4:5, :]))

plt.imshow(test_images[4])

# Lưu Model và nộp 
model.save("mymodel2.h5")