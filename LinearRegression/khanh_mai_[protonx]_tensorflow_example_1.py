# -*- coding: utf-8 -*-


# Import thư viện cần thiết 
import tensorflow as tf
import numpy as np

# Dữ liệu 
xs = [2, 7, 9, 3, 10, 6, 1, 8]
ys = [13, 35, 41, 19, 45, 28, 10, 55]

xs = np.array(xs)
ys = np.array(ys)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Định nghĩa Model cho bài toán Linear Regression 
model = Sequential()
model.add(Dense(1, input_shape=[1]))

model.compile(optimizer='sgd', loss='mean_squared_error')

model.summary()

# Train model 
# TODO 
model.fit(xs, ys, epochs=1000)

model.predict([10, 50])

# Lưu Model và nộp 
model.save("mymodel.h5")