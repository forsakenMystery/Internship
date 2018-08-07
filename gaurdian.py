import keras
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

number_classes = 10
image_size = 28
image_flatten = image_size * image_size
number_channels = 1

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

Y_train = train["label"]
X_train = train.drop(labels=['label'], axis=1)
X_train = X_train / 255.0
test = test / 255.0
X_train = X_train.values.reshape(-1, image_size, image_size, number_channels)
test = test.values.reshape(-1, image_size, image_size, number_channels)
Y_train = keras.utils.np_utils.to_categorical(Y_train, num_classes=number_classes)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=2)
model = keras.models.load_model('weights-improvement-22-1.00.hdf5')

model.summary()
results=model.predict(test)
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)