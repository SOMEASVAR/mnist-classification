# Convolutional Deep Neural Network for Digit Classification

## Aim:

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement:
Digit classification and to verify the response for scanned handwritten images.
## Dataset:
The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

## Neural Network Model:



## DESIGN STEPS:
### STEP 1:
Import tensorflow and preprocessing libraries

### STEP 2:
Build a CNN model

### STEP 3:
Compile and fit the model and then predict

## Program:
## Libraries:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
```
## Data Loading and Shaping:
```
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape
X_test.shape
single_image= X_train[0]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
```
## One hot encoding:
```
X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
```
## Reshape Inputs:
```
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
```
## Build CNN Model:
```
model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
```
## Metrics:
```
metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['loss','val_loss']].plot()
metrics[['accuracy','val_accuracy']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))
```
## Predicting for own Hand-Written input:
```
img = image.load_img('2.jpg')

type(img)

img = image.load_img('2.jpg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)


print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)
```

## Output:

### Accuracy, Validation Accuracy Vs Iteration:
![image](https://github.com/SOMEASVAR/mnist-classification/assets/93434149/54911c63-d27b-4835-b465-ac5ee5b1784c)
### Training Loss, Validation Loss Vs Iteration:
![image](https://github.com/SOMEASVAR/mnist-classification/assets/93434149/b62a79ea-19fc-44f7-ae37-019809ac0dee)


### Classification Report:
![image](https://github.com/SOMEASVAR/mnist-classification/assets/93434149/a3e0a96b-0942-4d12-9526-c022e7dcf5f8)


### Confusion Matrix:

![image](https://github.com/SOMEASVAR/mnist-classification/assets/93434149/751dffd7-abef-4272-b379-76050ac296f5)


### New Sample Data Prediction:

![image](https://github.com/SOMEASVAR/mnist-classification/assets/93434149/1c0acbde-fbb4-4a62-b7e6-1486059e7ec2)
![image](https://github.com/SOMEASVAR/mnist-classification/assets/93434149/0ad49d2d-aef8-41d0-90da-0c9eb69d5e49)


## Result:
Thus to create a program for creating convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is created and compiled successfully.
