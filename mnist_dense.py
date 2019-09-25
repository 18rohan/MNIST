import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.utils import to_categorical
import seaborn as sns
from sklearn.model_selection import train_test_split

## importing data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

type(train)
train.head()
#shape of data imported
train.shape, test.shape

train.iloc[:,0].tail()


#converting to the data to array
train_data = train.iloc[:,1:]
train_labels = train.iloc[:,0]

np.unique(train_labels)
type(train_labels)

train_data = np.array(train_data)
train_labels = np.array(train_labels)

train_labels.shape
#one hot encoding for train_labels
train_labels = to_categorical(train_labels, num_classes = 10)
train_labels.shape

type(train_data)

train_data = np.array(train_data)
test = np.array(test)

train_data = train_data.astype('float32')

test = test.astype('float32')
train_data = train_data / 255.0
test = test / 255.0

# plotting an image
img = train_data[1].reshape(28,28)
plt.imshow(img, cmap = 'bone')

#plotting graphs for the train_labels
sns.countplot(train['label'])

#train_labels data
np.unique(train_labels)

#splitting data to training and validation data
x_train, x_val, y_train, y_val = train_test_split(train_data,train_labels, random_state = 0, shuffle = True)
x_train.shape, x_val.shape


#Dense model
input_dim = 784
model = Sequential()

model.add(Dense(256, activation = 'relu', input_dim = 784))
model.add(BatchNormalization())
model.add(Dense(256, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(10, activation = 'softmax'))

#COMPILING MODEL
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Fitting the MODEL

model.fit(x_train,y_train, epochs = 25, batch_size = 120, validation_data =(x_val,y_val), verbose = 1)
