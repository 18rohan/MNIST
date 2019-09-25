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

train.iloc[:,0].head()


#converting to the data to array
train_data = train.iloc[:,1:]
train_labels = train.iloc[:,0]


#one hot encoding for train_labels
train_labels = to_categorical(train_labels, num_classes = 10)
train_labels.shape



train_data = np.array(train_data)
test = np.array(test)

train_data = train_data.astype('float32')
train_labels = train_labels.astype('int32')
test = test.astype('float32')
train_data = train_data / 255.0
test = test / 255.0

# plotting an image
img = train_data[999].reshape(28,28)
plt.imshow(img, cmap = 'bone')

#plotting graphs for the train_labels
sns.countplot(train['label'])

#train_labels data
train_labels

#splitting data to training and validation data
x_train, x_val, y_train, y_val = train_test_split(train_data,train_labels, random_state = 0, shuffle = True)
x_train.shape, x_val.shape

#converting the data to keras required dimensions
x_train = x_train.reshape(x_train.shape[0], img_rows,img_cols, 1)
y_train = np.array(y_train)
y_train = to_categorical(y_train)

x_train.shape, y_train.shape

#validation data
x_val = x_val.reshape(x_val.shape[0],img_rows, img_cols, 1)
y_val = np.array(y_val)
y_val = to_categorical(y_val, num_classes = 10)

x_val.shape, y_val.shape

# Conv2D MODEL
img_rows, img_cols = 28,28
img_shape = (img_rows, img_cols, 1)


model = Sequential()

model.add(Conv2D(128, (3,3), activation = 'relu', kernel_initializer = 'he_normal', input_shape = img_shape))
model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(256, (3,3), activation = 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(512, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))


#COMPILING MODEL
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Fitting the MODEL

model.fit(x_train,y_train, epochs = 1, batch_size = 120, validation_data =(x_val,y_val), verbose = 1)
