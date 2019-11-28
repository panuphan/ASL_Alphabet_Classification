#%%
import numpy as np
import pandas as pd 
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

#%%
def get_data_path(path='data\input\MNIST',verbose=False):
    res = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            file_path=os.path.join(dirname, filename)
            if verbose: print(file_path)
            res.append(file_path)
    return res

def read_data(filename):
    df = pd.read_csv(filename)
    labels = np.asarray(df.iloc[:,:1]).astype(np.float32)
    images = np.asarray(df.iloc[:,1:]).astype(np.float32)
    images = images.reshape((-1,28,28))
    images = np.expand_dims(images, axis=-1)
    
    return images, labels

def creat_model():
    model = keras.models.Sequential([
    keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28,1)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(26, activation='softmax')
    ])
    return model


class custom_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') > 0.8:
            print('Model crossed threshold accuracy, hence stopped training')
            self.mode.stop_training = True
            
callbacks = custom_callback()
#%%
train_datagen = ImageDataGenerator(rescale=1./255.,
                                  rotation_range=40,
                                  width_shift_range=0.25,
                                  height_shift_range=0.25,
                                  shear_range=0.2,
                                  zoom_range=0.3,
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255.)
#%%
test_data_path,train_data_path=get_data_path()
train_images, train_labels = read_data(train_data_path)
test_images, test_labels = read_data(test_data_path)

# %%
model = creat_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_images,train_labels,batch_size=32,epochs=20,verbose=2)

# %%
history.params

# %%
print("555")

# %%
