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

history = model.fit(train_images,train_labels,batch_size=32,epochs=10,verbose=2)

# %%
history.params

# %%
# y_pred =model.predict(test_images,test_labels)
score = model.evaluate(test_images, test_labels, verbose=0)
score

# %%
def prediction(pred):
    return(chr(pred+ 65))

# def keras_predict(model, image):
#     data = np.asarray( image, dtype="int32" )
#     pred_probab = model.predict(data)[0]
#     pred_class = list(pred_probab).index(max(pred_probab))
#     # return max(pred_probab), pred_class
#     return prediction(pred_class)

def preprocessing(img):
    # im2 = crop_image(image_frame, 300,300,300,300)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (15,15), 0)
    img = cv2.resize(blur, (28,28), interpolation = cv2.INTER_AREA)
    # img = np.resize(img, (28, 28, 1))
    # img = np.expand_dims(img, axis=0)
    # img = np.array(img)
    img = img.reshape((-1,28,28))
    img = np.expand_dims(img, axis=-1)
    return img.astype(np.float32)

img = cv2.imread('data\input\\test\\1.png')
img_pre = preprocessing(img)
y_pred = model.predict(img_pre)
y_pred = np.argmax(y_pred,axis=1)
# prediction(y_pred)
plt.imshow(np.squeeze(img_pre[0], axis=-1), cmap='binary')
plt.show()
print("5")
# test_images.shape
# img_pre[0]
# from sklearn.metrics import accuracy_score
# y_pred = model.predict(test_images)
# y_pred= np.argmax(y_pred,axis=1)
# # print(f'acc: {accuracy_score(test_labels,y_pred)}')
# y_pred
# %%
