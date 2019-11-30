#%%
from pathlib import Path
from keras.models import load_model
import pickle , cv2
import numpy as np
import os, cv2, skimage
from skimage.transform import resize
from keras.utils.np_utils import to_categorical

slimcnn_model = 'model/slimcnn/2019-12-01_01.23.00-model'
AlexNet_model = 'model/myAlexNet/2019-11-30_19.37.12-model'
VGG16_model = 'model/myVGG16/2019-11-30_23.18.29-model'

def reload_model(filename_model_path=AlexNet_model):
    MODEL_SAVE_DIR = filename_model_path + '.h5'
    MODEL_SAVE_WEIGHTS_DIR = filename_model_path + '.weights.h5'
    MODEL_SAVE_TRAIN_LOG_DIR = filename_model_path + '-train-log.pickle'

    old_model_file = Path(MODEL_SAVE_DIR)
    old_weight_file = Path(MODEL_SAVE_WEIGHTS_DIR)
    old_train_log_file = Path(MODEL_SAVE_TRAIN_LOG_DIR)
    if old_model_file.is_file() and old_weight_file.is_file() and old_train_log_file.is_file():
        print("Reloading old model, weights and training log from disk")
        model = load_model(MODEL_SAVE_DIR)
        model.load_weights(MODEL_SAVE_WEIGHTS_DIR)
        with open(MODEL_SAVE_TRAIN_LOG_DIR, 'rb') as file:
            train_log = pickle.load(file)
        print("Done!")
        return model, train_log
    else:
        print("Cannot reload the old model, weight and training log from\n  * \"%s\"\n  * \"%s\"\n  * \"%s\""
              % (MODEL_SAVE_DIR, MODEL_SAVE_WEIGHTS_DIR, MODEL_SAVE_TRAIN_LOG_DIR))
        print("Please check if the path is correct or not")
        return None, None


def read_image(image_path):
    imageSize=64
    img_file = cv2.imread(image_path)
    if img_file is not None:
        img_file = skimage.transform.resize(
            img_file, (imageSize, imageSize, 3))
        img_arr = np.asarray(img_file).reshape((-1, imageSize, imageSize, 3))

        return img_arr


def read_label(prediction):
    CLASSES = ['A',
               'B',
               'C',
               'D',
               'E',
               'F',
               'G',
               'H',
               'I',
               'K',
               'L',
               'M',
               'N',
               'O',
               'P',
               'Q',
               'R',
               'S',
               'T',
               'U',
               'V',
               'W',
               'X',
               'Y']
    return CLASSES[(np.argmax(prediction, axis=1))[0]]


#**** EXAMPLE TO USE ****
model, train_log = reload_model() 
#%%

test_path = 'data/input/pred/B/B1.jpg'
img = read_image(test_path)
predictions = model.predict(img)
print('predictions:', read_label(predictions))



# %%
