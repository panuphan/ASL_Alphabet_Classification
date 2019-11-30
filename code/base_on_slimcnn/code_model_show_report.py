#%%
## LOAD Model
from pathlib import Path
from keras.models import load_model
import pickle

slimcnn_model = 'model/slimcnn/2019-12-01_01.23.00-model'
AlexNet_model = 'model/myAlexNet/2019-11-30_19.37.12-model'
VGG16_model = 'model/myVGG16/2019-11-30_23.18.29-model' 

def reload_model(filename_model_path=slimcnn_model):
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

model, train_log = reload_model()

# %%

from matplotlib import pyplot as plt
import matplotlib.image as mpimg

print("The final classification result on validation test...")
print("loss: %.2f" % train_log.history['val_loss'][-1])
print("acc:  %.2f" % train_log.history['val_accuracy'][-1])

loss = train_log.history['loss']
val_loss = train_log.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, color='red', label='Training loss')
plt.plot(epochs, val_loss, color='green', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = train_log.history['accuracy']
val_acc = train_log.history['val_accuracy']
plt.plot(epochs, acc, color='red', label='Training acc')
plt.plot(epochs, val_acc, color='green', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# %%
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

def plot_confusion_matrix(cm, classes,
                      normalize=False,
                      title='Confusion matrix',
                      cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return

def plot_confusion_matrix_with_default_options(y_pred, y_true, classes, options=None):
    '''Plot a confusion matrix heatmap with a default size and default options.'''
    cm = confusion_matrix(y_true, y_pred)
    with sns.axes_style('ticks'):
        plt.figure(figsize=(16, 16))
        if options is not None:
            plot_confusion_matrix(cm, classes, **options)
        else:
            plot_confusion_matrix(cm, classes)
        plt.show()
    return

from glob import glob
import numpy as np
import os, cv2, skimage
from skimage.transform import resize
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
sns.set()
import itertools

def get_data(folder,limit=10):
    imageSize=64
    train_len = limit*len(os.listdir(folder))
    print("num_datas:",train_len)
    X = np.empty((train_len, imageSize, imageSize, 3), dtype=np.float32)
    y = np.empty((train_len,), dtype=np.int)
    cnt = 0

    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['A']:
                label = 0
            elif folderName in ['B']:
                label = 1
            elif folderName in ['C']:
                label = 2
            elif folderName in ['D']:
                label = 3
            elif folderName in ['E']:
                label = 4
            elif folderName in ['F']:
                label = 5
            elif folderName in ['G']:
                label = 6
            elif folderName in ['H']:
                label = 7
            elif folderName in ['I']:
                label = 8
            # elif folderName in ['J']:
            #     label = 9
            elif folderName in ['K']:
                label = 10-1
            elif folderName in ['L']:
                label = 11-1
            elif folderName in ['M']:
                label = 12-1
            elif folderName in ['N']:
                label = 13-1
            elif folderName in ['O']:
                label = 14-1
            elif folderName in ['P']:
                label = 15-1
            elif folderName in ['Q']:
                label = 16-1
            elif folderName in ['R']:
                label = 17-1
            elif folderName in ['S']:
                label = 18-1
            elif folderName in ['T']:
                label = 19-1
            elif folderName in ['U']:
                label = 20-1
            elif folderName in ['V']:
                label = 21-1
            elif folderName in ['W']:
                label = 22-1
            elif folderName in ['X']:
                label = 23-1
            elif folderName in ['Y']:
                label = 24-1
            # elif folderName in ['Z']:
            #     label = 25
            # elif folderName in ['del']:
            #     label = 26
            # elif folderName in ['nothing']:
            #     label = 27
            # elif folderName in ['space']:
            #     label = 28           
            else:
                label = 25
            limit = len(os.listdir(folder + folderName)) if limit is None else limit
            # print(f"folder{label}: {limit}")
            for iter,image_filename in enumerate(os.listdir(folder + folderName)):
                if(iter < limit):
                    img_file = cv2.imread(folder + folderName + '/' + image_filename)
                    if img_file is not None:
                        img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))
                        img_arr = np.asarray(img_file).reshape((-1, imageSize, imageSize, 3))
                        
                        X[cnt] = img_arr
                        y[cnt] = label
                        # print(y[cnt])
                        cnt += 1
                    # X.append(img_arr)
                    # y.append(label)]
                else: continue
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y

train_dir = 'data/input/NewData/train/'
test_dir = 'data/input/NewData/test/'
CLASSES = [os.path.basename(folder) for folder in glob(train_dir + '/*')]
CLASSES.sort()
num_classes=25

X_custom,y_custom=get_data(test_dir,30)#'data/input/NewData/test/'
y_customnHot = to_categorical(y_custom, num_classes=num_classes)
predictions = model.predict(X_custom)
y_pred=np.argmax(predictions,axis=1)
plot_confusion_matrix_with_default_options(y_pred=y_pred, y_true=y_custom, classes=CLASSES)

# %%
report = classification_report(y_pred, y_custom,target_names=CLASSES,output_dict=False)
print(report)
# import pandas as pd
# df = pd.DataFrame(report).transpose()
# df.to_csv('model/slimcnn/report/classification_report.csv', index = False)
# %%
model.summary()

# %%
