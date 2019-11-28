from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
# from keras.models import load_model

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model_path = 'model/mobilenet_teachable_ML/keras_model.h5'
label_path = 'model/mobilenet_teachable_ML/labels.txt'
# model = tensorflow.keras.models.load_model(model_path)
model = load_model(model_path)

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image_path = 'data/input/train/A-samples/12.jpg'
image = Image.open(image_path)

# Make sure to resize all images to 224, 224 otherwise they won't fit in the array
image = image.resize((224, 224))
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)
print(prediction)
