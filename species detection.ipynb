import tensorflow as tf
import numpy as np
from PIL import Image

# ask user for image location
img_path = input("Enter image file path: ")

# load and prepare image
img = Image.open(img_path).resize((224,224))
img_array = np.expand_dims(np.array(img)/255.0, axis=0)

# load pretrained model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# predict
pred = model.predict(img_array)
decoded = tf.keras.applications.mobilenet.decode_predictions(pred, top=1)[0][0]

print("Predicted Animal:", decoded[1])
print("Confidence:", round(decoded[2]*100, 2), "%")
