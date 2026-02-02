# digit_prediction_from_image.py

import tensorflow as tf
import numpy as np
from PIL import Image

# ---- 1) Load and prepare MNIST ----
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

x_train = x_train.reshape(-1,28,28,1)
x_test  = x_test.reshape(-1,28,28,1)

# ---- 2) Build a simple CNN ----
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ---- 3) Train (quick) ----
print("Training model...")
model.fit(x_train, y_train, epochs=3, batch_size=128,
          validation_data=(x_test, y_test), verbose=2)

# ---- 4) Ask for image path ----
path = input("Enter path to your digit image (PNG/JPG): ").strip()

# ---- 5) Load and preprocess that image ----
img = Image.open(path).convert("L")     # grayscale
img = img.resize((28,28))               # MNIST size
img = np.array(img)

# if background is white and digit is dark, invert automatically:
if img.mean() > 127:
    img = 255 - img

img = img/255.0                         # normalize
img = img.reshape(1,28,28,1).astype("float32")

# ---- 6) Predict ----
pred = model.predict(img, verbose=0)
digit = np.argmax(pred)

print("\nPredicted digit:", digit)
