import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Huber loss function
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic_part = 0.5 * tf.square(error)
    linear_part = delta * (abs_error - 0.5 * delta)
    return tf.where(abs_error <= delta, quadratic_part, linear_part)

# Generate dummy data (10 inputs)
X = np.random.rand(10, 10).astype(np.float32)
y = np.random.rand(10, 1).astype(np.float32)

# Build simple neural network
model = Sequential([
    Dense(8, input_shape=(10,), activation='relu'),
    Dense(1, activation='linear')
])

# Compile with Huber loss
model.compile(optimizer='adam', loss=huber_loss, metrics=['mse'])

# Train the model
print("\nTraining model...")
model.fit(X, y, epochs=50, verbose=0)

# Evaluate and predict
loss, mse = model.evaluate(X, y, verbose=0)
predictions = model.predict(X, verbose=0)

print(f"\nFinal Loss: {loss:.4f}, Final MSE: {mse:.4f}")
print("\nTrue values (first 5):\n", y.flatten()[:5])
print("\nPredicted values (first 5):\n", predictions.flatten()[:5])
