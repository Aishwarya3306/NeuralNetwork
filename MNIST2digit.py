import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf # Ensure tf is imported for model functions

# Assuming 'model' variable (trained in cell eeiP6O4VeO9w) is still in the kernel's scope

def preprocess_two_digit_image(image_path):
    """
    Loads a user's image, assumes it contains two digits side-by-side,
    resizes it to 28x56, splits it into two 28x28 images,
    and normalizes them for prediction.
    """
    img = Image.open(image_path).convert('L')  # Convert to grayscale

    # Resize to a standard size for two digits (28 height, 56 width)
    # This assumes the input image is an approximate concatenation of two 28x28 digits.
    img = img.resize((56, 28)) # width, height

    img_array = np.array(img)

    # Invert colors if background is white and digit is black (common for handwritten digits)
    if img_array.mean() > 127: # Check if it's a light image (likely white background)
        img_array = 255 - img_array

    # Normalize pixel values to 0-1 range
    img_array = img_array.astype("float32") / 255.0

    # Split the 56-pixel wide image into two 28x28 pixel images
    digit1_img = img_array[:, :28]  # First digit (left half)
    digit2_img = img_array[:, 28:] # Second digit (right half)

    # Reshape for model prediction (add channel dimension and batch dimension)
    digit1_input = digit1_img.reshape(1, 28, 28, 1)
    digit2_input = digit2_img.reshape(1, 28, 28, 1)

    return digit1_input, digit2_input, img_array # Return original combined for display

# --- Main execution for user input and prediction ---

image_path = input("Enter the path to the image containing two handwritten digits: ")

try:
    digit1_input, digit2_input, combined_img_display = preprocess_two_digit_image(image_path)

    # Make predictions using the global 'model' object
    pred1_probs = model.predict(digit1_input, verbose=0)[0]
    pred2_probs = model.predict(digit2_input, verbose=0)[0]

    predicted_digit1 = np.argmax(pred1_probs)
    predicted_digit2 = np.argmax(pred2_probs)

    combined_prediction = int(str(predicted_digit1) + str(predicted_digit2))

    print(f"\nPredicted first digit: {predicted_digit1}")
    print(f"Predicted second digit: {predicted_digit2}")
    print(f"Combined predicted number: {combined_prediction}")

    # Display the combined input image and the individual predicted digits
    plt.figure(figsize=(8, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(combined_img_display, cmap='gray')
    plt.title(f"Input: {image_path.split('/')[-1]}")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(digit1_input.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_digit1}")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(digit2_input.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_digit2}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Error: Image file not found at '{image_path}'. Please check the path and try again.")
except Exception as e:
    print(f"An error occurred during processing: {e}")
