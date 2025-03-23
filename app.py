import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import streamlit as st
from model import MNISTClassifier
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import cv2 
import psycopg2
from sqlalchemy import create_engine, text
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

# Database setup
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = "mnist_predictions"
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@localhost/{DB_NAME}")

# Load the trained model
model = MNISTClassifier()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

def preprocess_image(image_data):
    """Prepares the drawn digit for MNIST model prediction."""
    # Convert to grayscale
    if image_data.shape[-1] == 4:  # If RGBA (canvas output)
        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGBA2GRAY)
    elif image_data.shape[-1] == 3:  # If RGB
        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)

    # Invert colors (MNIST has white digits on black background)
    image_data = 255 - image_data

    # Threshold to make sure we have a clear digit
    _, thresh = cv2.threshold(image_data, 127, 255, cv2.THRESH_BINARY)

    # Find contours (digit boundaries)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get bounding box around the digit
        x, y, w, h = cv2.boundingRect(np.vstack(contours))  # Consider all contours
        digit = image_data[y:y+h, x:x+w]  # Crop to bounding box
    else:
        digit = image_data  # If no contours, keep original

    # Resize digit to fit within 20x20 while keeping aspect ratio
    h, w = digit.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Create a 28x28 black canvas and center the digit
    final_image = np.zeros((28, 28), dtype=np.uint8)
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2
    final_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit

    # Convert back to python image and apply transforms
    final_image = Image.fromarray(final_image)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    return transform(final_image).unsqueeze(0)  # Convert to batch tensor

    # # Debug image output to check preprocessing
    # image = transform(final_image) 
    # fig, ax = plt.subplots()
    # ax.imshow(image.squeeze(0), cmap="gray")  # Remove channel dimension for display
    # ax.axis("off")
    # st.pyplot(fig)  # Display in Streamlit
    # return image.unsqueeze(0)

def save_prediction(predicted_digit, true_label=None):
    """Saves the prediction to the PostgreSQL database."""
    timestamp = datetime.now()

    with engine.connect() as connection:
        query = text("SELECT current_user;")
        result = connection.execute(query).scalar()
        print(f"Current User: {result}")

        query = text("""
            INSERT INTO predictions (timestamp, predicted_digit, true_label)
            VALUES (:timestamp, :predicted_digit, :true_label);
        """)
        connection.execute(query, {'timestamp': timestamp, 'predicted_digit': predicted_digit, 'true_label': true_label if true_label else None})
        connection.commit()

# Streamlit app
st.title("MNIST Digit Prediction")

# Canvas for drawing the digit
st.subheader("Draw a digit")
canvas_result = st_canvas(
    background_color="white",
    stroke_width=10,
    stroke_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Input for true label
true_label = st.text_input("Enter the real digit value (optional)")

if st.button("Predict"):
    if canvas_result.image_data is not None:
        image_data = np.array(canvas_result.image_data)
        image = preprocess_image(image_data)
        with torch.no_grad():
            output = model(image)
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.max(torch.nn.functional.log_softmax(output, dim=1)).exp().item()

        save_prediction(prediction, true_label if true_label else None)

        st.write(f"Predicted Digit: {prediction}")
        st.write(f"Confidence: {confidence * 100:.2f}%")
