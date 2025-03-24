import streamlit as st
from streamlit_drawable_canvas import st_canvas
from sqlalchemy import create_engine, text
from datetime import datetime
from dotenv import load_dotenv
import os
import requests
import numpy as np

load_dotenv()

# Database setup
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = "mnist_predictions"
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@db:5432/{DB_NAME}")

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://api:8000")

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
        image_data = canvas_result.image_data.astype(np.uint8).tolist()  # Convert safely
        response = requests.post(f"{FASTAPI_URL}/predict", json={"image_data": image_data})

        response_json = response.json()
        prediction = response_json["prediction"]
        confidence = response_json["confidence"]

        save_prediction(prediction, true_label if true_label else None)

        st.write(f"Predicted Digit: {prediction}")
        st.write(f"Confidence: {confidence * 100:.2f}%")
