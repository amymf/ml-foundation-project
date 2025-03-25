from fastapi import FastAPI
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from model import MNISTClassifier
from pydantic import BaseModel

app = FastAPI()

# Load model
model = MNISTClassifier()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

class ImageDataRequest(BaseModel):
    image_data: list  # List of pixel values as integers

def preprocess_image(image_data):
    """Prepares the drawn digit for MNIST model prediction."""
    image_data = np.array(image_data, dtype=np.uint8)
    # Convert to grayscale
    if image_data.shape[-1] == 4:
        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGBA2GRAY)
    elif image_data.shape[-1] == 3:
        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)

    # Invert colors (MNIST has white digits on black background)
    image_data = 255 - image_data

    # Threshold to make sure we have a clear digit
    _, thresh = cv2.threshold(image_data, 127, 255, cv2.THRESH_BINARY)

    # Find contours (digit boundaries)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get bounding box around the digit
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        digit = image_data[y:y+h, x:x+w]  # Crop to bounding box
    else:
        digit = image_data

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

@app.post("/predict")
async def predict(request: ImageDataRequest):
    """Receives image as list of pixel values and returns prediction."""
    image_data = np.array(request.image_data, dtype=np.uint8)  
    image = preprocess_image(image_data)
    with torch.no_grad():
        output = model(image)
    prediction = torch.argmax(output, dim=1).item()
    confidence = torch.max(torch.nn.functional.log_softmax(output, dim=1)).exp().item()
    return {"prediction": prediction, "confidence": confidence}