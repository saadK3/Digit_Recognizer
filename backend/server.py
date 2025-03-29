
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
from flask_cors import CORS
import numpy as np
import cv2
import os
from flask import Flask, request, jsonify, send_from_directory
import torch

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# Define the improved model architecture
class ImprovedDigitClassifier(nn.Module):
    def __init__(self):
        super(ImprovedDigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = x.view(-1, 128 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load model
model = ImprovedDigitClassifier()
try:
    model.load_state_dict(torch.load("best_digit_model.pth", map_location=torch.device('cpu')))
except:
    model.load_state_dict(torch.load("final_digit_model.pth", map_location=torch.device('cpu')))
model.eval()

# Standard transformation
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess_image(image):
    """
    Preprocess the image to improve digit recognition:
    - Convert to numpy array for OpenCV operations
    - Invert colors (canvas has white on black, MNIST has black on white)
    - Apply thresholding to eliminate noise
    - Find and center the digit
    """
    # Convert PIL Image to numpy array for OpenCV processing
    img_array = np.array(image)
    
    # If image has RGB channels, convert to grayscale
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Our canvas has white digit on black background, but MNIST has black on white
    # No need to invert if already in the right format
    
    # Apply thresholding to create a binary image
    _, binary = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY)
    
    # Find contours of the digit
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest contour (should be the digit)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box for the digit
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding
        padding = 4
        x_min = max(0, x - padding)
        y_min = max(0, y - padding)
        x_max = min(img_array.shape[1], x + w + padding)
        y_max = min(img_array.shape[0], y + h + padding)
        
        # Crop image to bounding box (with padding)
        if x_max > x_min and y_max > y_min:  # Make sure the crop region is valid
            digit = binary[y_min:y_max, x_min:x_max]
            
            # Calculate aspect ratio
            aspect_ratio = w / h if h > 0 else 1
            
            # Determine size for resizing based on aspect ratio
            if aspect_ratio > 1:
                # Width is greater than height
                target_size = (20, int(20 / aspect_ratio))
            else:
                # Height is greater than width
                target_size = (int(20 * aspect_ratio), 20)
            
            # Resize the digit, maintaining aspect ratio
            try:
                digit_resized = cv2.resize(digit, target_size, interpolation=cv2.INTER_AREA)
                
                # Create a blank 28x28 image (black background)
                digit_centered = np.zeros((28, 28), dtype=np.uint8)
                
                # Calculate position to center the digit
                center_y = 14 - target_size[1] // 2
                center_x = 14 - target_size[0] // 2
                
                # Place the resized digit in the center
                digit_centered[center_y:center_y+target_size[1], center_x:center_x+target_size[0]] = digit_resized
                
                return Image.fromarray(digit_centered)
            except Exception as e:
                print(f"Error in resizing: {e}")
    
    # If contour processing fails, return the original binary image resized to 28x28
    return Image.fromarray(binary).resize((28, 28))

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())).convert("L")
    
    # Apply preprocessing
    preprocessed_image = preprocess_image(image)
    
    # Apply transformations and prepare for model
    image_tensor = transform(preprocessed_image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image_tensor)
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        # Get the highest probability class
        prediction = torch.argmax(probabilities, dim=0).item()
        # Convert probabilities to list and round to 4 decimal places
        prob_list = [round(float(p), 4) for p in probabilities.tolist()]
    
    return jsonify({
        "digit": prediction,
        "probabilities": prob_list
    })

# @app.route("/health", methods=["GET"])
# def health_check():
#     return jsonify({"status": "healthy"})

# if __name__ == "__main__":
#     app.run(debug=True)
    

# Add this at the end of the file:
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

# Add a route to serve the frontend
@app.route('/')
def serve_frontend():
    return send_from_directory('../frontend', 'index.html')