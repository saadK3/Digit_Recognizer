import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
from flask_cors import CORS
import numpy as np
import cv2
import os
from flask import Flask, request, jsonify, send_from_directory, send_file
import torch

app = Flask(__name__, static_folder="../frontend")
CORS(app)  # Enable CORS for all origins

# Improved model
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

# Image transformation
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess_image(image):
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        padding = 4
        x_min = max(0, x - padding)
        y_min = max(0, y - padding)
        x_max = min(img_array.shape[1], x + w + padding)
        y_max = min(img_array.shape[0], y + h + padding)
        if x_max > x_min and y_max > y_min:
            digit = binary[y_min:y_max, x_min:x_max]
            aspect_ratio = w / h if h > 0 else 1
            if aspect_ratio > 1:
                target_size = (20, int(20 / aspect_ratio))
            else:
                target_size = (int(20 * aspect_ratio), 20)
            try:
                digit_resized = cv2.resize(digit, target_size, interpolation=cv2.INTER_AREA)
                digit_centered = np.zeros((28, 28), dtype=np.uint8)
                center_y = 14 - target_size[1] // 2
                center_x = 14 - target_size[0] // 2
                digit_centered[center_y:center_y+target_size[1], center_x:center_x+target_size[0]] = digit_resized
                return Image.fromarray(digit_centered)
            except Exception as e:
                print(f"Error in resizing: {e}")
    return Image.fromarray(binary).resize((28, 28))

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())).convert("L")
    preprocessed_image = preprocess_image(image)
    image_tensor = transform(preprocessed_image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        prediction = torch.argmax(probabilities, dim=0).item()
        prob_list = [round(float(p), 4) for p in probabilities.tolist()]
    return jsonify({"digit": prediction, "probabilities": prob_list})

# Serve index.html
@app.route("/")
def serve_index():
    return send_file(os.path.join(app.static_folder, "index.html"))

# Run app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
