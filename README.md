# 🧠 Digit Recognizer

A full‑stack handwritten digit recognition web app using **Flask (Python)** for the backend and **HTML/JS Canvas** for the frontend. Draw a digit in your browser and get instant predictions powered by a PyTorch model.

---

## 📁 Project Structure

digit_recognizer/
├── backend/
│ ├── data/
│ │ └── init.py
│ ├── best_digit_model.pth
│ ├── digit_model.pth
│ ├── final_digit_model.pth
│ ├── server.py
│ ├── train.py
│ └── training_curves.png
├── frontend/
│ └── index.html
├── .gitignore
├── build.sh
├── Procfile
└── requirements.txt


---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/saadK3/Digit_Recognizer.git
cd digit-recognizer

cd backend
pip install -r requirements.txt
python server.py
```
http://127.0.0.1:5000/

## ✏️ How to Use

1. Visit the app in your browser.
2. Use your mouse or finger (touchscreen) to draw a digit on the canvas.
3. Click **Predict**.
4. View the predicted digit and confidence probabilities.

---

## 🧠 How It Works

- **Frontend:**  
  The HTML `<canvas>` element captures the user's drawing. JavaScript converts this into an image and sends it to the Flask backend via a `fetch` POST request.

- **Backend:**  
  Flask receives the image, applies preprocessing using OpenCV (grayscale conversion, thresholding, centering, resizing), and prepares it for model input.

- **Model:**  
  A 3-layer Convolutional Neural Network (CNN) trained on the MNIST dataset processes the image and returns the most likely digit along with a probability distribution.

## ⚙️ Features

- Real-time digit recognition
- Responsive canvas UI for mouse and touch input
- Centering and normalization of digits using OpenCV
- Visual display of prediction confidence using bar charts

---

## 🛠 Troubleshooting

- **404 or frontend not loading?**  
  Ensure the Flask server is running and you're visiting:  
  [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

- **Prediction not working or CORS errors?**  
  Make sure the JavaScript `fetch("/predict")` call is sent to the same origin that serves the frontend (i.e., served by Flask).  
  Avoid using Live Server unless you manually change the fetch URL to point to the Flask backend.

---

## 📬 Contact

For feedback, questions, or contributions, reach out to:

saadahmadkhan1612@gmail.com
