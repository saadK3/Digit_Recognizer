#!/bin/bash
# Install dependencies
pip install -r requirements.txt

# Run model training if the model doesn't exist
if [ ! -f "best_digit_model.pth" ]; then
    echo "Training model..."
    python backend/train.py
fi