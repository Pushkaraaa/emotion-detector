#!/bin/bash

# Create and activate virtual environment
echo "Creating Python virtual environment..."
python3 -m venv myenv
source myenv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing required packages..."
pip install streamlit
pip install opencv-python
pip install deepface
pip install numpy
pip install tf-keras  # This was needed for TensorFlow compatibility

echo "Setup complete! To run the application:"
echo "1. Make sure you're in the virtual environment: source myenv/bin/activate"
echo "2. Run: streamlit run app.py" 