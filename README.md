# Fracture Segmentation Inference API

This project demonstrates a simple Flask-based Inference API for fracture segmentation from core images.  
A trained U-Net model predicts a binary mask highlighting fracture zones.

The web app allows users to upload an image and view the predicted mask directly in their browser, without needing to reload the page.

---

## Features

- Upload a core image through a web interface
- Perform inference using a trained U-Net model
- Return and display a predicted fracture mask immediately

---

## Getting Started

### 1. Clone this repository
```bash
git clone https://github.com/ashwinvel2000/Fracture-Segmentation-Api.git
cd fracture-segmentation-api
```
### 2. Create a virtual environment (recommended)
Create a new Conda environment using the provided environment.yml file:
```bash
conda env create -f environment.yml
```
Then activate the environment:
```bash
conda activate fracture-seg-env
```
### 3. Due to GitHub file size limits, the model (model_clahe.pkl) is hosted externally.

Download the model file manually here:

➡️ https://drive.google.com/file/d/1RKfkalXJBcoKwmP5P24SxHrLCSADkp-s/view?usp=drive_link

After downloading, place model_clahe.pkl into the project root directory (same folder as main.py).

### 4. Run the Flask app
Start the API server locally:
```bash
python main.py
```
Then open your browser and navigate to host site.
You can upload a core image and view the predicted fracture mask directly in your browser!

