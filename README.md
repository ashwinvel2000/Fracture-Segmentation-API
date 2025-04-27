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
git clone https://github.com/YOUR-USERNAME/fracture-segmentation-api.git
cd fracture-segmentation-api
