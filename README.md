# Face Detection App

A Streamlit web application for real-time face detection using OpenCV's DNN module.

## Features

- Upload images for face detection
- Adjustable confidence threshold
- Side-by-side view of original and processed images
- Download processed images
- Modern and clean user interface

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Download the model files:
- `deploy.prototxt` (included in repo)
- `res10_300x300_ssd_iter_140000_fp16.caffemodel` (download from OpenCV's repository)

## Running Locally

```bash
streamlit run app.py
```

## Deployment

This app is ready to be deployed on Streamlit Cloud:

1. Push the code to GitHub
2. Connect your GitHub repository to Streamlit Cloud
3. Deploy!
