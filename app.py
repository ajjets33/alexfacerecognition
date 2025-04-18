import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Face Detection App",
    page_icon="👤",
    layout="wide"
)

# Load the pre-trained face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image):
    # Convert PIL Image to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return image, len(faces)

def main():
    st.title("Face Detection App")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        
        # Process image
        with st.spinner('Detecting faces...'):
            # Convert PIL Image to RGB (OpenCV uses BGR)
            image = image.convert('RGB')
            
            # Detect faces
            result_image, face_count = detect_faces(image)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Detected Faces")
                st.image(result_image, use_column_width=True)
            
            # Display detection info
            st.success(f"Found {face_count} faces in the image!")

if __name__ == "__main__":
    main()

        st.markdown(get_image_download_link(out_image, "face_output.jpg", 'Download Output Image'),
                   unsafe_allow_html=True)
