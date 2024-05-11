import streamlit as st
from PIL import Image
from io import BytesIO
from ultralytics import YOLO

st.title('YOLOv8 for geographical feature detection')

# Load the YOLO model
model = YOLO("/Users/rajdipingale/Downloads/weights/best.pt")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
st.subheader("Input image:")
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform object detection on the uploaded image
    detections = model(image)

    # Display the detected objects
    st.subheader("Output image:")
   # Saving images
    for i, image in enumerate(detections):
        image.save(f"image_{i}.png")  # Replace this with the appropriate method for saving your image data
        #image.show(f"image_{i}.png")
        st.image(f"image_{i}.png",caption="detected geographical features", use_column_width=True)