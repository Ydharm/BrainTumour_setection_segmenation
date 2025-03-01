import streamlit as st
from PIL import Image
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from inference import perform_inference

st.title("Brain Tumor Detection and segmentation")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Perform Operation"):
        temp_file_path = "temp_image.jpg"
        image.save(temp_file_path)

        result = perform_inference(temp_file_path)

        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader("Detection Result:")
            st.write(np.array(result["detection"]))
        import os
        os.remove(temp_file_path)