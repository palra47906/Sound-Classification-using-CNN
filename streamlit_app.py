import os
import urllib.request

import streamlit as st
from tensorflow.keras.models import load_model 

model_url = "https://huggingface.co/palra47906/Sound_Classification_model_using_CNN/resolve/bb833f2ee8f4c14c73f07158b2e7313dad25d7c7/urbansound8k_cnn.h5"
model_path = "urbansound8k_cnn.h5"

# Download model if not already downloaded
if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        urllib.request.urlretrieve(model_url, model_path)

# Load the model
model = load_model(model_path)
