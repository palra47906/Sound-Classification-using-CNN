import streamlit as st
import numpy as np
import librosa
from keras.models import load_model
import os
import gdown

# Google Drive direct download link (file ID)
drive_url = "https://drive.google.com/uc?id=1gzMXQawv8R-VelwxYVR4Tn9tWcrvwJGW"
model_path = "urban_sound_model.h5"

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(model_path):
        gdown.download(drive_url, model_path, quiet=False)
    model = load_model(model_path)
    return model

model = download_and_load_model()

def preprocess_audio(file, sample_rate=22050, duration=4, offset=0.5):
    audio, sr = librosa.load(file, sr=sample_rate, duration=duration, offset=offset)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.expand_dims(mfcc, axis=-1)
    mfcc = np.expand_dims(mfcc, axis=0)
    return mfcc

st.title("UrbanSound Classification")

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    features = preprocess_audio(uploaded_file)
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction)
    st.success(f"Predicted Class: {predicted_class}")
