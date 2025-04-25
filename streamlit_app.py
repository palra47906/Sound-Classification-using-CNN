import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.models import load_model
import urllib.request
import os

# --- Load Class Mapping from CSV ---
@st.cache_data
def load_class_mapping():
    metadata_url = "https://huggingface.co/palra47906/Sound_Classification_model_using_CNN/resolve/main/UrbanSound8K.csv"  # UPDATE
    df = pd.read_csv(metadata_url)
    return dict(zip(df['classID'], df['class']))

class_mapping = load_class_mapping()

# --- Download & Load Model from Hugging Face or URL ---
@st.cache_resource
def load_trained_model():
    model_url = "https://huggingface.co/palra47906/Sound_Classification_model_using_CNN/resolve/main/Urbansound8K.keras"  # UPDATE
    model_path = "urbansound8k_cnn.keras"
    if not os.path.exists(model_path):
        with st.spinner("Downloading model..."):
            urllib.request.urlretrieve(model_url, model_path)
    return load_model(model_path)

model = load_trained_model()

# --- Feature Extraction ---
def extract_features(file, fixed_length=168):
    try:
        y, sr = librosa.load(file, sr=22050, mono=True)
        n_fft = min(2048, len(y))
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, n_mels=168, fmax=8000)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        if mel_db.shape[1] > fixed_length:
            mel_db = mel_db[:, :fixed_length]
        else:
            mel_db = np.pad(mel_db, ((0, 0), (0, fixed_length - mel_db.shape[1])), mode='constant')

        return mel_db, sr, y
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None, None

# --- Prediction ---
@tf.function(reduce_retracing=True)
def make_prediction(model, input_tensor):
    return model(input_tensor, training=False)

def predict_class(file):
    features, sr, audio = extract_features(file)
    if features is None:
        return None, None, None

    # Normalize and reshape
    features = (features - np.mean(features)) / np.std(features)
    features = np.expand_dims(features, axis=-1)
    features = np.expand_dims(features, axis=0)
    input_tensor = tf.convert_to_tensor(features, dtype=tf.float32)

    prediction = make_prediction(model, input_tensor)
    predicted_class = np.argmax(prediction.numpy())
    label = class_mapping.get(predicted_class, "Unknown")

    return label, sr, audio, features

# --- Streamlit App UI ---
# --- Streamlit App UI with FC Barcelona Logo ---

# App title and instructions (remain the same)
# --- Streamlit App UI with Title + FC Barcelona Logo Inline ---
st.markdown(
    """
    <h1 style='display: flex; align-items: center; gap: 10px;'>
        🎧 Urban Sound Classifier
        <img src='https://upload.wikimedia.org/wikipedia/en/4/47/FC_Barcelona_%28crest%29.svg' width='60'>
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown("Upload a `.wav` file to predict the sound class using a CNN model trained on UrbanSound8K.")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    
    label, sr, audio, features = predict_class(uploaded_file)

    if label:
        st.success(f"✅ Predicted Class: **{label}**")
        colormap = st.selectbox("Select Color Map", ['viridis', 'plasma', 'inferno', 'magma'])

        # Plot Mel Spectrogram
        st.subheader("Mel Spectrogram")
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.specshow(features[0, :, :, 0], sr=sr, x_axis='time', y_axis='mel', cmap=colormap)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Mel Spectrogram - {label}")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        st.pyplot(fig)
