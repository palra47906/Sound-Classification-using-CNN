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
import requests
from datetime import datetime
import pytz

# --- Page Config ---
st.set_page_config(page_title="Urban Sound Classifier 🎧", layout="wide")

# --- Get User Location, Date and Time ---
def get_user_location_and_time():
    try:
        response = requests.get("http://ip-api.com/json/")
        if response.status_code == 200:
            data = response.json()
            city = data.get("city", "Unknown City")
            region = data.get("regionName", "Unknown Region")
            country = data.get("country", "Unknown Country")
            timezone = data.get("timezone", "UTC")

            local_zone = pytz.timezone(timezone)
            now = datetime.now(local_zone)

            formatted_time = now.strftime("%I:%M %p")
            formatted_date = now.strftime("%A, %d %B %Y")

            return {
                "location": f"{city}, {region}, {country}",
                "date": formatted_date,
                "time": formatted_time
            }
    except:
        return {
            "location": "Unknown Location",
            "date": datetime.utcnow().strftime("%A, %d %B %Y"),
            "time": datetime.utcnow().strftime("%I:%M %p")
        }

user_info = get_user_location_and_time()

# --- Custom CSS ---
st.markdown(f"""
<style>
    .stButton>button {{
        color: white;
        background-color: #4CAF50;
        font-size: 18px;
        border-radius: 8px;
    }}
    .css-1aumxhk {{
        background-color: #f0f2f6;
    }}
    .weather-box {{
        position: absolute;
        top: 10px;
        right: 20px;
        text-align: right;
        font-size: 14px;
        color: #444;
    }}
</style>
<div class="weather-box">
    📍 {user_info['location']}<br>
    🗕️ {user_info['date']}<br>
    🕒 {user_info['time']}
</div>
""", unsafe_allow_html=True)

# --- Load Class Mapping ---
@st.cache_data
def load_class_mapping():
    metadata_url = "https://huggingface.co/palra47906/Sound_Classification_model_using_CNN/resolve/main/UrbanSound8K.csv"
    df = pd.read_csv(metadata_url)
    return dict(zip(df['classID'], df['class']))

class_mapping = load_class_mapping()

# --- Load Model ---
@st.cache_resource
def load_trained_model():
    model_url = "https://huggingface.co/palra47906/Sound_Classification_model_using_CNN/resolve/main/Urbansound8K_using_CNN.keras"
    model_path = "Urbansound8K_using_CNN.keras"
    if not os.path.exists(model_path):
        with st.spinner("Downloading model..."):
            urllib.request.urlretrieve(model_url, model_path)
    return load_model(model_path)

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

def predict_class(model, file):
    features, sr, audio = extract_features(file)
    if features is None:
        return None, None, None, None

    features = (features - np.mean(features)) / np.std(features)
    features = np.expand_dims(features, axis=-1)
    features = np.expand_dims(features, axis=0)
    input_tensor = tf.convert_to_tensor(features, dtype=tf.float32)

    prediction = make_prediction(model, input_tensor)
    predicted_class = np.argmax(prediction.numpy())
    confidence = np.max(prediction.numpy())
    label = class_mapping.get(predicted_class, "Unknown")

    return label, sr, audio, features, prediction.numpy()

# --- Streamlit App ---
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

# --- Model Selection ---
model_option = st.selectbox("Select Model", ["UrbanSound8K CNN Model"])
model = load_trained_model()

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    label, sr, audio, features, prediction = predict_class(model, uploaded_file)

    if label:
        st.success(f"✅ Predicted Class: **{label}**")

        confidence = np.max(prediction)
        st.metric(label="Prediction Confidence", value=f"{confidence * 100:.2f}%")

        # --- Top 3 Predictions ---
        st.subheader("Top 3 Predicted Classes")
        top3_idx = np.argsort(prediction[0])[-3:][::-1]
        top3_labels = [class_mapping.get(i, "Unknown") for i in top3_idx]
        top3_scores = prediction[0][top3_idx]

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(top3_labels, top3_scores, color='skyblue')
        ax.invert_yaxis()
        ax.set_xlabel('Confidence', fontsize=10)
        ax.set_title('Top 3 Predictions', fontsize=12)
        ax.tick_params(axis='both', labelsize=9)
        st.pyplot(fig)

        # --- Waveform Plot ---
        st.subheader("Waveform")
        fig, ax = plt.subplots(figsize=(8, 2))
        librosa.display.waveshow(audio, sr=sr)
        ax.set_title("Waveform", fontsize=12)
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Amplitude", fontsize=10)
        ax.tick_params(axis='both', labelsize=9)
        st.pyplot(fig)

        # --- Mel Spectrogram ---
        st.subheader("Mel Spectrogram")
        colormap = st.selectbox("Select Color Map", ['viridis', 'plasma', 'inferno', 'magma'], index=0)

        fig, ax = plt.subplots(figsize=(8, 3))
        librosa.display.specshow(features[0, :, :, 0], sr=sr, x_axis='time', y_axis='mel', cmap=colormap)
        plt.colorbar(format='%+2.0f dB')
        ax.set_title(f"Mel Spectrogram - {label}", fontsize=12)
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Frequency (Hz)", fontsize=10)
        ax.tick_params(axis='both', labelsize=9)
        st.pyplot(fig)

# --- Footer ---
st.markdown("---")
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Darlington+Signature&display=swap');
    </style>
    <center>Made with ❤️ by <span style="font-family: 'Darlington Signature', cursive;">Arijit Pal</span></center>
""", unsafe_allow_html=True)

