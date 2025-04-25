<h1>🎧 Urban Sound Classifier</h1>
This project is a machine learning application built using Streamlit and TensorFlow that classifies urban sound recordings. 
It uses a Convolutional Neural Network (CNN) trained on the UrbanSound8K dataset to predict various sound categories based on uploaded .wav audio files. 
The model's predictions are displayed along with visualizations like the mel spectrogram and audio waveform.

<h1>📁 Dataset</h1>
The UrbanSound8K dataset consists of 8,732 labeled sound excerpts (<=4s) of urban sounds from 10 classes:
🚗 Car Horn
🎶 Street Music
🐶 Dog Bark 
🔫 Gun Shot
🚧 Jackhammer
👧 Children Playing
🚓 Siren
🛠️ Drilling
🚛 Engine Idling
❄️ Air Conditioner

<h1>🧠 Model Architecture</h1>
The CNN model consists of:

    2 Convolutional Layers + MaxPooling + Dropout

    Flattening layer

    Dense Layer with 128 neurons + Dropout

    Output Layer with Softmax (10 classes)

<h1>🚀 Features</h1>

    MFCC Extraction: Converts audio signals to 2D MFCC features

    Padding: All MFCC matrices are padded to the same time length

    Label Encoding: String labels are encoded to numeric classes

    Train/Test Split: 80/20 stratified split for model evaluation

