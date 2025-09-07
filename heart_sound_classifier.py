# -*- coding: utf-8 -*-
"""
Heart Sound Classification with Deep Learning and Streamlit Deployment

This script handles the entire machine learning pipeline for classifying heart sounds:
1. Data loading and preprocessing (assuming data is unzipped).
2. Feature extraction using Mel-spectrograms.
3. Building and training an advanced Convolutional Neural Network (CNN).
4. Saving the trained model.
5. Deploying the model with a user-friendly Streamlit UI for real-time prediction.

Dependencies:
- numpy
- pandas
- librosa
- scikit-learn
- tensorflow
- streamlit
- zipfile (standard library)
- matplotlib (for plotting)
- pydub (for mp4/mp3 to wav conversion)
- ffmpeg (required by pydub, must be installed separately)

To install the required libraries, run the following command in your terminal:
pip install numpy pandas librosa scikit-learn tensorflow streamlit matplotlib pydub

Note: You must also have ffmpeg installed and available in your system's PATH.
"""

import os
import zipfile
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import streamlit as st
import warnings

# Attempt to import pydub and handle potential import errors
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    st.error("The `pydub` library is not installed. MP4/MP3 conversion will not work. Please install it using `pip install pydub`.")
    PYDUB_AVAILABLE = False
except Exception as e:
    st.error(f"An error occurred during `pydub` import: {e}. Please ensure you have `ffmpeg` installed and in your system's PATH. Reinstalling `pydub` and `ffmpeg` may help.")
    PYDUB_AVAILABLE = False
    
# Suppress all warnings
warnings.filterwarnings("ignore")

# --- Configuration and Constants ---
DATA_ZIP_FILE = 'RP1Heart Data final.zip'
DATA_DIR = 'RP Heart Data/Heart_data'
# The native Keras format (.keras) is recommended over the legacy HDF5 format (.h5)
MODEL_FILENAMES = ['heart_sound_model.keras', 'final_heartbeat_model.keras']
IMG_WIDTH, IMG_HEIGHT = 128, 128
SAMPLE_RATE = 22050
DURATION = 5  # seconds
MAX_SHAPE = (IMG_WIDTH, IMG_HEIGHT)
CLASSES = ['AS (Mitral Valve prolapse)', 'MR (Mitral Regurgitation)', 'MS (Mitral Stenosis)', 'MVP (Mitral Valve Prolapse)', 'N (Normal)']

# --- Helper Functions ---

def extract_zip_file():
    """Extracts the zip file if it exists and the data directory is not present."""
    if not os.path.isdir(os.path.join('.', 'RP Heart Data')):
        st.info(f"Extracting data from {DATA_ZIP_FILE}...")
        try:
            with zipfile.ZipFile(DATA_ZIP_FILE, 'r') as zip_ref:
                zip_ref.extractall('.')
            st.success("Data extracted successfully!")
        except FileNotFoundError:
            st.error(f"Error: {DATA_ZIP_FILE} not found. Please ensure it's in the same directory as this script.")
            st.stop()
    else:
        st.success("Data directory already exists. Skipping extraction.")

def extract_features(audio, sr):
    """
    Extracts Mel-spectrogram features from audio data.
    
    Args:
        audio (np.array): Audio time series.
        sr (int): Sampling rate of the audio.
        
    Returns:
        np.array: A 2D numpy array of the Mel-spectrogram, or None if an error occurs.
    """
    try:
        # Trim silence from the beginning and end
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Pad or truncate the audio to a fixed length
        audio_length = SAMPLE_RATE * DURATION
        if len(audio) > audio_length:
            audio = audio[:audio_length]
        else:
            padding = audio_length - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
            
        # Compute the Mel-spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=IMG_WIDTH)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Pad or truncate the spectrogram to a fixed shape (IMG_WIDTH, IMG_HEIGHT)
        current_frames = mel_spectrogram_db.shape[1]
        if current_frames > IMG_HEIGHT:
            mel_spectrogram_resized = mel_spectrogram_db[:, :IMG_HEIGHT]
        else:
            padding_frames = IMG_HEIGHT - current_frames
            mel_spectrogram_resized = np.pad(mel_spectrogram_db, ((0, 0), (0, padding_frames)), 'constant')

        return mel_spectrogram_resized

    except Exception as e:
        print(f"Error processing audio data: {e}")
        return None

def load_dataset(data_dir):
    """
    Loads audio files and their labels, extracting features.
    
    Args:
        data_dir (str): Path to the directory containing subfolders of audio files.
        
    Returns:
        tuple: A tuple containing lists of features and labels.
    """
    features = []
    labels = []
    
    total_files = sum([len(files) for r, d, files in os.walk(data_dir)])
    processed_count = 0
    
    for category in CLASSES:
        category_path = os.path.join(data_dir, category)
        if not os.path.isdir(category_path):
            st.warning(f"Category folder '{category}' not found at {category_path}. Skipping.")
            continue
            
        for file_name in os.listdir(category_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(category_path, file_name)
                # Load audio for feature extraction
                try:
                    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
                    feature = extract_features(audio, sr)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    feature = None
                
                if feature is not None:
                    features.append(feature)
                    labels.append(category)
                
                processed_count += 1
                progress = processed_count / total_files
                st.progress(progress, text=f"Processing files: {processed_count}/{total_files}")
    
    return np.array(features), np.array(labels)

def build_cnn_model(input_shape, num_classes):
    """
    Builds and returns a deep Convolutional Neural Network (CNN) model.
    
    Args:
        input_shape (tuple): The shape of the input data (height, width, channels).
        num_classes (int): The number of output classes.
        
    Returns:
        tensorflow.keras.models.Sequential: The compiled Keras model.
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Flatten and Dense layers for classification
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# --- Main Execution Logic ---
def run_training_and_saving():
    """
    Performs data loading, training, and model saving.
    """
    st.header("Training Deep Learning Model")
    st.write("This process will take some time. Please wait...")

    # Load data
    with st.spinner("Loading and extracting features from audio files..."):
        features, labels = load_dataset(os.path.join('.', DATA_DIR))
    
    if len(features) == 0:
        st.error("No features were extracted. Please check your data directory and file paths.")
        st.stop()
        
    features = features[..., np.newaxis] # Add channel dimension

    # Preprocess labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    y_one_hot = to_categorical(y_encoded)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, y_one_hot, test_size=0.2, random_state=42, stratify=y_one_hot
    )
    
    # Build and train the model
    st.write("Building and training the CNN model...")
    model = build_cnn_model(X_train.shape[1:], len(CLASSES))
    
    # Define callbacks for training to prevent overfitting
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(MODEL_FILENAMES[0], monitor='val_loss', save_best_only=True, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )
    
    st.success("Model training complete!")

    # Plotting the accuracy and loss curves for visualization
    history_df = pd.DataFrame(history.history)
    
    st.header("Training History")
    st.write("Visualizing the model's performance over epochs.")

    # Plot Accuracy
    st.subheader("Accuracy")
    fig_acc = plt.figure()
    plt.plot(history_df['accuracy'], label='Training Accuracy')
    plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    st.pyplot(fig_acc)

    # Plot Loss
    st.subheader("Loss")
    fig_loss = plt.figure()
    plt.plot(history_df['loss'], label='Training Loss')
    plt.plot(history_df['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot(fig_loss)
    
    st.info("The graphs clearly show the point where the validation loss stops improving. The `EarlyStopping` callback has prevented the model from further overfitting.")
    
    st.write("Training accuracy:", history.history['accuracy'][-1])
    st.write("Validation accuracy:", history.history['val_loss'][-1])
    
    # Use the correct function to rerun the app
    st.rerun()

def display_spectrogram(y, sr):
    """
    Plots and displays the Mel-spectrogram of audio data.
    
    Args:
        y (np.array): Audio time series.
        sr (int): Sampling rate of the audio.
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=IMG_WIDTH)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 3))
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='viridis')
    ax.set_title('Spectrogram')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mel Frequency')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    st.pyplot(fig)


# --- Streamlit UI and Prediction Logic ---
def run_streamlit_app(model_file):
    """
    Runs the Streamlit UI for heart sound classification.
    """
    st.write("### Upload a WAV, MP3 or MP4 file to classify its heart sound.")

    # Define the display names for the models
    DISPLAY_NAMES = ["Model by Hemil Shah 🔥", "Model by Niharika Das 🙌"]

    # Create a simple two-column layout for the main content
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Model Selection")
        # Find the index of the default model file to set the default selection
        default_index = MODEL_FILENAMES.index(model_file)
        
        selected_display_name = st.selectbox(
            'Choose a pre-trained model:',
            options=DISPLAY_NAMES,
            index=default_index
        )
        
        st.subheader("Upload Audio")
        # Check if pydub is available to allow mp4/mp3 uploads
        allowed_types = ["wav"]
        if PYDUB_AVAILABLE:
            allowed_types.extend(["mp3", "mp4"])
            
        uploaded_file = st.file_uploader("Choose an audio/video file...", type=allowed_types)
        st.info("💡 Tip: The model is trained on sounds that are approximately 5 seconds long.")
        
        converted_file_path = None
        
        if uploaded_file is not None:
            y = None
            sr = None
            try:
                # Load audio data without a fixed duration to display the full length
                y, sr = librosa.load(uploaded_file, sr=SAMPLE_RATE)
            except Exception as e:
                # Fallback for pydub conversion for mp3/mp4
                if PYDUB_AVAILABLE and uploaded_file.name.split('.')[-1].lower() in ['mp3', 'mp4']:
                    try:
                        with st.spinner("Converting file to WAV..."):
                            temp_path = os.path.join(".", uploaded_file.name)
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            audio = AudioSegment.from_file(temp_path)
                            converted_file_path = os.path.join(".", "converted_audio.wav")
                            audio.export(converted_file_path, format="wav")
                            os.remove(temp_path)
                        
                        # Load converted file without a fixed duration
                        y, sr = librosa.load(converted_file_path, sr=SAMPLE_RATE)
                    except FileNotFoundError:
                        st.error("Error: FFmpeg was not found. Please ensure it is installed and added to your system's PATH. You can download it from [ffmpeg.org](https://ffmpeg.org/download.html).")
                        st.stop()
                    except Exception as e:
                        st.error(f"An error occurred during file conversion: {e}.")
                        st.stop()
                else:
                    st.error(f"Error loading audio file: {e}")
                    st.stop()
            
            # Display the audio player
            st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1].lower()}')

            # New code to display the Mel-Spectrogram
            st.subheader("Mel-Spectrogram")
            display_spectrogram(y, sr)

    with col2:
        if uploaded_file is not None:
            # New code to display the sound wave
            st.subheader("Sound Wave")
            
            # Create a matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 3))
            
            # Plot the waveform
            librosa.display.waveshow(y, sr=sr, ax=ax, color='purple')
            ax.set_title('Audio Waveform')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            
            # Display the plot in Streamlit
            st.pyplot(fig)

            st.subheader("Prediction Result")
            
            # Extract features and predict
            with st.spinner("Analyzing the audio file..."):
                # Get the actual filename from the selected display name
                selected_model_index = DISPLAY_NAMES.index(selected_display_name)
                selected_model_filename = MODEL_FILENAMES[selected_model_index]

                # Load the selected model
                try:
                    model = load_model(selected_model_filename)
                except Exception as e:
                    st.error(f"Error loading model '{selected_model_filename}': {e}. Please check if the model file exists.")
                    st.stop()

                # Process the entire audio by segmenting it into fixed-duration chunks
                predictions = []
                audio_duration = librosa.get_duration(y=y, sr=sr)
                num_chunks = int(np.ceil(audio_duration / DURATION))

                if num_chunks == 0:
                    st.error("Audio is too short to process.")
                else:
                    for i in range(num_chunks):
                        start_time = i * DURATION
                        end_time = min((i + 1) * DURATION, audio_duration)
                        
                        # Get the current chunk and pad if necessary
                        chunk_y = y[int(start_time * sr):int(end_time * sr)]
                        
                        # The `extract_features` function already handles padding to the model's expected length
                        feature = extract_features(chunk_y, sr)
                        
                        if feature is not None:
                            # Reshape for the model and predict
                            feature_reshaped = feature.reshape(1, IMG_WIDTH, IMG_HEIGHT, 1)
                            chunk_prediction = model.predict(feature_reshaped, verbose=0)
                            predictions.append(chunk_prediction)
                    
                    if not predictions:
                        st.error("Could not process any segment of the audio file.")
                    else:
                        # Aggregate predictions by taking the mean of probabilities
                        mean_prediction = np.mean(predictions, axis=0)
                        predicted_class_index = np.argmax(mean_prediction)
                        confidence = np.max(mean_prediction) * 100
                        
                        # Get the class name
                        predicted_class_name = CLASSES[predicted_class_index]
                        
                        st.balloons()
                        st.success(f"Prediction: **{predicted_class_name}** with **{confidence:.2f}%** confidence.")
                        
                        # Display the prediction probabilities in a bar chart
                        st.write("---")
                        st.subheader("Prediction Probabilities")
                        prediction_df = pd.DataFrame(mean_prediction.T, index=CLASSES, columns=['Probability'])
                        st.bar_chart(prediction_df)
            
            # Clean up the temporary files
            if converted_file_path and os.path.exists(converted_file_path):
                os.remove(converted_file_path)
    
# --- Main entry point for the script ---
if __name__ == '__main__':
    st.set_page_config(
        page_title="Heart Sound Classifier App 💓",
        page_icon="✨",
        layout="wide"
    )
    st.title("✨ Heart Sound Classifier")
    
    st.write("This app uses a trained deep learning model to classify heart sounds from WAV, MP3 or MP4 files.")
    
    # Check for the existence of both model files
    if not os.path.isfile(MODEL_FILENAMES[0]):
        st.info(f"Model file `{MODEL_FILENAMES[0]}` not found. The app will now train the model from scratch.")
        with st.expander("Show Training Details"):
            extract_zip_file()
            st.progress(0, text="Starting data loading...")
            run_training_and_saving()
    elif not os.path.isfile(MODEL_FILENAMES[1]):
        st.warning(f"Model file `{MODEL_FILENAMES[1]}` not found. Please ensure it is in the same directory.")
        st.stop()
    else:
        st.info(f"Model files found. You can now select a model for classification.")
        run_streamlit_app(MODEL_FILENAMES[0])