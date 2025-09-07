# CardioBeatAI
CardioBeat AI – Heart Sound Classification
📌 Project Overview

CardioBeat AI is a deep learning–based system for the automatic classification of heart sounds.
Users can upload a .wav, .mp3, .mp4 heartbeat recording, and the system predicts whether it belongs to one of five classes:

Normal (N)
Mitral Regurgitation (MR)
Mitral Stenosis (MS)
Mitral Valve Prolapse (MVP)
Aortic Stenosis (AS)

The system uses MFCC feature extraction + CNN model, outputs a prediction with confidence score, and provides visualizations (waveform & spectrogram).

✨ Features

🎵 Accepts heartbeat recordings in .wav , .mp3, .mp4 format.
🧠 CNN model trained on MFCCs for robust classification.
📊 Outputs prediction + confidence score.
🌐 Flask-based web interface for user uploads.
⚡ Designed for early screening in rural/low-resource settings.

🏗️ Project Architecture

Audio Input → Upload .wav/.mp3/.mp4 file.
Preprocessing → Bandpass filtering, padding/trimming, MFCC extraction.
Deep Learning Model → CNN trained on 5-class dataset.
Prediction → Heart condition label + confidence.

📂 Repository Structure
CardioBeat-AI/


⚙️ Installation

Install dependencies:
🚀 Usage
1.
2. Upload a Heartbeat File

Upload a .wav/.mp3/.mp4 file.
Get prediction + confidence score on the result page.

📊 Model Performance

Test Accuracy: ~99%
Test Loss: ~0.02
Evaluated using K-Fold cross-validation.
Confusion matrix and classification report generated for each fold.

🔮 Future Scope

📱 Mobile app integration.
🎙️ Real-time predictions from microphone input.
🏥 Clinical trials with larger datasets.
☁️ Cloud deployment with database logging.

👨‍💻 Contributors

Hemil Shah (Project Lead)
Niharika Das


