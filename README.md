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

📂 Repository Structure CardioBeat-AI/

│── heart_sound_classifier.py       # Extra script (can be merged or kept for testing)
│── final_heartbeat_model.keras     # Trained model saved in .keras format
│── label_encoder.joblib            # Label encoder for mapping classes
│── requirements.txt                # Python dependencies
│── README.md                       # Project description & usage instructions
│
├── notebooks/                      # Jupyter/Colab notebooks for experimentation
│   └── HeartBeat_Classifier.ipynb
│
├── reports/                        # Project reports & presentations
    ├── Heart Sound Classification.pdf
    └── Heart Sound Classification.pptx



🚀 Usage

1. Upload a Heartbeat File

2. Upload a .wav/.mp3/.mp4 file.

3. Get prediction + confidence score on the result page.

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


Link for the other CNN Model:
https://drive.google.com/file/d/1xV9Czxh5OjyWM4cxSliqCGnpCNd2xZWf/view?usp=sharing

👨‍💻 Contributors

Hemil Shah (Project Lead)

Niharika Das


