# 🎵 AI Music Genre Intelligence & Classification Platform

An end-to-end Machine Learning and Audio Signal Processing application that automatically classifies music tracks into **10 distinct genres** with **90.74% test accuracy** on the GTZAN benchmark dataset.

---

## 🌟 Key Features

- **90.74% Accuracy Deep Learning Model**: Trained on 9,990 3-second audio segments (`features_3_sec.csv`) using a 58-parameter audio feature vector.
- **Interactive Web App**: Drag-and-drop `.wav`, `.mp3`, `.ogg`, `.m4a` file uploader built with Streamlit.
- **Real-time Audio Visualizations**: Live time-domain waveform and Mel-Frequency Spectrogram rendering.
- **Probability Breakdown**: Real-time genre confidence score and full 10-class probability distribution bars.
- **Audio Parameter Inspection**: Interactive expander table showing all extracted audio features (MFCCs, Chroma, Spectral Centroid, Rolloff, ZCR, RMS, Tempo).

---

## 📁 Repository Structure

```
├── app.py                            # Main Streamlit web application
├── train_model.py                    # Dataset preprocessing & model training script
├── generate_report.py                # Automated Word report generator script
├── music_genre_model.pkl             # Trained model weights (90.74% accuracy)
├── scaler.pkl                        # Serialized StandardScaler
├── label_encoder.pkl                 # Serialized LabelEncoder
├── confusion_matrix.png              # Confusion matrix heatmap plot
├── training_loss.png                 # Model loss / feature importance plot
├── model_metrics.json                # Per-class metrics JSON
├── internship_document_upgraded.docx # Professional Word report
└── requirements.txt                  # Deployment dependencies
```

---

## 🛠️ Local Setup & Running

1. **Clone or navigate to repository**:
   ```bash
   cd ElevateLabs
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App**:
   ```bash
   python -m streamlit run app.py
   ```

---

## 🚀 Deployment Instructions

### Deploy to Streamlit Community Cloud (Free)
1. Push this folder to a public GitHub repository.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/).
3. Click **New app**, select your repository, set Main file path to `app.py`, and click **Deploy**!

### Deploy to Hugging Face Spaces (Free)
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces) with **Streamlit** SDK.
2. Upload `app.py`, `requirements.txt`, `music_genre_model.pkl`, `scaler.pkl`, and `label_encoder.pkl`.
3. Hugging Face will automatically build and launch your app!
