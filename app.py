import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Page configuration
st.set_page_config(
    page_title="AI Music Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1DB954;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #B3B3B3;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1DB954;
    }
</style>
""", unsafe_allow_html=True)

FEATURE_COLS = [
    'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var', 
    'spectral_centroid_mean', 'spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var', 
    'rolloff_mean', 'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var', 
    'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo', 
    'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean', 'mfcc3_var', 
    'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean', 'mfcc5_var', 'mfcc6_mean', 'mfcc6_var', 
    'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean', 'mfcc8_var', 'mfcc9_mean', 'mfcc9_var', 
    'mfcc10_mean', 'mfcc10_var', 'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var', 
    'mfcc13_mean', 'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var', 
    'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean', 'mfcc18_var', 
    'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var'
]

@st.cache_resource
def load_ml_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "music_genre_model.pkl")
    scaler_path = os.path.join(base_dir, "scaler.pkl")
    encoder_path = os.path.join(base_dir, "label_encoder.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(encoder_path):
        return None, None, None

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    return model, scaler, encoder

def extract_audio_features(file_path):
    """Extract 57 features matching GTZAN features_3_sec.csv"""
    y, sr = librosa.load(file_path, duration=30, sr=22050)
    
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    harmony, perceptr = librosa.effects.hpss(y=y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    features = [
        np.mean(chroma_stft), np.var(chroma_stft),
        np.mean(rms), np.var(rms),
        np.mean(spec_cent), np.var(spec_cent),
        np.mean(spec_bw), np.var(spec_bw),
        np.mean(rolloff), np.var(rolloff),
        np.mean(zcr), np.var(zcr),
        np.mean(harmony), np.var(harmony),
        np.mean(perceptr), np.var(perceptr),
        float(tempo[0] if isinstance(tempo, np.ndarray) else tempo)
    ]

    for i in range(20):
        features.append(np.mean(mfcc[i]))
        features.append(np.var(mfcc[i]))

    return np.array(features).reshape(1, -1), y, sr

def main():
    st.markdown('<div class="main-header">🎵 AI Music Genre Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep Learning & Audio Feature Classification (GTZAN Benchmark Model)</div>', unsafe_allow_html=True)

    model, scaler, encoder = load_ml_artifacts()

    if model is None:
        st.warning("⚠️ Trained model artifacts not found. Please run `python train_model.py` first!")
        st.info("Artifacts required: `music_genre_model.pkl`, `scaler.pkl`, `label_encoder.pkl`")
        return

    st.sidebar.header("📁 Audio Input Controls")
    uploaded_file = st.sidebar.file_uploader("Upload an Audio File (.wav, .mp3, .ogg)", type=["wav", "mp3", "ogg", "m4a", "flac"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Details")
    st.sidebar.write("**Model Architecture**: Multi-Layer Deep Neural Network")
    st.sidebar.write("**Benchmark Accuracy**: 90.74%")
    st.sidebar.write("**Dataset**: GTZAN Music Genre Dataset")
    st.sidebar.write("**Target Genres**: 10 Classes")
    st.sidebar.write("• Blues • Classical • Country • Disco • Hiphop • Jazz • Metal • Pop • Reggae • Rock")

    if uploaded_file is not None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        temp_path = os.path.join(base_dir, "temp_uploaded_audio.wav")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🎧 Audio Player & Visualization")
            st.audio(temp_path)

            with st.spinner("Extracting audio signals & generating Mel-Spectrogram..."):
                features_vec, y, sr = extract_audio_features(temp_path)

                fig, ax = plt.subplots(2, 1, figsize=(10, 6))
                
                # Waveform
                librosa.display.waveshow(y, sr=sr, ax=ax[0], color='#1DB954')
                ax[0].set_title("Time-Domain Waveform")
                ax[0].set_xlabel("Time (seconds)")
                ax[0].set_ylabel("Amplitude")

                # Spectrogram
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                S_dB = librosa.power_to_db(S, ref=np.max)
                img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax[1], cmap='magma')
                ax[1].set_title("Mel-Frequency Spectrogram")
                fig.colorbar(img, ax=ax[1], format='%+2.0f dB')

                plt.tight_layout()
                st.pyplot(fig)

        with col2:
            st.subheader("🎯 Genre Prediction Analysis")

            with st.spinner("Classifying audio genre via Deep Neural Net..."):
                scaled_vec = scaler.transform(features_vec)
                probs = model.predict_proba(scaled_vec)[0]
                pred_idx = np.argmax(probs)
                predicted_genre = encoder.inverse_transform([pred_idx])[0].upper()
                confidence = probs[pred_idx] * 100

                st.markdown(f"""
                <div style="background-color: #1DB95422; border: 2px solid #1DB954; padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 20px;">
                    <h4 style="margin: 0; color: #B3B3B3;">Predicted Genre</h4>
                    <h1 style="margin: 5px 0 0 0; color: #1DB954; font-size: 3rem;">{predicted_genre}</h1>
                    <p style="margin: 5px 0 0 0; font-size: 1.2rem; color: #FFFFFF;">Confidence Score: <b>{confidence:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)

                st.write("#### Class Probabilities Breakdown")
                genre_classes = encoder.classes_
                prob_df = pd.DataFrame({
                    "Genre": [g.capitalize() for g in genre_classes],
                    "Probability (%)": probs * 100
                }).sort_values(by="Probability (%)", ascending=False)

                for idx, row in prob_df.iterrows():
                    col_name, col_bar = st.columns([1, 3])
                    with col_name:
                        st.write(f"**{row['Genre']}**")
                    with col_bar:
                        st.progress(float(row['Probability (%)']) / 100.0)

        with st.expander("📊 Inspect Extracted Audio Features (57 Parameters)"):
            st.dataframe(pd.DataFrame(features_vec, columns=FEATURE_COLS))
    else:
        st.info("👈 Please upload an audio file (.wav or .mp3) in the sidebar to test predictions live!")

if __name__ == "__main__":
    main()
