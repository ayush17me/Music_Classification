import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf
from collections import Counter
import io

# --- Page Config ---
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="",
    layout="centered"
)

# --- Load Model & Scaler ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('best_ann_model.keras')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    return model, scaler, le

model, scaler, le = load_model()

# --- Feature Extraction ---
def extract_features(y, sr):
    features = []

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.append(np.mean(chroma))
    features.append(np.var(chroma))

    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))
    features.append(np.var(rms))

    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(spec_cent))
    features.append(np.var(spec_cent))

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(np.mean(spec_bw))
    features.append(np.var(spec_bw))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(rolloff))
    features.append(np.var(rolloff))

    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.var(zcr))

    harmony = librosa.effects.harmonic(y)
    features.append(np.mean(harmony))
    features.append(np.var(harmony))

    perceptr = librosa.effects.percussive(y)
    features.append(np.mean(perceptr))
    features.append(np.var(perceptr))

    tempo = librosa.beat.tempo(y=y, sr=sr)[0]
    features.append(tempo)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features.append(np.mean(mfcc[i]))
        features.append(np.var(mfcc[i]))

    return np.array(features)

# --- Genre Prediction ---
def predict_genre(audio_bytes):
    y, sr = librosa.load(io.BytesIO(audio_bytes), duration=30)
    chunk_samples = 3 * sr
    predictions = []
    confidences = []

    for i in range(10):
        start = i * chunk_samples
        end = start + chunk_samples
        chunk = y[start:end]

        if len(chunk) < chunk_samples:
            break

        features = extract_features(chunk, sr)
        features = features.reshape(1, -1)
        features = scaler.transform(features)

        pred = model.predict(features, verbose=0)
        genre_idx = np.argmax(pred)
        predictions.append(le.inverse_transform([genre_idx])[0])
        confidences.append(np.max(pred))

    if not predictions:
        return None, [], []

    final_genre = Counter(predictions).most_common(1)[0][0]
    return final_genre, predictions, confidences

# --- Genre Colors ---
GENRE_COLORS = {
    'blues': '#1E40AF',
    'classical': '#7C3AED',
    'country': '#D97706',
    'disco': '#EC4899',
    'hiphop': '#EF4444',
    'jazz': '#F59E0B',
    'metal': '#374151',
    'pop': '#10B981',
    'reggae': '#059669',
    'rock': '#DC2626',
}

# --- UI ---
st.markdown("""
<style>
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #4B5563;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .footer-text {
        text-align: center;
        color: #6B7280;
        font-size: 0.85rem;
    }
    @media (prefers-color-scheme: dark) {
        .main-title {
            background: linear-gradient(135deg, #a78bfa 0%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            color: #D1D5DB;
        }
        .footer-text {
            color: #D1D5DB;
        }
    }
    .genre-result {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    .chunk-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.5rem;
        margin: 1rem 0;
    }
    .chunk-item {
        text-align: center;
        padding: 0.5rem;
        border-radius: 0.5rem;
        background: #F3F4F6;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">Music Genre Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a song and it will predict its genre</p>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "ogg", "flac"],
    help="Supports WAV, MP3, OGG, and FLAC formats"
)

if uploaded_file is not None:
    # Show audio player
    st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")

    # Predict
    with st.spinner("Analyzing audio... splitting into chunks and extracting features"):
        audio_bytes = uploaded_file.read()
        final_genre, chunk_predictions, chunk_confidences = predict_genre(audio_bytes)

    if final_genre:
        color = GENRE_COLORS.get(final_genre, '#6366F1')

        # Result
        st.markdown(f"""
        <div class="genre-result" style="background: {color}15; border: 2px solid {color};">
            <span style="color: {color};">{final_genre.upper()}</span>
        </div>
        """, unsafe_allow_html=True)

        # Chunk details
        with st.expander("Chunk-by-Chunk Predictions", expanded=True):
            cols = st.columns(5)
            for i, (pred, conf) in enumerate(zip(chunk_predictions, chunk_confidences)):
                with cols[i % 5]:
                    c = GENRE_COLORS.get(pred, '#6366F1')
                    st.markdown(f"""
                    <div style="text-align:center; padding:0.5rem; border-radius:0.5rem; background:{c}10; border:1px solid {c}30; margin-bottom:0.5rem;">
                        <div style="font-size:0.7rem; color:#9CA3AF;">Chunk {i+1}</div>
                        <div style="font-size:0.8rem; font-weight:600; color:{c};">{pred}</div>
                        <div style="font-size:0.7rem; color:#6B7280;">{conf:.0%}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Vote summary
        vote_counts = Counter(chunk_predictions)
        st.markdown("### Vote Summary")
        for genre, count in vote_counts.most_common():
            color_g = GENRE_COLORS.get(genre, '#6366F1')
            pct = count / len(chunk_predictions)
            st.progress(pct, text=f"{genre}: {count}/{len(chunk_predictions)} votes")

    else:
        st.error("Could not analyze the audio. Make sure the file is at least 3 seconds long.")

# Footer
st.markdown("---")
st.markdown(
    "<p class='footer-text'>"
    "Made by using Artificial Neural Networks"
    "</p>",
    unsafe_allow_html=True
)
