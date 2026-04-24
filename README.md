# Music Genre Classification

A deep learning project that classifies audio tracks into 10 music genres using an Artificial Neural Network (ANN) trained on the GTZAN dataset.

## Genres
Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock

## How It Works

1. **Feature Extraction** - Audio features (MFCCs, chroma, spectral centroid, bandwidth, rolloff, zero crossing rate, tempo, harmonics) are extracted using Librosa
2. **Data Augmentation** - Combined 3-second and 30-second audio feature datasets to create ~11,000 training samples
3. **Training** - ANN with a 512 → 256 → 128 → 64 → 10 architecture, trained with L2 regularization, Batch Normalization, Dropout, and label smoothing
4. **Prediction** - Splits a song into 3-second clips, classifies each, and uses majority voting for the final genre

## Results

- **Test Accuracy**: ~94%
- **Data Split**: 70% train / 15% validation / 15% test (stratified, no data leakage)

## Tech Stack

- Python, TensorFlow/Keras, Scikit-learn, Librosa, Pandas, NumPy

## Project Structure

```
├── improved_ANN_model.ipynb   # Main notebook (training + prediction)
├── improved_ANN_model.py      # Python script version
├── best_ann_model.keras       # Saved model weights
├── features_3_sec.csv         # 3-second audio features
├── features_30_sec.csv        # 30-second audio features
└── test_audio/                # Sample audio files for testing
```

## Quick Start

```bash
# Install dependencies
pip install tensorflow scikit-learn librosa pandas numpy

# Run the notebook
jupyter notebook improved_ANN_model.ipynb

# Or run the script
python improved_ANN_model.py
```

## Usage

To predict the genre of any audio file, run the prediction cell in the notebook:

```python
predict_song("path/to/your/audio.wav")
```

The function loads 30 seconds of audio, splits it into 3-second chunks, extracts features from each, and returns the most common predicted genre.
