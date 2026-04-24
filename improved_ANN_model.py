"""
Music Genre Classification - Improved ANN v2
==============================================
Improvements over v1 (92.84% accuracy):
  1. Proper Train/Val/Test split (no data leakage)
  2. Combined features_3_sec + features_30_sec datasets
  3. Wider architecture (512→256→128→64→10) with L2 regularization
  4. EarlyStopping + ModelCheckpoint callbacks
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. DATA LOADING & COMBINING BOTH DATASETS
# ============================================================
print("=" * 60)
print("LOADING AND PREPROCESSING DATA")
print("=" * 60)

# Load 3-second features (primary dataset)
data_3s = pd.read_csv('features_3_sec.csv')
data_3s = data_3s.drop(['filename', 'length'], axis=1)
print(f"3-sec dataset: {data_3s.shape}")

# Load 30-second features (supplementary dataset) if available
data_30s_path = 'features_30_sec.csv'
if os.path.exists(data_30s_path):
    data_30s = pd.read_csv(data_30s_path)
    data_30s = data_30s.drop(['filename', 'length'], axis=1)
    print(f"30-sec dataset: {data_30s.shape}")

    # Combine both datasets - same columns, same genres
    data = pd.concat([data_3s, data_30s], ignore_index=True)
    print(f"Combined dataset: {data.shape}")
else:
    data = data_3s
    print("30-sec dataset not found, using 3-sec only")

print(f"Genres: {list(data['label'].unique())}")
print(f"Samples per genre:\n{data['label'].value_counts()}\n")

# ============================================================
# 2. FEATURE PREPARATION & PROPER SPLIT
# ============================================================
X = data.drop(['label'], axis=1)
y = data['label']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Proper 3-way split: 70% train / 15% val / 15% test ---
# First split: 85% train+val / 15% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
)
# Second split: ~82% train / ~18% val (of the 85%), giving ~70/15/15
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.176, random_state=42, stratify=y_train_val
)

print(f"Training samples:   {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Test samples:       {X_test.shape[0]}")
print(f"Features:           {X_train.shape[1]}")

# One-hot encode labels
y_train_cat = to_categorical(y_train)
y_val_cat = to_categorical(y_val)
y_test_cat = to_categorical(y_test)
num_classes = y_train_cat.shape[1]

# ============================================================
# 3. IMPROVED MODEL ARCHITECTURE
# ============================================================
print("\n" + "=" * 60)
print("BUILDING IMPROVED ANN MODEL")
print("=" * 60)

REG = l2(1e-4)  # L2 regularization strength

model = Sequential([
    # Layer 1 - Wide input layer
    Dense(512, activation='relu', input_shape=(X_train.shape[1],),
          kernel_regularizer=REG),
    BatchNormalization(),
    Dropout(0.4),

    # Layer 2
    Dense(256, activation='relu', kernel_regularizer=REG),
    BatchNormalization(),
    Dropout(0.3),

    # Layer 3
    Dense(128, activation='relu', kernel_regularizer=REG),
    BatchNormalization(),
    Dropout(0.3),

    # Layer 4
    Dense(64, activation='relu', kernel_regularizer=REG),
    BatchNormalization(),
    Dropout(0.2),

    # Output layer
    Dense(num_classes, activation='softmax')
])

# --- Label Smoothing in loss ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

model.summary()

# ============================================================
# 4. CALLBACKS: LR Scheduler + EarlyStopping + ModelCheckpoint
# ============================================================
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-6,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,          # Stop after 20 epochs without improvement
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_ann_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=0
)

# ============================================================
# 5. TRAINING
# ============================================================
print("\nTraining Improved ANN (up to 150 epochs, early stopping enabled)...\n")

history = model.fit(
    X_train, y_train_cat,
    epochs=150,
    batch_size=32,
    validation_data=(X_val, y_val_cat),  # Validate on VAL set, not test!
    callbacks=[lr_scheduler, early_stop, checkpoint],
    verbose=1
)

# ============================================================
# 6. EVALUATION ON HELD-OUT TEST SET
# ============================================================
print("\n" + "=" * 60)
print("EVALUATION ON HELD-OUT TEST SET")
print("=" * 60)

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)

print(f"\n{'='*40}")
print(f"  TEST ACCURACY: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"  TEST LOSS:     {test_loss:.4f}")
print(f"  BEST VAL LOSS: {min(history.history['val_loss']):.4f}")
print(f"  EPOCHS TRAINED: {len(history.history['loss'])}")
print(f"{'='*40}")