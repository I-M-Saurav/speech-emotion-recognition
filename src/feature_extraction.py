import os
import librosa
import numpy as np
import pandas as pd

# Folders
AUDIO_DIR = "audio"
OUTPUT_CSV = "features/audio_features.csv"

# Mapping emotion codes to labels
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}
# update
# Function to extract features from a single file
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)

    # MFCC (40 values)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    # Delta of MFCC (40 values)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta_mfcc.T, axis=0)

    # Spectral features (4 values)
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T)
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr).T)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T)

    # Combine all features (total 85 values)
    features = np.hstack([mfcc_mean, delta_mean, spec_centroid, spec_bw, rolloff, zcr])
    return features


# Loop over all .wav files
data = []

for root, _, files in os.walk(AUDIO_DIR):
    for file in files:
        if file.endswith(".wav"):
            path = os.path.join(root, file)
            emotion_code = file.split("-")[2]
            if emotion_code in emotion_map:
                label = emotion_map[emotion_code]
                try:
                    features = extract_features(path)
                    data.append(np.append(features, label))
                except Exception as e:
                    print(f"Error in {file}: {e}")

# Create column names
columns = [f"mfcc_{i+1}" for i in range(40)] + \
          [f"delta_{i+1}" for i in range(40)] + \
          ["spec_centroid", "spec_bw", "rolloff", "zcr", "label"]

# Create DataFrame and save
df = pd.DataFrame(data, columns=columns)
os.makedirs("features", exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
print("✅ Feature extraction complete! CSV saved at:", OUTPUT_CSV)
