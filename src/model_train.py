import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# ======================================================
# Speech Emotion Recognition – Step 4 Script
# Description: <Brief explanation of what this file does>
# ======================================================


# Step 2: Created project folders and organized code scripts

# Load extracted features
df = pd.read_csv("features/audio_features.csv")

# Split into features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc:.2f}")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧮 Confusion Matrix (absolute counts):\n", confusion_matrix(y_test, y_pred))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/rf_emotion_model.pkl")
print("\n✅ Model saved to models/rf_emotion_model.pkl")
