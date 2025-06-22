import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Load data and model
df = pd.read_csv("features/audio_features.csv")
model = joblib.load("models/rf_emotion_model.pkl")

# Split again to get same test data
X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Predict
y_pred = model.predict(X_test)

# Confusion Matrix (absolute)
cm = confusion_matrix(y_test, y_pred)
labels = sorted(y.unique())

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Emotion Classification")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
plt.show()

# Save classification report
report = classification_report(y_test, y_pred)
with open("models/classification_report.txt", "w") as f:
    f.write(report)

print("\n✅ Evaluation complete: Confusion matrix image + report saved.")
