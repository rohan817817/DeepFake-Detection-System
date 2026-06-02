import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset.audio_dataset import AudioDataset
from models.audio_model import AudioClassifier

# -----------------------------
# LOAD DATASET
# -----------------------------

AUDIO_DATASET_PATH = "audio_data"

audio_paths = []
labels = []

real_path = os.path.join(AUDIO_DATASET_PATH, "real")

for file in os.listdir(real_path):
    if file.endswith(".wav"):
        audio_paths.append(os.path.join(real_path, file))
        labels.append(0)

fake_path = os.path.join(AUDIO_DATASET_PATH, "fake")

for file in os.listdir(fake_path):
    if file.endswith(".wav"):
        audio_paths.append(os.path.join(fake_path, file))
        labels.append(1)

# SAME SPLIT AS TRAINING

_, test_paths, _, test_labels = train_test_split(
    audio_paths,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

test_dataset = AudioDataset(
    test_paths,
    test_labels
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

# -----------------------------
# LOAD MODEL
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = AudioClassifier().to(device)

model.load_state_dict(
    torch.load(
        "outputs/checkpoints/best_audio_model.pth",
        map_location=device
    )
)

model.eval()

# -----------------------------
# EVALUATION
# -----------------------------

all_predictions = []
all_labels = []

with torch.no_grad():

    for mfccs, labels in test_loader:

        mfccs = mfccs.to(device)

        outputs = model(mfccs)

        _, predicted = torch.max(outputs, 1)

        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

# -----------------------------
# METRICS
# -----------------------------

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)

print("\nAudio Model Results")
print("-" * 40)

print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall   : {recall*100:.2f}%")
print(f"F1 Score : {f1*100:.2f}%")

# -----------------------------
# CONFUSION MATRIX
# -----------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:")
print(cm)

os.makedirs(
    "outputs/graphs",
    exist_ok=True
)

plt.figure(figsize=(6,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Real","Fake"],
    yticklabels=["Real","Fake"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Audio Model Confusion Matrix")

plt.savefig(
    "outputs/graphs/audio_confusion_matrix.png"
)

plt.show()

print(
    "\nSaved: outputs/graphs/audio_confusion_matrix.png"
)