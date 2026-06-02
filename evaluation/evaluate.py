import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from torch.utils.data import DataLoader
from models.resnet_model import DeepFakeModel
from dataset.video_dataset import DeepFakeDataset
from utils.data_loader import load_dataset
from sklearn.model_selection import train_test_split

dataset_path = "data"

video_paths, labels = load_dataset(dataset_path)

_, test_paths, _, test_labels = train_test_split(
    video_paths,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

test_dataset = DeepFakeDataset(
    test_paths,
    test_labels
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = DeepFakeModel().to(device)

model.load_state_dict(
    torch.load(
        "outputs/checkpoints/best_model.pth",
        map_location=device
    )
)

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for frames, labels in test_loader:

        frames = frames.to(device)

        outputs = model(frames)

        _, predicted = torch.max(outputs, 1)

        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

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

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print(cm)

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
plt.title("Confusion Matrix")

plt.savefig(
    "outputs/graphs/best_model_confusion_matrix.png"
)

plt.show()