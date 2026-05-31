import os
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from torch.utils.data import DataLoader

from dataset.audio_dataset import AudioDataset
from models.audio_model import AudioClassifier

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

print("Total Audio Files:", len(audio_paths))
print("Real Audio:", labels.count(0))
print("Fake Audio:", labels.count(1))

train_paths, test_paths, train_labels, test_labels = train_test_split(
    audio_paths,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

train_dataset = AudioDataset(train_paths, train_labels)
test_dataset = AudioDataset(test_paths, test_labels)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

model = AudioClassifier().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)

EPOCHS = 20

best_accuracy = 0

os.makedirs(
    "outputs/checkpoints",
    exist_ok=True
)

for epoch in range(EPOCHS):

    # Training
    model.train()

    running_loss = 0

    correct = 0
    total = 0

    for mfccs, labels in train_loader:

        mfccs = mfccs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(mfccs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_accuracy = (100 * correct / total)

    # Validation
    model.eval()

    val_loss = 0

    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for mfccs, labels in test_loader:

            mfccs = mfccs.to(device)
            labels = labels.to(device)

            outputs = model(mfccs)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())

            all_labels.extend(labels.cpu().numpy())

    val_accuracy = (100 * correct / total)

    precision = precision_score(all_labels, all_predictions)

    recall = recall_score(all_labels, all_predictions)

    f1 = f1_score(all_labels, all_predictions)

    cm = confusion_matrix(all_labels, all_predictions) 
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

    print(f"Training Loss: {running_loss:.4f}")

    print(f"Training Accuracy: {train_accuracy:.2f}%")

    print(f"Validation Loss: {val_loss:.4f}")

    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    print(f"Precision: {precision:.4f}")

    print(f"Recall: {recall:.4f}")

    print(f"F1 Score: {f1:.4f}")

    print("Confusion Matrix:")
    print(cm)

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "outputs/checkpoints/best_audio_model.pth"
        )

        print("Best audio model saved.")

torch.save(
    model.state_dict(),
    "outputs/checkpoints/audio_model.pth"
)

print("Audio model saved.")