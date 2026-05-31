import os 
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
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

train_paths, test_paths, train_labels, test_labels = train_test_split(audio_paths, labels, test_size = 0.2, random__state = 42)

train_dataset = AudioDataset(train_paths, train_labels)
test_dataset = AudioDataset(test_paths, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)

model = AudioClassifier().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)

EPOCHS = 20

for epoch in range(EPOCHS):

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

    accuracy = (100 * correct / total)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {running_loss:.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )

torch.save(model.state_dict(), "outputs/checkpoints/audio_model.pth")

print("Audio model saved.")