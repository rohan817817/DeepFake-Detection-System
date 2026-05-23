import torch
import torch.nn as nn #n loss function
import torch.optim as optim #optimizer
from torch.utils.data import DataLoader
from models.resnet_model import DeepFakeModel
from dataset.video_dataset import DeepFakeDataset

video_paths = ["Portal Tech P90.mp4", "Portal Tech P90.mp4", "Portal Tech P90.mp4", "Portal Tech P90.mp4"]
               
labels = [0, 1, 0, 1]

dataset = DeepFakeDataset(video_paths, labels)

loader = DataLoader(dataset, batch_size = 2, shuffle = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

model = DeepFakeModel().to(device)

criterion = nn.CrossEntropyLoss() #loss function (measure how wrong is AI)

optimizer = optim.Adam(model.parameters(), lr = 0.001) #optimizer

EPOCHS = 5

for epoch in range(EPOCHS):
    
    model.train()
    running_loss = 0.0

    for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad() #clear gradients from previous step
        outputs = model(frames)

        loss= criterion(outputs, labels) #calculate loss
        loss.backward() #backpropagation

        optimizer.step() #update model parameters

        running_loss += loss.item() #accumulate loss for the epoch

    average_loss = running_loss / len(loader)

    print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {average_loss:.4f}") 