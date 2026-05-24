import torch
import torch.nn as nn #n loss function
import torch.optim as optim #optimizer
from torch.utils.data import DataLoader
from models.resnet_model import DeepFakeModel
from dataset.video_dataset import DeepFakeDataset
from utils.data_loader import load_dataset
from sklearn.model_selection import train_test_split

video_paths, labels = load_dataset() #auto load dataset from data directory

train_paths, test_paths, train_labels, test_labels = train_test_split(video_paths, labels, 
                                                                      test_size = 0.2, random_state = 42,
                                                                      stratify = labels)
#test_size = 0.2 means 20% of the data will be used for testing, random_state = 42 ensures reproducibility,
#stratify = labels ensures that the split maintains the same proportion of real and fake videos in both sets

train_dataset = DeepFakeDataset(train_paths, train_labels)
test_dataset = DeepFakeDataset(test_paths, test_labels)

train_loader = DataLoader(train_dataset, batch_size = 2, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 2, shuffle = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

model = DeepFakeModel().to(device)

criterion = nn.CrossEntropyLoss() #loss function (measure how wrong is AI)

optimizer = optim.Adam(model.parameters(), lr = 0.001) #optimizer

EPOCHS = 5

for epoch in range(EPOCHS):
    
    model.train()
    running_loss = 0.0

    for frames, labels in train_loader:
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad() #clear gradients from previous step
        outputs = model(frames)

        loss= criterion(outputs, labels) #calculate loss
        loss.backward() #backpropagation

        optimizer.step() #update model parameters

        running_loss += loss.item() #accumulate loss for the epoch

    average_loss = running_loss / len(train_loader)

    print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {average_loss:.4f}") 