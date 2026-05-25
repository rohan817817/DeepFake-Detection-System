import os
import torch
import torch.nn as nn #n loss function
import torch.optim as optim #optimizer
from torch.utils.data import DataLoader
from models.resnet_model import DeepFakeModel
from dataset.video_dataset import DeepFakeDataset
from utils.data_loader import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

def evaluate(model, loader, device):

    model.eval()
    predictions = []
    actual_labels = []

    with torch.no_grad(): #no need to calculate gradients during evaluation
        for frames, labels in loader:
            frames = frames.to(device)
            labels = labels.to(device)
            outputs = model(frames)
            preds = torch.argmax(outputs, dim = 1)
            predictions.extend(preds.cpu().numpy()) #move predictions to CPU and convert to numpy array
            actual_labels.extend(labels.cpu().numpy()) #move actual labels to CPU and convert to numpy array

    accuracy = accuracy_score(actual_labels, predictions)

    return accuracy

os.makedirs("outputs/checkpoints", exist_ok = True) #create directory for saving model checkpoints if it doesn't exist

EPOCHS = 5

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

for epoch in range(EPOCHS):
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for frames, labels in train_loader:
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad() #clear gradients from previous step
        outputs = model(frames)

        loss= criterion(outputs, labels) #calculate loss
        loss.backward() #backpropagation

        optimizer.step() #update model parameters

        running_loss += loss.item() #accumulate loss for the epoch

        _, predicted = torch.max(outputs, 1) #get predicted labels
        total += labels.size(0) #count total number of samples in the batch
        correct += (predicted == labels).sum().item() #count correct predictions

    average_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    
    train_losses.append(average_loss)
    train_accuracies.append(epoch_accuracy)

    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for frames, labels in test_loader:
            frames = frames.to(device)
            labels = labels.to(device)
            
            outputs = model(frames)

            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item() 

        val_loss = val_running_loss / len(test_loader)

        val_accuracy = 100 * val_correct / val_total

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)



    torch.save(model.state_dict(), "outputs/checkpoints/deepfake_model.pth") #save model checkpoint after each epoch
    print("Model saved") #.pth stores learned weights

    print(f"Epoch [{epoch + 1}/{EPOCHS}] ")
    print(f"Training Loss: {average_loss:.4f}") 
    print(f"Training Accuracy: {epoch_accuracy:.2f}%")

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")