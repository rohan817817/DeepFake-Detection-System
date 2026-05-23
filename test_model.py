import torch
from torch.utils.data import DataLoader
from dataset.video_dataset import DeepFakeDataset
from models.resnet_model import DeepFakeModel

video_paths = ["Portal Tech P90.mp4", "Portal Tech P90.mp4"] 
labels = [0, 1]

dataset = DeepFakeDataset(video_paths, labels)
loader = DataLoader(dataset, batch_size = 2, shuffle = True)

device = torch.device("cuda" if torch.cuda.is_available() 
                      else "cpu") #detect if GPU is available, else use CPU
print("Using device:", device)

model = DeepFakeModel()

model = model.to(device) # move model to GPU if available

for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device)
        outputs = model(frames)
        print("Output Shapes:", outputs.shape)
        print(outputs)

        break