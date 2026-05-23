from torch.utils.data import DataLoader
from dataset.video_dataset import DeepFakeDataset

video_paths = ["Portal Tech P90.mp4", "Portal Tech P90.mp4"] #list of video paths
labels = [0, 1] #0 for real, 1 for fake

dataset = DeepFakeDataset(video_paths, labels)

loader = DataLoader(dataset, batch_size = 2, shuffle = True) #batch size of 2, shuffle the data for training

for frames, labels in loader:
    print("Batch Frames Shape:", frames.shape)
    print("Labels Shape:", labels.shape)
    print("Labels:", labels)

    break