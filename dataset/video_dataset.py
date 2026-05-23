from torch.utils.data import Dataset
from utils.frame_extractor import extract_frames
import torch

class DeepFakeDataset(Dataset):
    def __init__(self, video_paths, labels):
        self.video_paths = video_paths
        self.labels = labels

    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = extract_frames(video_path)
        frames = torch.tensor(frames) #numpy array to tensor

        frames = frames.permute(0, 3, 1, 2)
        label = torch.tensor(label).long() #convert label to long tensor

        return frames, label
