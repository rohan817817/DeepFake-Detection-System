import torch
from torch.utils.data import Dataset
from audio.feature_extractor import extract_mfcc

class AudioDataset(Dataset):
    def __init__(self, audio_paths, labels):
        self.audio_paths = audio_paths
        self.labels = labels

    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        mfcc = extract_mfcc(self.audio_paths[idx])
        mfcc = torch.tensor(mfcc, dtype = torch.float32)
        label = torch.tensor(self.labels[idx], dtype = torch.long)

        return mfcc, label