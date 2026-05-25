#convert video and label to tensor format for training

from torch.utils.data import Dataset
from utils.frame_extractor import extract_frames
import torch
from torchvision import transforms
from PIL import Image

class DeepFakeDataset(Dataset):
    def __init__(self, video_paths, labels):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.RandomHorizontalFlip(p = 0.5), #50% chance to flip the image horizontally for data augmentation
                                            transforms.RandomRotation(10),
                                            transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                mean = [0.485, 0.456, 0.406],
                                                std = [0.229, 0.224, 0.225]
                                            )]) #data augmentation and normalization for better generalization of the model

    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = extract_frames(video_path)
        processed_frames = []

        for frame in frames:
            frame = (frame *255).astype("uint8") #convert float32(0.0 -> 1.0) to uint8(0 -> 255) for PIL compatibility
            frame = Image.fromarray(frame) #convert numpy array to PIL image
            frame = self.transform(frame) #apply transformations to the frame
            processed_frames.append(frame)
        
        frames = torch.stack(processed_frames)
        label = torch.tensor(label). long() #convert label to tensor format for training  

        return frames, label
