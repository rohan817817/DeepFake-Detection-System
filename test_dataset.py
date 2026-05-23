from dataset.video_dataset import DeepFakeDataset

video_paths = ["Portal Tech P90.mp4"]
labels = [0] #0 for real, 1 for fake

dataset = DeepFakeDataset(video_paths, labels)

frames, label = dataset[0]

print("Frames Shape:", frames.shape)
print("Labels:", label)