from utils.data_loader import load_dataset

video_paths, labels = load_dataset()

print(video_paths[:5]) #print first 5 video paths
print(labels[:5]) #print first 5 labels