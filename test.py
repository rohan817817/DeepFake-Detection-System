from utils.data_loader import load_dataset

video_paths, labels = load_dataset("data")

print("Real:", labels.count(0))
print("Fake:", labels.count(1))