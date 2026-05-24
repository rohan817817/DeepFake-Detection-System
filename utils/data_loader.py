import glob #used to find all the path of the files in a directory
import os

def load_dataset(data_dir = "data"): 
    video_paths = []
    labels = []

    real_dir = os.path.join(data_dir, "real")
    fake_dir = os.path.join(data_dir, "fake")

    real_videos = glob.glob(os.path.join(real_dir, "*.mp4")) #find all the mp4 files in the real directory
    fake_videos = glob.glob(os.path.join(fake_dir, "*.mp4")) #find all the mp4 files in the fake directory

    for video in real_videos:
        video_paths.append(video)
        labels.append(1) #label 1 for real videos

    for video in fake_videos:
        video_paths.append(video)
        labels.append(0) #label 0 for fake videos

    print(f"Loaded {len(real_videos)} real videos.")
    print(f"Loaded {len(fake_videos)} fake videos.")
    print(f"Total videos loaded: {len(video_paths)}")

    return video_paths, labels

