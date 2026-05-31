import os
from utils.audio_extractor import extract_audio

VIDEO_DATASET = "data"
AUDIO_DATASET = "audio_data"

os.makedirs(f"{AUDIO_DATASET}/real", exist_ok=True)
os.makedirs(f"{AUDIO_DATASET}/fake", exist_ok=True)

for label in ["real", "fake"]:

    video_folder = os.path.join(VIDEO_DATASET, label)
    audio_folder = os.path.join(AUDIO_DATASET, label)

    for file in os.listdir(video_folder):

        if not file.endswith(".mp4"):
            continue

        video_path = os.path.join(video_folder, file)

        audio_path = os.path.join(
            audio_folder,
            file.replace(".mp4", ".wav")
        )

        # Skip already converted files
        if os.path.exists(audio_path):
            continue

        print("Processing:", video_path)

        try:
            extract_audio(video_path, audio_path)
            print("Converted:", file)

        except Exception as e:
            print("Failed:", file)
            print(e)