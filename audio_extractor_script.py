import os
from utils.audio_extractor import extract_audio

SOURCE_ROOT = r"C:\Users\HI\Downloads\archive-2\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"

REAL_AUDIO_FOLDERS = [
    "RealVideo-RealAudio",
    "FakeVideo-RealAudio"
]

FAKE_AUDIO_FOLDERS = [
    "RealVideo-FakeAudio",
    "FakeVideo-FakeAudio"
]

REAL_OUTPUT = "audio_data/real"
FAKE_OUTPUT = "audio_data/fake"

os.makedirs(REAL_OUTPUT, exist_ok=True)
os.makedirs(FAKE_OUTPUT, exist_ok=True)

print("\nGenerating REAL audio dataset...\n")

for folder in REAL_AUDIO_FOLDERS:

    folder_path = os.path.join(
        SOURCE_ROOT,
        folder
    )

    for root, dirs, files in os.walk(folder_path):

        for file in files:

            if not file.endswith(".mp4"):
                continue

            audio_filename = (
                f"{folder}_{file.replace('.mp4', '.wav')}"
            )

            audio_path = os.path.join(
                REAL_OUTPUT,
                audio_filename
            )

            # Skip already extracted files
            if os.path.exists(audio_path):
                continue

            video_path = os.path.join(
                root,
                file
            )

            try:

                extract_audio(
                    video_path,
                    audio_path
                )

                print(
                    f"[REAL] {audio_filename}"
                )

            except Exception as e:

                print(
                    f"[FAILED] {file}"
                )

                print(e)

                continue

print("\nGenerating FAKE audio dataset...\n")

for folder in FAKE_AUDIO_FOLDERS:

    folder_path = os.path.join(
        SOURCE_ROOT,
        folder
    )

    for root, dirs, files in os.walk(folder_path):

        for file in files:

            if not file.endswith(".mp4"):
                continue

            audio_filename = (
                f"{folder}_{file.replace('.mp4', '.wav')}"
            )

            audio_path = os.path.join(
                FAKE_OUTPUT,
                audio_filename
            )

            # Skip already extracted files
            if os.path.exists(audio_path):
                continue

            video_path = os.path.join(
                root,
                file
            )

            try:

                extract_audio(
                    video_path,
                    audio_path
                )

                print(
                    f"[FAKE] {audio_filename}"
                )

            except Exception as e:

                print(
                    f"[FAILED] {file}"
                )

                print(e)

                continue

print("\n========================")
print("Audio Extraction Complete")
print("========================")

print(
    "Real Audio Files:",
    len(os.listdir(REAL_OUTPUT))
)

print(
    "Fake Audio Files:",
    len(os.listdir(FAKE_OUTPUT))
)

print("========================")