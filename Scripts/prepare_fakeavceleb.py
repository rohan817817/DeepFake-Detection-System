import os
import shutil

SOURCE_ROOT = r"C:\Users\HI\Downloads\archive-2\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"

REAL_FOLDERS = [
    "RealVideo-RealAudio",
    "RealVideo-FakeAudio"
]

FAKE_FOLDERS = [
    "FakeVideo-RealAudio",
    "FakeVideo-FakeAudio"
]

REAL_DEST = "data/real"
FAKE_DEST = "data/fake"

os.makedirs(REAL_DEST, exist_ok=True)
os.makedirs(FAKE_DEST, exist_ok=True)


def copy_videos(source_folder, destination):

    count = 0

    for root, dirs, files in os.walk(source_folder):

        for file in files:

            if file.endswith(".mp4"):

                src = os.path.join(root, file)

                dst = os.path.join(
                    destination,
                    f"{count}_{file}"
                )

                shutil.copy2(src, dst)

                count += 1

                if count >= 100:
                    return count

    return count


real_count = 0

for folder in REAL_FOLDERS:

    folder_path = os.path.join(
        SOURCE_ROOT,
        folder
    )

    real_count += copy_videos(
        folder_path,
        REAL_DEST
    )


fake_count = 0

for folder in FAKE_FOLDERS:

    folder_path = os.path.join(
        SOURCE_ROOT,
        folder
    )

    fake_count += copy_videos(
        folder_path,
        FAKE_DEST
    )

print("Real videos:", real_count)
print("Fake videos:", fake_count)