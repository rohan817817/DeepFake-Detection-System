import os

fake_files = os.listdir("audio_data/real")

for file in fake_files[999:]:
    os.remove(os.path.join("audio_data/real", file))