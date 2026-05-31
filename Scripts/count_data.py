import os

print("Real:", len(os.listdir("data/real")))
print("Fake:", len(os.listdir("data/fake")))
#audio
print("Real:", len(os.listdir("audio_data/real")))
print("Fake:", len(os.listdir("audio_data/fake")))