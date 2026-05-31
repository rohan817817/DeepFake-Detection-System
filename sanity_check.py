# sanity_check.py

from inference.predictor import predict_video

videos = [
    "data/real/101_00173.mp4",
    "data/real/100_00028_fake.mp4",
    "data/real/100_00028.mp4",
    "data/real/0_00109_fake.mp4",
    "data/fake/1013_00036_id01691_IVtS5z8Jrrk_id00944_wavtolip.mp4",
]

for video in videos:
    pred, conf = predict_video(video)

    label = "REAL" if pred == 0 else "FAKE"

    print(video)
    print(label, conf)
    print("-" * 40)