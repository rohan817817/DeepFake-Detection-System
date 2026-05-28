import cv2
import glob


real_videos = glob.glob("data/real/*.mp4")

fake_videos = glob.glob("data/fake/*.mp4")


videos = real_videos + fake_videos


bad_videos = []


for video in videos:

    cap = cv2.VideoCapture(video)

    success, frame = cap.read()


    if not success:

        bad_videos.append(video)


print("\nBAD VIDEOS:\n")


for bad_video in bad_videos:

    print(bad_video)


print(f"\nTotal Bad Videos: {len(bad_videos)}")