from utils.frame_extractor import extract_frames

video_path = "Portal Tech P90.mp4"

frames = extract_frames(video_path)

print("Frames Shape:", frames.shape)
print("Number of Frames:", len(frames))
print("Singlr Frame Shape:", frames[0].shape)
