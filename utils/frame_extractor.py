import cv2
import numpy as np

def extract_frames(video_path, num_frames = 16, size = (224, 224)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError("Could not read video or video has no frames.")
        
        return []
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype = int)

    frames = [] #store the extracted frames

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()

        if not success:
            continue
    
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, size)
        frame = frame.astype(np.float32) / 255.0
        
        frames.append(frame)

    cap.release() #release the video capture object

    return np.array(frames)

