import cv2
import numpy as np


def extract_frames(
    video_path,
    num_frames=8,
    size=(224, 224)
):

    frames = []

    cap = cv2.VideoCapture(video_path)

    # Skip broken videos
    if not cap.isOpened():

        return np.zeros(
            (num_frames, size[0], size[1], 3),
            dtype=np.float32
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # Invalid video
    if total_frames <= 0:

        cap.release()

        return np.zeros(
            (num_frames, size[0], size[1], 3),
            dtype=np.float32
        )

    # Evenly spaced indices
    indices = np.linspace(
        0,
        total_frames - 1,
        num_frames,
        dtype=int
    )

    for idx in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            idx
        )

        try:

            success, frame = cap.read()

            # Failed frame
            if not success or frame is None:
                continue

            # Resize EARLY to reduce memory
            frame = cv2.resize(
                frame,
                size
            )

            # Convert BGR -> RGB
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Normalize
            frame = frame.astype(
                np.float32
            ) / 255.0

            frames.append(frame)

        except Exception:

            continue

    cap.release()

    # Pad missing frames
    while len(frames) < num_frames:

        if len(frames) > 0:

            frames.append(frames[-1])

        else:

            blank = np.zeros(
                (size[0], size[1], 3),
                dtype=np.float32
            )

            frames.append(blank)

    frames = frames[:num_frames]

    return np.array(frames)