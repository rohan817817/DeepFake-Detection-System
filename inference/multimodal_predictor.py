import os

from inference.predictor import predict_video
from inference.predict_audio import predict_audio
from utils.audio_extractor import extract_audio


def multimodal_predict(video_path):

    video_prediction, video_confidence = predict_video(video_path)

    audio_path = "temp_audio.wav"

    audio_prediction = 0
    audio_confidence = 50.0

    try:

        if os.path.exists(audio_path):
            os.remove(audio_path)

        extract_audio(
            video_path,
            audio_path
        )

        if os.path.exists(audio_path):

            audio_prediction, audio_confidence = predict_audio(
                audio_path
            )

        else:

            print(
                "No audio extracted. Using neutral audio score."
            )

    except Exception as e:

        print(
            "Audio Error:",
            e
        )

    video_fake_score = (
        video_confidence
        if video_prediction == 1
        else 100 - video_confidence
    )

    audio_fake_score = (
        audio_confidence
        if audio_prediction == 1
        else 100 - audio_confidence
    )

    final_fake_score = (
        0.7 * video_fake_score +
        0.3 * audio_fake_score
    )

    if final_fake_score >= 50:

        final_prediction = "FAKE"

    else:

        final_prediction = "REAL"

    return (
        final_prediction,
        final_fake_score,
        video_confidence,
        audio_confidence
    )