from inference.predictor import predict_video
from inference.predict_audio import predict_audio

from utils.audio_extractor import extract_audio

def multimodal_predict(video_path):
    video_prediction, video_confidence = predict_video(video_path)

    audio_path = "temp_audio.wav"
    extract_audio(video_path, audio_path)

    audio_prediction, audio_confidence = predict_audio(audio_path)

    final_score = (video_confidence + audio_confidence) / 2

    if final_score >= 50:
        final_prediction = "FAKE"
    else:
        final_prediction = "REAL"

    return (final_prediction, final_score, video_confidence, audio_confidence)