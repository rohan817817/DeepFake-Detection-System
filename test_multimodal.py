from inference.multimodal_predictor import multimodal_predict

prediction, final_score, video_score, audio_score = multimodal_predict(
    "data/fake/1003_00027_id00371_t20i0HtPwW0.mp4"
)

print("Final Prediction:", prediction)
print(f"Final Score: {final_score:.2f}%")
print(f"Video Confidence: {video_score:.2f}%")
print(f"Audio Confidence: {audio_score:.2f}%")