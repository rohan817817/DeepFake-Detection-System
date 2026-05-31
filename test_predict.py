from inference.predictor import predict_video

prediction, confidence = predict_video(
    "data/fake/805_011.mp4"
)

print("Prediction:", prediction)
print("Confidence:", confidence)