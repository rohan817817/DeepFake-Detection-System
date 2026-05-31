from inference.predict_audio import predict_audio

prediction, confidence = predict_audio(
    "audio_data/fake/1001_00007_id02342_RJPBPhJB8TA.wav"
)

if prediction == 0:
    label = "REAL"
else:
    label = "FAKE"

print("Prediction:", label)
print(f"Confidence: {confidence:.2f}%")