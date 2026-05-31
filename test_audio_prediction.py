from inference.predict_audio import predict_audio

prediction, confidence = predict_audio(
    "audio.wav"
)

if prediction == 0:
    label = "REAL"
else:
    label = "FAKE"

print("Prediction:", label)
print(f"Confidence: {confidence:.2f}%")