# using what it learned to make predictions on new videos
import torch
from models.resnet_model import DeepFakeModel
from utils.frame_extractor import extract_frames
import torch.nn.functional as F #raw logits → probabilities


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

model = DeepFakeModel().to(device)

model.load_state_dict(torch.load("outputs/checkpoints/best_model.pth", map_location = device)) #load model checkpoint

model.eval() #set model to evaluation mode

video_path = "Portal Tech P90.mp4"
frames = extract_frames(video_path)
frames = torch.tensor(frames, dtype = torch.float32)
frames = frames.permute(0, 3, 1, 2)
frames = frames.unsqueeze(0) #add batch dimension
frames = frames.to(device)

print("Input Shape:", frames.shape)
with torch.no_grad():
    outputs = model(frames)

    probabilities = F.softmax(outputs, dim = 1) #convert raw logits to probabilities
    print("Probabilities:", probabilities)
    
    confidence, predictions = torch.max(probabilities, 1)
        
    confidence = confidence.item() * 100 #convert to percentage
    if confidence >= 90:
        risk = "HIGH RISK"
    elif confidence >= 70:
        risk = "MEDIUM RISK"
    else:
        risk = "LOW RISK"

    

#get predicted class (0 or 1)
if predictions.item() == 0:
    label = "PREDICTION: REAL"
else:
    label = "PREDICTION: FAKE"

print(f"\n Final Video Predictions: {label}")
print(f"Confidence Score: {confidence:.2f}%")
print(f"Risk Assessment: {risk}")