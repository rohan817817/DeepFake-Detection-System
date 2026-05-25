# using what it learned to make predictions on new videos
import torch
from models.resnet_model import DeepFakeModel
from utils.frame_extractor import extract_frames

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

model = DeepFakeModel()

model.load_state_dict(torch.load("outputs/checkpoints/deepfake_model.pth", map_location = device)) #load model checkpoint

model = model.to(device)
model.eval() #set model to evaluation mode

video_path = "Portal Tech P90.mp4"
frames = extract_frames(video_path)

frames = torch.tensor(frames)
frames = frames.permute(0, 3, 1, 2)
frames = frames.unsqueeze(0) #add batch dimension
frames = frames.to(device)


with torch.no_grad():
    outputs = model(frames)
    predictions = torch.argmax(outputs, dim = 1).item() #get predicted label

if predictions == 0:
    print("PREDICTION: REAL")
else:
    print("PREDICTION: FAKE")
