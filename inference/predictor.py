import torch
import torch.nn.functional as F

from models.resnet_model import DeepFakeModel
from utils.frame_extractor import extract_frames

def predict_video(video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepFakeModel().to(device)
    model.load_state_dict(torch.load("outputs/checkpoints/best_model.pth", map_location = device))
    model.eval()

    frames = extract_frames(video_path)

    frames = torch.tensor(frames, dtype = torch.float32)
    frames = frames.permute(0, 3, 1, 2)
    frames = frames.unsqueeze(0)
    frames = frames.to(device)

    with torch.no_grad():
        outputs = model(frames)
        probabilities = F.softmax(outputs, dim = 1)
        confidence, prediction = torch.max(probabilities, 1)

    return prediction.item(), confidence.item() * 100
                                     
