import torch
import torch.nn.functional as F #raw logits → probabilities

from audio.feature_extractor import extract_mfcc
from models.audio_model import AudioClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_audio(audio_path):
    model = AudioClassifier()

    model.load_state_dict(torch.load("outputs/checkpoints/best_audio_model.pth", map_location = device))
    model = model.to(device)
    model.eval()

    mfcc = extract_mfcc(audio_path)
    mfcc = torch.tensor(mfcc, dtype = torch.float32)
    mfcc = mfcc.unsqueeze(0)
    mfcc = mfcc.to(device)

    with torch.no_grad():
        outputs = model(mfcc)
        
        probabilities = F.softmax(outputs, dim = 1)
        confidence, prediction = torch.max(probabilities, 1)

    return (prediction.item(), confidence.item() * 100)