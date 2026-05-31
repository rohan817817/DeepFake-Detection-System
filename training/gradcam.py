import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from models.resnet_model import DeepFakeModel
from utils.frame_extractor import extract_frames

def generate_gradcam(video_path):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print("Using Device:", device)

    model = DeepFakeModel()
    model.load_state_dict(torch.load("outputs/checkpoints/best_model.pth", map_location = device)) #load the best model checkpoint
    model = model.to(device)
    model.eval() #set model to evaluation mode

    target_layer = model.backbone.layer4 #last convolutional layer in ResNet-18 to capture high-level features

    gradients = None
    activations = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    frames = extract_frames(video_path)
    frame = frames[0]
    frame = (frame * 255).astype("uint8")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])

    input_tensor = transform(frame).unsqueeze(0)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    #forward Pass
    output = model(input_tensor)

    predicted_class = output.argmax(dim = 1)

    #Backward Pass
    model.zero_grad()
    output[0, predicted_class.item()].backward()

    print("Activations Shape:", activations.shape)
    print("Gradients Shape:", gradients.shape)

    pooled_gradients= gradients.mean(dim = [0, 2, 3]) #compute GRAD-CAM weights
    print("Pooled Shape:", pooled_gradients.shape)
    activations = activations[0]

    for i in range(pooled_gradients.shape[0]):
        activations[i] *= pooled_gradients[i]

    heatmap = activations.mean(dim = 0).cpu().detach().numpy() 
    heatmap = np.maximum(heatmap, 0) #ReLU

    if heatmap.max() != 0:
        heatmap /= heatmap.max()

    heatmap = cv2.resize(heatmap, (224, 224))

    # To Color Map
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_image = cv2.resize(frame, (224, 224))

    superimposed = heatmap * 0.4 + original_image
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

    saved = cv2.imwrite("outputs/graphs/gradcam.jpg", superimposed)
    print("GradCam saved:", saved)

    forward_handle.remove()
    backward_handle.remove()

    return "outputs/graphs/gradcam.jpg"
