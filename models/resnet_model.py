import torch.nn as nn #import neural network module from PyTorch
import torchvision.models as models #import pre-trained models from torchvision

class DeepFakeModel(nn.Module): 
    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(weights = models.ResNet18_Weights.DEFAULT) #use ResNet-18 as the backbone for feature extraction

        in_features = self.backbone.fc.in_features #remove final classification layer only real an fake classification
        self.backbone.fc = nn.Linear(in_features, 2) #replace final layer with a new linear layer for binary classification

    def forward(self, x):
        b, f, c, h, w = x.shape #batch size, number of frames, channels, height, width

        x = x.view(b * f, c, h ,w)

        features = self.backbone(x) #extract features from the frames
        features = features.view(b, f, -1) #reshape features back to (batch size, number of frames, feature dimension)
        output = features.mean(dim = 1)

        return output