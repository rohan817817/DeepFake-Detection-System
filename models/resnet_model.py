import torch.nn as nn #import neural network module from PyTorch
import torchvision.models as models #import pre-trained models from torchvision

class DeepFakeModel(nn.Module): 
    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(weights = models.ResNet18_Weights.DEFAULT) #use ResNet-18 as the backbone for feature extraction

        in_features = self.backbone.fc.in_features #remove final classification layer only real an fake classification
        self.backbone.fc = nn.Identity()
        self.attention = nn.Linear(in_features, 1)
        self.classifier = nn.Linear(in_features, 2)

    def forward(self, x):
        b, f, c, h, w = x.shape #batch size, number of frames, channels, height, width

        x = x.view(b * f, c, h ,w)

        features = self.backbone(x) #extract features from the frames
        features = features.view(b, f, -1) #reshape features back to (batch size, number of frames, feature dimension)
        
        attention_score = self.attention(features)
        attention_weights = attention_score.softmax(dim = 1) #calculate attention weights across frames
        weighted_features = features * attention_weights #apply attention weights to the features
        video_features = weighted_features.sum(dim=1)

        output = self.classifier(video_features)

        return output