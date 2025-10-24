import torch.nn as nn
from torchvision.models.resnet import resnet50, ResNet50_Weights


class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50Classifier, self).__init__()

        # Load the pretrained model
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Freeze the all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Modify the output layers with num of classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
