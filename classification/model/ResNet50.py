import torch.nn as nn
import torchvision.models as models


class ResNet50(nn.Module):

    def __init__(self, num_classes):

        super(ResNet50, self).__init__()

        resnet = models.resnet50(pretrained=True)

        # Keep track of last layer input features
        num_features = resnet.fc.in_features
        print(num_features)

        # Remove last trained module (nn Sequential top classifier)
        modules = list(resnet.children())[:-1]
        self.base = nn.Sequential(*modules)

        # Freeze layers
        for param in self.base.parameters():
            param.requires_grad_(False)

        # Create a new top classifier, same code used in PyTorch ResNet50
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):

        features_base = self.base(x)

        # Resize batch_size x num_classes x 1 x 1 to batch_size x num_classes
        x = features_base.view(features_base.size(0), -1)

        out = self.fc(x)

        return out