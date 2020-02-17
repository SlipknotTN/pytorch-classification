import torch.nn as nn
import torchvision.models as models


class SqNet(nn.Module):

    def __init__(self, num_classes):

        super(SqNet, self).__init__()

        sqnet = models.squeezenet1_1(pretrained=True)

        # Remove last trained module (nn Sequential top classifier)
        modules = list(sqnet.children())[:-1]
        self.base = nn.Sequential(*modules)

        # Freeze layers
        for param in self.base.parameters():
            param.requires_grad_(False)

        # Create a new top classifier, same code used in PyTorch Sqnet
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13, stride=1)
        )

    def forward(self, x):

        features_base = self.base(x)

        out = self.classifier(features_base)

        # Resize batch_size x num_classes x 1 x 1 to batch_size x num_classes
        out = out.view(out.size(0), -1)

        return out
