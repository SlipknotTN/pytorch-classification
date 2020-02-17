import torch.nn as nn
import torch.nn.functional as F


class SimpleNetBW(nn.Module):
    """
    Simple Net architecture for BW images
    """
    def __init__(self, input_size, num_classes):

        super(SimpleNetBW, self).__init__()

        # Define convs, poolings and final fc

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, stride=1)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(p=0.5)

        # Num features depends on input size, we resize only one time, because we use stride 1 convolutions and
        # only one max poolings
        num_features = int(input_size / 2) * int(input_size / 2) * self.conv4.out_channels
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):

        # Apply convs, poolings and final fc

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))

        x = self.max_pool(F.relu(self.conv4(x)))

        # flatten
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x