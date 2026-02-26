import torch
import torch.nn as nn
import torch.nn.functional as F

class SoundCNN(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 32 * 54, 128)
       # self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        print("Shape before flatten:", x.shape)
        x = torch.flatten(x, 1)

        
        x = F.relu(self.fc1(x))
       # x = self.dropout(x)
        x = self.fc2(x)

        return x
