import torch
import torch.nn as nn
import torch.nn.functional as F


class modVGG(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            nn.Flatten(),
            nn.Linear(4*4*256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, n_classes),
        )

    def forward(self, X: torch.Tensor):
        X = self.seq(X)
        return X


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)         # 28x28
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)        # 10x10
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 5x5
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.classifier = nn.Linear(84, 10)

    def forward(self, X: torch.Tensor):
        X = self.pool1(F.relu(self.conv1(X)))
        X = self.pool2(F.relu(self.conv2(X)))
        X = self.flatten(X)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        y = self.classifier(X)
        return y


if __name__ == '__main__':
    # model = SimpleCNN()
    model = modVGG()
    print(sum(param.numel() for param in model.parameters()))
