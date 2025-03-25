import torch.nn as nn
import torch.nn.functional as F

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)   # Flatten the image
        x = F.relu(self.fc1(x)) # Apply ReLU activation function to introduce non-linearity
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # No activation function as we want raw output
        return x
