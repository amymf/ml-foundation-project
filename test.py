import torch
from dataset import get_dataloader
from model import MNISTClassifier

# Load the trained model
model = MNISTClassifier()
model.load_state_dict(torch.load("mnist_model.pth"))    # Load state (learned parameters)
model.eval()

_, test_loader = get_dataloader()   # Get the test data

correct, total = 0, 0   # Track the number of correct predictions
with torch.no_grad():
    for images, labels in test_loader:  # Iterate over each batch
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")