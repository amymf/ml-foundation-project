import torch
import torch.optim as optim
import torch.nn as nn
from dataset import get_dataloader
from model import MNISTClassifier
import matplotlib.pyplot as plt

# Hyperparameters
epochs = 30
learning_rate = 0.001

# Load Data
train_loader, test_loader = get_dataloader()

# Initialize Model
model = MNISTClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Track loss for plotting
loss_history = []

# Training Loop
for epoch in range(epochs):
    model.train()   # Set the model to training mode
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()   # Zero gradients to avoid accumulation
        outputs = model(images) # Forward pass
        loss = criterion(outputs, labels)
        loss.backward()         # Backward pass
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")

# Save the trained model
torch.save(model.state_dict(), "mnist_model.pth")   # Save state
print("Model saved as mnist_model.pth")

# Plot the training loss
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), loss_history, marker='o', linestyle='-')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid()
plt.show()