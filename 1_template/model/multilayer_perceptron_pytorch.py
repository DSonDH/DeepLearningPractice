import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

# Define the MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out

# Define the hyperparameters
input_size = 2
hidden_size = 10
output_size = 1
lr = 0.1
num_epochs = 2000

# Create the MLP model
model = MLP(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# Generate some dummy data
x = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y = np.array([0, 1, 1, 0]).T

# Train the MLP
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(torch.tensor(x, dtype=torch.float32))
    loss = criterion(outputs, torch.tensor(y, dtype=torch.float32))
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print the loss every 100 epochs
    if (epoch+1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
