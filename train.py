import torch
from model import BioRNN
from data import create_batch

# Hyperparameters
# color vector
input_size = 8
# number of neurons
hidden_size = 64
# position vector
output_size = 6
# delta time
dt = 0.1
# for every neuron
threshold = 1.0
# number of steps
T = 10
# examples per iteration
batch_size = 32
# Training cycles
epochs = 100


# Device - I have only cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
model = BioRNN(input_size, hidden_size, output_size, dt, threshold).to(device)

# Loss and optimizer - todo - learn about
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
for epoch in range(epochs):
    model.train()

    # Generate a new batch of data
    inputs, targets = create_batch(batch_size=batch_size, T=T)
    inputs = inputs.to(device)
    targets = targets.to(device)  # we only care about last step

    # Forward pass
    outputs = model(inputs)

    # Use only the last output (final decision of the network)
    final_output = outputs[-1]
    final_target = targets

    # ----delete---
    # final_target = targets[-1]
    # start_point = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)


    loss = criterion(final_output, final_target)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Logging
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

print("\nEvaluating on a new example...")
# New sample
model.eval()
with torch.no_grad():
    inputs, targets = create_batch(batch_size=1, T=T)
    inputs = inputs.to(device)
    targets = targets.to(device)

    outputs = model(inputs)
    prediction = outputs.squeeze()
    target = targets[-1].squeeze()

    print("Predicted position:", prediction.cpu().numpy())
    print("Target position:   ", target.cpu().numpy())
