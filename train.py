import torch
from model import BioRNN
from data import create_batch
import matplotlib.pyplot
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


import numpy as np


# Hyperparameters
# color vector
input_size = 8
# number of neurons
hidden_size = 64
# position vector
# output_size = 3
output_size = 2
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

    loss = criterion(final_output, final_target)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Logging
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# print("\nEvaluating on a new example...")
# # New sample
# model.eval()
# with torch.no_grad():
#     inputs, targets = create_batch(batch_size=1, T=T)
#     inputs = inputs.to(device)
#     targets = targets.to(device)
#
#     outputs = model(inputs)
#     prediction = outputs.squeeze()
#     target = targets[-1].squeeze()
#
#     print("Predicted position:", prediction.cpu().numpy())
#     print("Target position:   ", target.cpu().numpy())


import matplotlib.pyplot as plt
import numpy as np
import datetime
import os

# Evaluate model on a new random sample
print("\nEvaluating on a new example...")
model.eval()
with torch.no_grad():
    # Generate one random input-target pair repeated over T steps
    inputs, targets = create_batch(batch_size=1, T=T)
    inputs = inputs.to(device)
    targets = targets.to(device)

    outputs = model(inputs)  # shape: (T, 1, output_size)
    predicted = outputs.squeeze().cpu().numpy()  # shape: (T, 2)
    target = targets[-1].squeeze().cpu().numpy()  # shape: (2,)

    print("Predicted ball positions over time:")
    for t, pos in enumerate(predicted):
        print(f"t={t}: {pos}")
    print("Target position:", target)

# Define color landmarks
color_positions = {
    "red": (0.0, 0.0),
    "green": (1.0, 0.0),
    "blue": (0.0, 1.0),
    "yellow": (1.0, 1.0)
}

# Create figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title("Predicted Ball Trajectory", fontsize=14)
ax.set_xlabel("X position")
ax.set_ylabel("Y position")
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_aspect('equal')

# Plot color reference points
for color, (x, y) in color_positions.items():
    ax.scatter(x, y, s=180, color=color, edgecolor='black', zorder=3)

# Plot predicted path
ax.plot(predicted[:, 0], predicted[:, 1], color='black', linewidth=2, zorder=2, label='Predicted Path')

# Mark start and target
ax.scatter(predicted[0, 0], predicted[0, 1], color='gray', s=100, zorder=4, label='Start')
ax.scatter(target[0], target[1], color='black', marker='x', s=100, zorder=4, label='Target')

ax.legend()

# Save the figure with a timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)
filename = os.path.join(output_dir, f"trajectory_{timestamp}.png")
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved trajectory plot to {filename}")
