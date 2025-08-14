import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

SEQ_LEN = 150
NUM_COLORS = 8
HIDDEN_SIZE = 64
INPUT_DURATION = 30

class ColorToTrajectoryDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        color_idx = self.y[idx]
        color_one_hot = F.one_hot(color_idx, NUM_COLORS).float()

        color_input = torch.zeros(SEQ_LEN, NUM_COLORS)
        color_input[:INPUT_DURATION] = color_one_hot

        return color_input, self.X[idx]

class ColorBrainGRU(nn.Module):
    def __init__(self, input_size=NUM_COLORS, hidden_size=HIDDEN_SIZE, output_size=3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.predict_xyz = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hidden_states, _ = self.gru(x)  # shape: (batch, seq_len, hidden_size)
        predicted_xyz = self.predict_xyz(hidden_states)  # shape: (batch, seq_len, 3)
        return predicted_xyz, hidden_states

def train(model, dataloader, epochs=20, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            preds, _ = model(inputs)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

@torch.no_grad()
def generate_trajectory(model, color_index):
    color = F.one_hot(torch.tensor(color_index), NUM_COLORS).float()
    color_seq = torch.zeros(SEQ_LEN, NUM_COLORS)
    color_seq[:INPUT_DURATION] = color
    color_seq = color_seq.unsqueeze(0)
    model.eval()
    predicted_traj, _ = model(color_seq)
    return predicted_traj.squeeze().numpy()
