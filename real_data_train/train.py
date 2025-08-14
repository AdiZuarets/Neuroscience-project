import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from load_data_from_mat import load_pre_data_from_mat

EPOCHS = 20
BATCH_SIZE = 32
LR = 0.001
SEQ_LEN = 150
NUM_COLORS = 8
INPUT_DURATION = 30
HIDDEN_SIZE = 64

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
        hidden_states, _ = self.gru(x)
        predicted_xyz = self.predict_xyz(hidden_states)
        return predicted_xyz, hidden_states

def train(model, dataloader, epochs=EPOCHS, lr=LR):
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

def main(X_tensor, y_tensor):
    idx_train, idx_test = train_test_split(range(len(X_tensor)), test_size=0.2, random_state=42)
    dataset = ColorToTrajectoryDataset(X_tensor[idx_train], y_tensor[idx_train])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ColorBrainGRU()
    train(model, dataloader)
    torch.save(model.state_dict(), "trained_model.pt")
    print("Model saved to trained_model.pt")

    return model

if __name__ == "__main__":
    X, y, _ = load_pre_data_from_mat("trajectories.mat", max_seq_len=150)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    main(X_tensor, y_tensor)
