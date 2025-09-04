import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from train import ColorToTrajectoryDataset, ColorBrainGRU, SEQ_LEN, NUM_COLORS, BATCH_SIZE
from load_data_from_mat import load_pre_data_from_mat


def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for inputs, targets in dataloader:
            preds, _ = model(inputs)
            loss = criterion(preds, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Test Loss: {avg_loss:.4f}")

def plot_trajectory_with_target(traj, target_point=None, predicted_end=None, title="Trajectory"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
    ax.plot(x, y, z, color='red', marker='o', label='Predicted Trajectory')

    if target_point is not None:
        ax.scatter(*target_point, color='green', s=100, label='Target Point')

    if predicted_end is not None:
        ax.scatter(*predicted_end, color='purple', s=60, label='Predicted End Point')

    ax.set_title(title)
    ax.legend()
    plt.show()

def main():
    X, y, s = load_pre_data_from_mat("trajectories.mat", max_seq_len=150)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    success_tensor = torch.tensor(s, dtype=torch.float32)

    idx_train, idx_test = train_test_split(range(len(X_tensor)), test_size=0.2, random_state=42)
    model = ColorBrainGRU()
    model.load_state_dict(torch.load("trained_model.pt"))
    model.eval()

    print("Showing one sample per color:")

    shown_colors = set()
    for i in idx_test:
        color = y_tensor[i].item()
        if color in shown_colors:
            continue

        success = success_tensor[i].item()
        true_traj = X_tensor[i].numpy()
        if success == 1:
            target_point = true_traj[-1]
        else:
            target_point = None

        color_one_hot = F.one_hot(torch.tensor(color), NUM_COLORS).float()
        color_input = torch.zeros(SEQ_LEN, NUM_COLORS)
        color_input[:10] = color_one_hot
        color_input = color_input.unsqueeze(0)  # batch dimension

        with torch.no_grad():
            pred_traj, _ = model(color_input)
        pred_traj = pred_traj.squeeze(0).numpy()

        predicted_end = pred_traj[-1]
        plot_trajectory_with_target(pred_traj, target_point=target_point, predicted_end=predicted_end, title=f"Color {color}")

        shown_colors.add(color)
        if len(shown_colors) >= NUM_COLORS:
            break

if __name__ == "__main__":
    main()
