import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class BioRNN(nn.Module):
    """
    input_size - color vector size
    hidden_size - the number of neurons in the hidden state
    output_size - color position vector size
    dt - delta time
    threshold - activate if higher
    """
    def __init__(self, input_size=8, hidden_size=50, output_size=6, dt=0.1, threshold=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.threshold = threshold

        # Input weights: connect input to recurrent neurons
        std_in = 1.0 / math.sqrt(input_size)
        self.W_in = nn.Parameter(torch.randn(hidden_size, input_size) * std_in, requires_grad= True)

        # Recurrent weights: connect neurons to each other
        std_rec = 1.0 / math.sqrt(hidden_size)
        self.W_rec = nn.Parameter(torch.randn(hidden_size, hidden_size) * std_rec, requires_grad= True)

        # Prevent self-connections (set diagonal to 0)
        with torch.no_grad():
            self.W_rec.data.fill_diagonal_(0.0)

        # Bias term added to each neuron's input - Personal threshold voltage
        self.bias = nn.Parameter(torch.zeros(hidden_size))

        # Output layer: maps hidden state to 3D output - the position
        self.W_out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        """
        input_seq: tensor of shape (T, B, input_size)
        T = sequence length (time steps)
        B = batch size (number of samples)
        input_size = dimension of input vector (8 for one-hot color)
        """
        T, B, _ = input_seq.shape

        # Initialize hidden state (neural activity) to zero
        h = torch.zeros(B, self.hidden_size, device=input_seq.device)

        outputs = []

        for t in range(T):
            x_t = input_seq[t]  # input at time step t (shape: B x input_size)

            # Euler update rule:
            # -h - Self-degradation

            # W_in * x_t – input drive from color vector at time t
            # In our case: same one-hot color vector repeated over time
            # W_rec * h	- neuron network
            # bias - Personal threshold voltage
            # dh - dh/dt
            # dt - delta time
            # dh/dt = -h + W_in * x + W_rec * h + bias
            # h ← h + dt * dh

            # Compute firing rate using ReLU activation
            r = F.relu(h)  # non-negative "rate" based on membrane potential
            # Compute dh/dt based on rate r
            # dh/dt = -h + W_in * x + W_rec * r + bias
            dh = -h + F.linear(x_t, self.W_in) + F.linear(r, self.W_rec) + self.bias
            # Euler update
            h = h + self.dt * dh


            # Use r (rate) as output drive
            y_t = self.W_out(r)
            outputs.append(y_t)

            # Stack outputs over time → shape: T x B x 6
        return torch.stack(outputs)