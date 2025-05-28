import torch
import random

# Input - colors vectors
vectors_colors = {
    "red": [1, 0, 0, 0, 0, 0, 0, 0],
    "green": [0, 1, 0, 0, 0, 0, 0, 0],
    "blue": [0, 0, 1, 0, 0, 0, 0, 0],
    "yellow": [0, 0, 0, 1, 0, 0, 0, 0],
    # "purple": [0, 0, 0, 0, 1, 0, 0, 0],
    # "orange": [0, 0, 0, 0, 0, 1, 0, 0],
    # "cyan": [0, 0, 0, 0, 0, 0, 1, 0],
    # "white": [0, 0, 0, 0, 0, 0, 0, 1],
}

# Output - 3D position for each color
color_to_point = {
    "red": [0.0, 0.0],
    "green": [1.0, 0.0],
    "blue": [0.0, 1.0],
    "yellow": [1.0, 1.0]
}
    # "red": [0.0, 0.0, 0.0],
    # "green": [1.0, 0.0, 0.0],
    # "blue": [0.0, 1.0, 0.0],
    # "yellow": [1.0, 1.0, 0.0],
#     "purple": [0.0, 0.0, 1.0],
#     "orange": [1.0, 0.0, 1.0],
#     "cyan": [0.0, 1.0, 1.0],
#     "white": [1.0, 1.0, 1.0],
# }

def generate_example():
    """Returns a random (input_vector, target_position) pair as torch tensors"""
    color = random.choice(list(vectors_colors.keys()))
    input_vec = torch.tensor(vectors_colors[color], dtype=torch.float32)
    #--------------------------------------
    target_vec = torch.tensor(color_to_point[color], dtype=torch.float32)
    # start_pos = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    # movement = target_vec - start_pos
    #---------------------------------
    return input_vec, target_vec

# def generate_example():
#     """Returns a (input_vector, target_movement_vector) pair"""
#     start_pos = torch.tensor([0.5, 0.5], dtype=torch.float32)  # always start from center
#
#     color = random.choice(list(vectors_colors.keys()))
#     input_vec = torch.tensor(vectors_colors[color], dtype=torch.float32)
#     target_pos = torch.tensor(color_to_point[color], dtype=torch.float32)
#
#     movement = target_pos - start_pos  # what the network should learn
#
#     return input_vec, movement


def create_batch(batch_size=32, T=10):
    """Creates a batch of repeated input-target sequences over T time steps."""
    inputs = []
    targets = []
    for _ in range(batch_size):
        x, y = generate_example()
        # make a matrix for the input with the time T
        x_seq = x.unsqueeze(0).repeat(T, 1)  # (T, 8)
        # y_seq = y.unsqueeze(0).repeat(T, 1)  # (T, 3)
        inputs.append(x_seq)
        targets.append(y)

    # Dividing matrices into time units
    inputs = torch.stack(inputs, dim=1)  # (T, B, 8)
    targets = torch.stack(targets, dim=0)  # (T, B, 3)
    return inputs, targets


"""
inputs =
[
  # t=0
  [
    [0, 1, 0, 0, 0, 0, 0, 0],   # green
    [0, 0, 0, 0, 1, 0, 0, 0],   # purple
  ],
  # t=1
  [
    [0, 1, 0, 0, 0, 0, 0, 0],   # green
    [0, 0, 0, 0, 1, 0, 0, 0],   # purple
  ],
  # t=2
  [
    [0, 1, 0, 0, 0, 0, 0, 0],   # green
    [0, 0, 0, 0, 1, 0, 0, 0],   # purple
  ]
]


targets =
[
  # t=0
  [
    [1.0, 0.0, 0.0],   # green
    [0.0, 0.0, 1.0],   # purple
  ],
  # t=1
  [
    [1.0, 0.0, 0.0],   # green
    [0.0, 0.0, 1.0],   # purple
  ],
  # t=2
  [
    [1.0, 0.0, 0.0],   # green
    [0.0, 0.0, 1.0],   # purple
  ]
]

"""