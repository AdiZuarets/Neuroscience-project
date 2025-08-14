def load_pre_data_from_mat(path, max_seq_len=150):
    import numpy as np
    import scipy.io

    def pad_or_truncate_sequence(seq, max_len):
        length = len(seq)
        if length >= max_len:
            return seq[:max_len]
        else:
            padding = np.tile(seq[-1], (max_len - length, 1))
            return np.vstack([seq, padding])

    data = scipy.io.loadmat(path)
    all_trials = data['all_trajectories'][0]

    trajectories, labels, successes = [], [], []

    for trial_struct in all_trials:
        pre = trial_struct[0, 0]['pre'][0, 0]
        xyz_all = pre['xyz_all'][0]
        trial_type = pre['trials_types'][0]
        success = pre['success'][0]

        for i in range(len(xyz_all)):
            xyz = xyz_all[i]
            if xyz.shape[1] != 3:
                continue
            padded = pad_or_truncate_sequence(xyz, max_seq_len)
            trajectories.append(padded)
            labels.append(trial_type[i] - 1)
            successes.append(success[i])

    return np.stack(trajectories), np.array(labels), np.array(successes)

X, y, s = load_pre_data_from_mat("/Users/adi/Library/CloudStorage/GoogleDrive-adi.zuarets@mail.huji.ac.il/My Drive/Studies/Year3/Semester 2/lab/part_1/pythonProject/real_data_train/real_data_train/trajectories.mat", max_seq_len=150)

import torch
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
s_tensor = torch.tensor(s, dtype=torch.float32)

# print("X shape:", X_tensor.shape)
# print("y shape:", y_tensor.shape)
# print("s shape:", s_tensor.shape)
# print(X_tensor[:10])
# print(y_tensor[:20])
# print(s_tensor[:10])

