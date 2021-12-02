import torch
from torch.utils.data import TensorDataset


def load_meld(mode, path):
    mode = mode.lower()
    text_dim = 600
    names_data = ['train_x_{}.pt', 'val_x_{}.pt', 'test_x_{}.pt',
                  'train_y_{}.pt', 'val_y_{}.pt', 'test_y_{}.pt',
                  'train_mask_{}.pt', 'val_mask_{}.pt', 'test_mask_{}.pt']
    
    data_path = path
    x_train = torch.load(data_path + names_data[0].format(mode)).to(dtype=torch.float32)
    x_valid = torch.load(data_path + names_data[1].format(mode)).to(dtype=torch.float32)
    x_test = torch.load(data_path + names_data[2].format(mode)).to(dtype=torch.float32)
    y_train = torch.load(data_path + names_data[3].format(mode)).to(dtype=torch.long)
    y_valid = torch.load(data_path + names_data[4].format(mode)).to(dtype=torch.long)
    y_test = torch.load(data_path + names_data[5].format(mode)).to(dtype=torch.long)
    mask_train = torch.load(data_path + names_data[6].format(mode)).to(dtype=torch.float32)
    mask_valid = torch.load(data_path + names_data[7].format(mode)).to(dtype=torch.float32)
    mask_test = torch.load(data_path + names_data[8].format(mode)).to(dtype=torch.float32)

    classes = torch.max(y_train).item() + 1
    total_dim = x_train.size(2)
    train_set = TensorDataset(x_train[:, :, text_dim:], x_train[:, :, :text_dim], y_train)
    valid_set = TensorDataset(x_valid[:, :, text_dim:], x_valid[:, :, :text_dim:], y_valid)
    test_set = TensorDataset(x_test[:, :, text_dim:], x_test[:, :, :text_dim:], y_test)

    return classes, {'audio_dim': total_dim-text_dim, 'text_dim': text_dim,
                     'n_train': x_train.size(0), 'n_valid': x_valid.size(0), 'n_test': x_test.size(0),
                     'num_utterance': x_train.size(1)}, train_set, valid_set, test_set, mask_train, mask_valid, mask_test
