import os
import numpy as np
from glob import glob
from itertools import islice
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset

# Constants
SAMPLES_DIR = 'outputs_dataset/txt2img-samples/intermediate_results'
SAMPLES_TEMPLATE = 'chunk_*_batch_*_x.npy'

CONDS_DIR = 'outputs_dataset/txt2img-samples/conditions'
CONDS_TEMPLATE = 'chunk_*_batch_*.npy'


def preprocess_data(n_files=160):
    files = glob(os.path.join(SAMPLES_DIR, SAMPLES_TEMPLATE))
    all_xs, all_prompts, all_conditions = [], [], []
    for i in range(n_files):
        filename = files[i]
        all_xs.append(np.load(filename))
        prompts = get_prompts(filename)
        all_prompts.extend(prompts)
        # filename looks like "outputs_dataset/txt2img-samples/intermediate_results/chunk_*_batch_*_x.npy"
        # and we need to get the corresponding condition file, which looks like "outputs_dataset/txt2img-samples/conditions/chunk_*_batch_*.npy"
        cond_filename = filename.replace(SAMPLES_DIR, CONDS_DIR).replace('_x.npy', '.npy')
        all_conditions.append(np.load(cond_filename))

    # The vectors
    all_xs = np.concatenate(all_xs)
    all_xs = np.reshape(all_xs, (all_xs.shape[0], all_xs.shape[1], -1))
    all_xs = np.transpose(all_xs, (1, 0, 2)) # [seq_len, samples, D=channels*H*W]
    print(f'Loaded {n_files} files -> {all_xs.shape[1]} samples with {all_xs.shape[0]} time steps each of dimensionality {all_xs.shape[2]}.')
    # The conditions
    all_conditions = np.concatenate(all_conditions) # [samples, n_tokens=77, D=1_024] # it's 77 for all of them
    all_conditions = np.reshape(all_conditions, (all_conditions.shape[0], -1)) # [samples, n_tokens*D=77*1_024=78_848]
    return all_xs, all_prompts, all_conditions

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def get_prompts(filename):
    chunk_num = filename.split('/')[-1].split('_')[1]
    batch_num = int(filename.split('/')[-1].split('_')[3])

    prompt_file = f'outputs_dataset/prompts/chunk_{chunk_num}.txt'

    with open(prompt_file, "r") as f:
        data = f.read().splitlines()

    # Now split into batches of 8 items (hardcoded becayse that's how the data was generated)
    data = list(chunk(data, 8))
    data = data[batch_num]
    return data


class PromptCondTensorDataset(Dataset):
    def __init__(self, data, conds, prompts):
        """
        Args:
            tensor_dataset (TensorDataset): A TensorDataset object containing your vectors.
            prompts (list of str): a list of strings associated with each vector.
        """
        self.seq_tensor_dataset = self.gen_seq_tensor_dataset(data)
        self.conds = TensorDataset(torch.from_numpy(conds).float())
        self.prompts = prompts

    def __len__(self):
        return len(self.seq_tensor_dataset)

    def __getitem__(self, idx):
        X, y = self.seq_tensor_dataset[idx]
        cond = self.conds[idx][0]
        prompt = self.prompts[idx]
        return X, y, cond, prompt

    def gen_seq_tensor_dataset(self, data): # , batch_size=64):
        # data has shape [samples, seq_len, features]
        X = torch.from_numpy(data[:, :-1, :]).float() # first X time steps
        y = torch.from_numpy(data[:, 1:, :]).float() # last X time steps
        return TensorDataset(X, y)

def get_dataloaders(all_data, all_prompts, all_conds, train_perc, batch_size=8):
    """
    all_data is array of shape [seq_len=21, samples, D=channels*H*W=16_384]
    all_prompts is list of length samples
    all_conds is array of shape [samples, n_tokens*D=77*1_024=78_848]
    """
    # Split (data, prompts and conditions) into train and test
    train_idx_limit = int(all_data.shape[1] * train_perc)

    train_data = all_data[:, :train_idx_limit, :].transpose(1, 0, 2)
    train_prompts = all_prompts[:train_idx_limit]
    train_conds = all_conds[:train_idx_limit]

    rest_data = all_data[:, train_idx_limit:, :]
    rest_prompts = all_prompts[train_idx_limit:]
    rest_conds = all_conds[train_idx_limit:]

    # Divide the rest into test and validation
    test_idx_limit = int(rest_data.shape[1] * 0.5)

    test_data = rest_data[:, :test_idx_limit, :].transpose(1, 0, 2)
    test_prompts = rest_prompts[:test_idx_limit]
    test_conds = rest_conds[:test_idx_limit]

    val_data = rest_data[:, test_idx_limit:, :].transpose(1, 0, 2)
    val_prompts = rest_prompts[test_idx_limit:]
    val_conds = rest_conds[test_idx_limit:]

    # Create the custom dataset and data loader
    train_dataset = PromptCondTensorDataset(train_data, train_conds, train_prompts)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    val_dataset = PromptCondTensorDataset(val_data, val_conds, val_prompts)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    test_dataset = PromptCondTensorDataset(test_data, test_conds, test_prompts)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    return train_data_loader, val_data_loader, test_data_loader
