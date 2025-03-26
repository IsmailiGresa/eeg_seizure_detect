import os
import torch
import argparse
from data.data_preprocess import get_data_preprocessed

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="/Users/gresaismaili/Desktop/TUH-EEG")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_channel", type=int, default=30)
parser.add_argument("--window_size", type=int, default=1)
parser.add_argument("--sample_rate", type=int, default=256)
parser.add_argument("--feature-sample_rate", type=int, default=50)
parser.add_argument("--eeg_type", type=str, default="bipolar", choices=["unipolar", "bipolar", "uni_bipolar"])

parser.add_argument("--additive_gaussian_noise_max", type=float, default=0.2)
parser.add_argument("--time_shift_min", type=int, default=-50)
parser.add_argument("--time_shift_max", type=int, default=50)
args = parser.parse_args([])

train_loader, val_loader, test_loader, train_size, val_size, test_size = get_data_preprocessed(args)

print(f"Training samples: {train_size}")
print(f"Validation samples: {val_size}")
print(f"Test samples: {test_size}")

for batch in train_loader:
    eeg_data, seizure_labels, seq_lengths, filenames = batch

    print("\nSample EEG Data Shape:", eeg_data.shape)  # Expected: (batch_size, num_channels, seq_length)
    print("Sample Seizure Label Shape:", seizure_labels.shape)  # Expected: (batch_size,)
    print("Sequence Lengths:", seq_lengths)
    print("Filenames:", filenames)
    break