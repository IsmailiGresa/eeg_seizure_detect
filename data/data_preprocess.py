import os
import numpy as np
import pandas as pd
import mne
import torch
from torch.utils.data import DataLoader, Dataset
from control.config import args
from utils.utils import search_walk


def bipolar_signals_func(signals):
    bipolar_signals = []
    bipolar_signals.append(signals[0] - signals[4])
    bipolar_signals.append(signals[1] - signals[5])
    bipolar_signals.append(signals[4] - signals[9])
    bipolar_signals.append(signals[5] - signals[10])
    bipolar_signals.append(signals[9] - signals[15])
    bipolar_signals.append(signals[10] - signals[16])
    bipolar_signals.append(signals[15] - signals[13])
    bipolar_signals.append(signals[16] - signals[14])
    bipolar_signals.append(signals[9] - signals[6])
    bipolar_signals.append(signals[7] - signals[10])
    bipolar_signals.append(signals[6] - signals[8])
    bipolar_signals.append(signals[8] - signals[7])
    bipolar_signals.append(signals[0] - signals[2])
    bipolar_signals.append(signals[1] - signals[3])
    bipolar_signals.append(signals[2] - signals[6])
    bipolar_signals.append(signals[3] - signals[7])
    bipolar_signals.append(signals[6] - signals[11])
    bipolar_signals.append(signals[7] - signals[12])
    bipolar_signals.append(signals[11] - signals[13])
    bipolar_signals.append(signals[12] - signals[14])

    return torch.stack(bipolar_signals)


def parse_csv_annotations(csv_path, sample_rate, total_samples):
    if not os.path.exists(csv_path):
        print(f"⚠️ Warning: Missing CSV file {csv_path}. Returning zeros.")
        return torch.zeros(total_samples, dtype=torch.long)

    print(f"Parsing {csv_path} with sample rate {sample_rate} and total samples {total_samples}")
    df = pd.read_csv(csv_path, comment='#')

    seizure_intervals = df[df['label'] == 'seiz'][['start_time', 'stop_time']].values
    seizure_labels = torch.zeros(total_samples, dtype=torch.long)

    for start_time, stop_time in seizure_intervals:
        start_idx = int(start_time * sample_rate)
        stop_idx = min(int(stop_time * sample_rate), total_samples - 1)
        seizure_labels[start_idx:stop_idx] = 1

    return seizure_labels.clone().detach().long()


class Detector_Dataset(Dataset):
    def __init__(self, args, edf_files, csv_files, augment=False, data_type="training dataset"):
        self.data_type = data_type
        self.edf_files = edf_files
        self.csv_files = csv_files
        self.num_channel = args.num_channel
        self.seq_length = args.window_size * args.sample_rate
        self.augment = augment

        print(f"Loading {len(self.edf_files)} EEG files for {data_type}...")

    def __len__(self):
        return len(self.edf_files)

    def load_edf(self, edf_path):
        cache_path = edf_path.replace('.edf', '.npy')
        if os.path.exists(cache_path):
            print(f"⚡ Loading cached signal from {cache_path}")
            return torch.tensor(np.load(cache_path), dtype=torch.float32)
        raw = mne.io.read_raw_edf(edf_path, preload=True)

        eeg_channels = [ch for ch in raw.ch_names if "EEG" in ch or "-LE" in ch]
        raw.pick_channels(eeg_channels)

        signals = raw.get_data()
        num_channels = signals.shape[0]
        print(f"📊 Loaded {num_channels} channels from {edf_path}")

        signals = (signals - np.mean(signals, axis=1, keepdims=True)) / (np.std(signals, axis=1, keepdims=True) + 1e-10)

        if hasattr(args, "eeg_type") and args.eeg_type == "bipolar":
            signals = bipolar_signals_func(signals)
        elif hasattr(args, "eeg_type") and args.eeg_type == "uni_bipolar":
            bipolar = bipolar_signals_func(signals)
            signals = torch.cat((signals, bipolar))

        if signals.shape[1] < self.seq_length:
            pad_width = self.seq_length - signals.shape[1]
            signals = np.pad(signals, ((0, 0), (0, pad_width)), mode='constant')
        else:
            signals = signals[:, :self.seq_length]

        np.save(cache_path, signals)

        return torch.tensor(signals, dtype=torch.float32)

    def __getitem__(self, index):
        edf_path = self.edf_files[index]
        csv_path = self.csv_files[index]

        eeg_data = self.load_edf(edf_path)

        if not hasattr(args, "sample_rate"):
            args.sample_rate = 50
            print(f"⚠️ Warning: args.sample_rate not found, setting to default {args.sample_rate}")

        seizure_labels = parse_csv_annotations(csv_path, args.sample_rate, eeg_data.shape[1])

        return eeg_data, seizure_labels, os.path.basename(edf_path)


def eeg_binary_collate_fn(batch):
    max_channels = max(sample[0].shape[0] for sample in batch)
    max_seq_size = max(sample[0].shape[1] for sample in batch)

    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_channels, max_seq_size)
    targets = torch.zeros(batch_size, max_seq_size, dtype=torch.long)
    filenames = []

    for i, (signals, label, filename) in enumerate(batch):
        seq_length = signals.shape[1]
        num_channels = signals.shape[0]

        seqs[i, :num_channels, :seq_length] = signals
        targets[i, :seq_length] = label
        filenames.append(filename)

    return seqs, targets, torch.IntTensor([s.shape[1] for s, _, _ in batch]), filenames


def get_data_preprocessed(args):
    print("EEG Type:", args.eeg_type)
    print("Preparing data for EEG seizure detection...")

    train_edf_path = os.path.join(args.data_path, "train")
    val_edf_path = os.path.join(args.data_path, "dev")
    test_edf_path = os.path.join(args.data_path, "eval")

    train_edf_files = search_walk({"path": train_edf_path, "extension": ".edf"})
    val_edf_files = search_walk({"path": val_edf_path, "extension": ".edf"})
    test_edf_files = search_walk({"path": test_edf_path, "extension": ".edf"})

    train_csv_files = [f.replace(".edf", ".csv") for f in train_edf_files]
    val_csv_files = [f.replace(".edf", ".csv") for f in val_edf_files]
    test_csv_files = [f.replace(".edf", ".csv") for f in test_edf_files]

    # train_edf_files = train_edf_files[:10]
    # train_csv_files = train_csv_files[:10]
    #
    # val_edf_files = val_edf_files[:5]
    # val_csv_files = val_csv_files[:5]
    #
    # test_edf_files = test_edf_files[:100]
    # test_csv_files = test_csv_files[:100]

    train_dataset = Detector_Dataset(args, train_edf_files, train_csv_files, augment=True, data_type="train")
    val_dataset = Detector_Dataset(args, val_edf_files, val_csv_files, data_type="dev")
    test_dataset = Detector_Dataset(args, test_edf_files, test_csv_files, data_type="eval")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=eeg_binary_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=eeg_binary_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=eeg_binary_collate_fn)

    return train_loader, val_loader, test_loader, len(train_edf_files), len(val_edf_files), len(test_edf_files)
