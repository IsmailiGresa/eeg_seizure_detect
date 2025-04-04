import platform
import numpy as np
import torch
import torch.nn as nn

class SPECTROGRAM_FEATURE_HYPEROPT(nn.Module):
    def __init__(self, args,
            sample_rate: int = 200,
            feature_sample_rate: int = 50,
            frame_length: int = 1,
            frame_shift: float = 1.0,
            freq_max: int = 100,
            feature_extract_by: str = 'kaldi'):
        super(SPECTROGRAM_FEATURE_HYPEROPT, self).__init__()

        self.args = args
        self.sample_rate = sample_rate
        self.feature_sample_rate = feature_sample_rate

        self.feature_extract_by = feature_extract_by.lower()
        self.freq_resolution = 1
        self.freq_max = freq_max

        self.final_batch = []

        assert platform.system().lower() == 'linux' or platform.system().lower() == 'darwin'
        import torchaudio

        self.transforms = torchaudio.transforms.Spectrogram(n_fft=256, win_length=128, hop_length=64)

        self.target_size = int(frame_length * self.feature_sample_rate)
        self.target_shift = frame_shift * self.feature_sample_rate
        self.seiz_requirement_pts = self.target_size // 5
        self.final_target_batch = []

    def psd(self, amp, begin, end):
        return torch.mean(amp[begin * self.freq_resolution: end * self.freq_resolution], dim = 0)

    def downsample_target(self, targets):
        win_size = self.target_size
        win_shift = self.target_shift
        target_list = []
        num_interval = int(targets.size(0) // self.target_shift) + 1

        for idx in range(0, num_interval):
            index = int(self.target_shift * idx)
            seiz_count = torch.count_nonzero(targets[index: index + win_size], dim=0)
            if seiz_count < self.seiz_requirement_pts:
                target_list.append(0)
            else:
                target_list.append(1)

        return target_list

    def forward(self, batch, targets, seq_lengths, target_lengths):
        self.final_batch = []
        self.final_target_batch = []

        for idx, signals in enumerate(batch):
            target = targets[idx][:target_lengths[idx]]
            signals = signals[:,:seq_lengths[idx]]
            transformed_sample = []

            for signal in signals:
                stft = self.transforms(signal)
                amp = (torch.log(torch.abs(stft) + 1e-10))[1:self.freq_max, :]
                transformed_sample.append(amp)

            tensor_feature = torch.stack(transformed_sample).permute(1,0,2)
            tensor_feature = tensor_feature.reshape(tensor_feature.size(0), -1)

            self.final_batch.append(tensor_feature.cpu().numpy())
            target_list = self.downsample_target(target)
            self.final_target_batch.append(target_list)

        return torch.tensor(np.array(self.final_batch), dtype=torch.float32), self.final_target_batch
