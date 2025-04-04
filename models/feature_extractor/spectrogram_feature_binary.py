# import platform
# import torch
# import torch.nn as nn
#
# # this spectrogram setting is specifically for alexnet eeg paper
#
# class SPECTROGRAM_FEATURE_BINARY(nn.Module):
#     def __init__(self,
#                  sample_rate: int = 200,
#                  frame_length: int = 16,
#                  frame_shift: int = 8,
#                  feature_extract_by: str = 'kaldi'):
#         super(SPECTROGRAM_FEATURE_BINARY, self).__init__()
#
#         self.sample_rate = sample_rate
#         self.feature_extract_by = feature_extract_by.lower()
#         self.freq_resolution = 1
#
#         if self.feature_extract_by == 'kaldi':
#             assert platform.system().lower() == 'linux' or platform.system().lower() == 'darwin'
#             import torchaudio
#
#             self.transforms = torchaudio.transforms.Spectrogram(n_fft=self.freq_resolution * self.sample_rate,
#                                                                 win_length=frame_length,
#                                                                 hop_length=frame_shift)
#
#         else:
#             self.n_fft = self.freq_resolution * self.sample_rate
#             self.hop_length = frame_shift
#             self.frame_length = frame_length
#
#     def forward(self, batch):
#         spectrogram_batch = []
#
#         for signals in batch:
#             spectrogram_sample = []
#             for signal in signals:
#                 if self.feature_extract_by == 'kaldi':
#                     stft = self.transforms(signal)
#                     amp = (torch.log(torch.abs(stft) + 1e-10))
#
#                 else:
#                     stft = torch.stft(
#                         signal, self.n_fft, hop_length=self.hop_length,
#                         win_length=self.frame_length, window=torch.hamming_window(self.frame_length),
#                         center=False, normalized=False, onesided=True
#                     )
#
#                     amp = (torch.log(torch.abs(stft) + 1e-10))
#
#                 spectrogram_sample.append(amp[:50, :])
#
#             spectrogram_batch.append(torch.stack(spectrogram_sample))
#
#         return torch.stack(spectrogram_batch)