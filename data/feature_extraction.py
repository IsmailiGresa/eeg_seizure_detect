import numpy as np
import math
from scipy.spatial.distance import pdist
from scipy.signal import stft, hilbert


def fractal_feature(signal, signal_samplerate, feature_samplerate):
    result = []
    a = int(signal_samplerate / float(feature_samplerate))

    for start_point in range(0, len(signal), signal_samplerate):

        onesec_signal = list(signal[start_point:start_point + signal_samplerate])

        for start_point in range(0, len(onesec_signal), a):
            one_signal = list(onesec_signal[start_point:start_point + a])
            e = 0.00000001
            time_tv = 1 / float(signal_samplerate)
            signal_size = len(one_signal)
            old_point = [0, 0]
            time = 0
            length = 0

            for k in range(signal_size):
                time += time_tv
                new_point = [time, one_signal[k]]
                d = pdist([old_point, new_point], metric = 'euclidean')
                length += d
                old_point = new_point
            Nprime = 1 / (2 * e)

            result.append(1 + math.log(length, 2) / math.log(2 * Nprime, 2))

    return np.asarray(result)


def power_spectral_density_feature(signal, samplerate, new_length):
    freq_resolution = 2

    def psd(amp, begin, end, freq_resol=freq_resolution):
        return np.average(amp[begin * freq_resol:end * freq_resol], axis = 0)

    n_per_seg = 8
    n_overlap = 4

    nfft = samplerate * freq_resolution
    freq, times, spec = stft(signal, samplerate, nperseg = n_per_seg, noverlap = n_overlap, nfft = nfft, padded = True, boundary = 'zeros')
    amp = (np.log(np.abs(spec) + 1e-10))

    new_length = int(new_length)
    if abs(amp.shape[1] - new_length) > 1:
        print("Difference is huge {} {}".format(amp.shape[1], new_length))
    amp = amp[:, :new_length]

    psds = []

    if samplerate == 256:
        psd1 = psd(amp, 0, 4)
        psd2 = psd(amp, 4, 8)
        psd3 = psd(amp, 8, 13)
        psd4 = psd(amp, 13, 20)
        psd5 = psd(amp, 20, 30)
        psd6 = psd(amp, 30, 40)
        psd7 = psd(amp, 40, 60)
        psd8 = psd(amp, 60, 80)
        psd9 = psd(amp, 80, 100)
        psd10 = psd(amp, 100, 128)
        psds = [psd1, psd2, psd3, psd4, psd5, psd6, psd7, psd8, psd9, psd10]
    elif samplerate == 1024:
        psd1 = psd(amp, 0, 4)
        psd2 = psd(amp, 4, 8)
        psd3 = psd(amp, 8, 13)
        psd4 = psd(amp, 13, 20)
        psd5 = psd(amp, 20, 30)
        psd6 = psd(amp, 30, 40)
        psd7 = psd(amp, 40, 60)
        psd8 = psd(amp, 60, 80)
        psd9 = psd(amp, 80, 100)
        psd10 = psd(amp, 100, 128)
        psd11 = psd(amp, 128, 256)
        psd12 = psd(amp, 256, 512)
        psds = [psd1, psd2, psd3, psd4, psd5, psd6, psd7, psd8, psd9, psd10, psd11, psd12]
    elif samplerate == 200:
        psd1 = psd(amp, 0, 4)
        psd2 = psd(amp, 4, 8)
        psd3 = psd(amp, 8, 13)
        psd4 = psd(amp, 13, 20)
        psd5 = psd(amp, 20, 30)
        psd6 = psd(amp, 30, 40)
        psd7 = psd(amp, 40, 50)
        psd8 = psd(amp, 50, 64)
        psd9 = psd(amp, 64, 80)
        psd10 = psd(amp, 80, 100)
        psds = [psd1, psd2, psd3, psd4, psd5, psd6, psd7, psd8, psd9, psd10]
    else:
        print("Select correct sample rate!")
        exit(1)

    return psds

def spectrogram_feature(signal, samplerate, feature_samplerate):
    freq_resolution = 2
    n_per_seg = 8
    n_overlap = 150
    nfft = int(samplerate * freq_resolution)
    freq, times, spec = stft(signal, samplerate, nperseg = n_per_seg, noverlap = n_overlap, nfft = nfft, padded = False, boundary = None)
    amp = (np.log(np.abs(spec) + 1e-10))
    return freq, times, amp

def hilbert_envelope_feature(signal, n):
    return hilbert(signal, N = n)