import numpy as np
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from scipy.io.wavfile import read

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, hparams, center=False, complex=False):
    # hop_size: 512  # For 22050Hz, 275 ~= 12.5 ms (0.0125 * sample_rate)
    # win_size: 2048  # For 22050Hz, 1100 ~= 50 ms (If None, win_size: fft_size) (0.05 * sample_rate)
    # fmin: 55  # Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
    # fmax: 10000  # To be increased/reduced depending on data.
    # fft_size: 2048  # Extra window size is filled with 0 paddings to match this parameter
    # n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax,
    n_fft = hparams["fft_size"]
    num_mels = hparams["audio_num_mel_bins"]
    sampling_rate = hparams["audio_sample_rate"]
    hop_size = hparams["hop_size"]
    win_size = hparams["win_size"]
    fmin = hparams["fmin"]
    fmax = hparams["fmax"]
    y = y.clamp(min=-1.0, max=1.0)
    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
    )

    if not complex:
        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
        spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
        spec = spectral_normalize_torch(spec)
    else:
        B, C, T, _ = spec.shape
        spec = spec.transpose(1, 2)  # [B, T, n_fft, 2]
    return spec
