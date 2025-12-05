import numpy as np
import math

from scipy.signal import butter, filtfilt, hilbert


def _hilbert_norm(signal):
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    normalized_signal = signal / amplitude_envelope
    return normalized_signal

def _butterworth_filter(signal, order):
    low = 0.75
    high = 2.5
    fs = 30
    [b, a] = butter(N=order, Wn=[low / fs * 2, high / fs * 2], btype='bandpass')
    filtered_signal = filtfilt(b, a, signal.astype(np.double))
    return filtered_signal

def _process_video(frames):
    """Calculates the average value of each frame."""
    RGB = []
    for frame in frames:
        summation = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(summation / (frame.shape[0] * frame.shape[1]))
    return np.asarray(RGB)


def POS_VITALSYNC(config, frames, fs):
    WinSec = 1.6
    RGB = _process_video(frames)
    N = RGB.shape[0]
    H = np.zeros((1, N))
    l = math.ceil(WinSec * fs)

    for n in range(N):
        m = n - l
        if m >= 0:
            Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
            Cn = np.mat(Cn).H
            S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
            h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
            mean_h = np.mean(h)
            for temp in range(h.shape[1]):
                h[0, temp] = h[0, temp] - mean_h
            H[0, m:n] = H[0, m:n] + (h[0])

    bvp = H.squeeze()

    return bvp
