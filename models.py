import rpca
import librosa
import numpy as np


def separate_signal_with_RPCA(M, improve=False):
    # Short-Time Fourier Transformation
    M_stft = librosa.stft(M, n_fft=1024, hop_length=256)
    # Get magnitude and phase
    M_mag, M_phase = librosa.magphase(M_stft)
    # RPCA
    L_hat, S_hat, r_hat, n_iter_hat = rpca.pcp_alm(M_mag)
    # Append phase back to result
    L_output = np.multiply(L_hat, M_phase)
    S_output = np.multiply(S_hat, M_phase)

    if improve:
        L_hat, S_hat, r_hat, n_iter_hat = rpca.pcp_alm(np.abs(S_output))
        S_output = np.multiply(S_hat, M_phase)
        L_output = M_stft - S_output

    return M_stft, L_output, S_output


def time_freq_masking(M_stft, L_hat, S_hat, gain):
    mask = np.abs(S_hat) - gain * np.abs(L_hat)
    mask = (mask > 0) * 1
    X_sing = np.multiply(mask, M_stft)
    X_music = np.multiply(1 - mask, M_stft)
    return X_sing, X_music
