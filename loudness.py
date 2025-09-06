# Audio loudness measurement and normalization.
# LUFS calculations are based on:
# ITU spec: https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-4-201510-I!!PDF-E.pdf
# EBU spec: https://tech.ebu.ch/docs/tech/tech3341.pdf
# pyloudnorm by csteinmetz1: https://github.com/csteinmetz1/pyloudnorm
# loudness.py by BrechtDeMan: https://github.com/BrechtDeMan/loudness.py
# Thanks to these authors. This code adapts their ideas for short-term and momentary loudness
# plus batch normalization of audio files.

import numpy as np
from scipy import signal

def amp2db(amp: float):
    """Convert amplitude [0,1] to decibels."""
    return 20 * np.log10(amp)

def db2amp(db: float):
    """Convert decibels (≤0) to amplitude."""
    return np.power(10, db / 20)

def change_vol(audio, db_change):
    return audio * db2amp(db_change)

def check_clipping(audio):
    assert np.amax(np.abs(audio)) < 1, "Clipping detected."

def recompare(original, reconstructed, sr=None):
    print("Comparison after reconstruction:")
    if sr:
        diff = (reconstructed.shape[0] - original.shape[0]) / sr
        print(f"Length difference: {round(diff, 4)}s")
    max_err = amp2db(np.amax(np.abs(reconstructed[: original.shape[0]] - original)))
    print(f"Max error: {round(max_err, 4)} dB")

def print_peak(audio):
    peak_amp = np.amax(np.abs(audio))
    peak_db = amp2db(peak_amp)
    print(f"peak_amp = {round(peak_amp,5)}, peak_db = {round(peak_db,2)}")

def get_peak(audio):
    return np.amax(np.abs(audio))

def get_peak_lr(audio):
    peak_lr = np.amax(np.abs(audio), axis=0)
    return peak_lr[0], peak_lr[1]

def norm_peak_mid(audio, peak_db=-12.0):
    """Normalize stereo mid channel peak amplitude under -3dB pan law."""
    scale = db2amp(peak_db) / np.amax(np.abs(np.mean(audio, axis=-1)))
    return audio * scale

def norm_peak_mono(audio, peak_db=-12.0):
    """Normalize mono audio peak amplitude."""
    scale = db2amp(peak_db) / np.amax(np.abs(audio))
    return audio * scale

class LUFSMeter:
    """LUFS calculator for momentary and integrated loudness."""
    def __init__(self, sr, T=0.4, overlap=0.75, threshold=-70.0, start=None):
        """
        sr: sample rate (Hz)
        T: window length in seconds (use 3 for short-term LUFS)
        overlap: fraction of overlap between windows
        threshold: LUFS floor (values below reported as -inf)
        start: start time (s) for analysis
        """
        self.sr, self.T, self.overlap, self.threshold, self.start = sr, T, overlap, threshold, start
        self.step = int(sr * T)
        self.hop = int(sr * T * (1 - overlap))
        self.z_thresh = np.power(10, (threshold + 0.691) / 10)
        self.n_start = int(sr * start) if start else None

        if sr == 48000:
            self.sos = np.array([
                [1.53512485958697, -2.69169618940638, 1.19839281085285, 1.0, -1.69065929318241, 0.73248077421585],
                [1.0, -2.0, 1.0, 1.0, -1.99004745483398, 0.99007225036621]
            ])
        elif sr == 44100:
            self.sos = np.array([
                [1.5308412300498355, -2.6509799951536985, 1.1690790799210682, 1.0, -1.6636551132560204, 0.7125954280732254],
                [1.0, -2.0, 1.0, 1.0, -1.9891696736297957, 0.9891990357870394]
            ])
        else:
            f0, G, Q = 1681.9744509555319, 3.99984385397, 0.7071752369554193
            K = np.tan(np.pi * f0 / sr)
            Vh = np.power(10.0, G / 20.0)
            Vb = np.power(Vh, 0.499666774155)
            a0 = 1.0 + K / Q + K**2
            b0 = (Vh + Vb * K / Q + K**2) / a0
            b1 = 2.0 * (K**2 - Vh) / a0
            b2 = (Vh - Vb * K / Q + K**2) / a0
            a1 = 2.0 * (K**2 - 1.0) / a0
            a2 = (1.0 - K / Q + K**2) / a0
            f0 = 38.13547087613982
            Q = 0.5003270373253953
            K = np.tan(np.pi * f0 / sr)
            a1_2 = 2.0 * (K**2 - 1.0) / (1.0 + K / Q + K**2)
            a2_2 = (1.0 - K / Q + K**2) / (1.0 + K / Q + K**2)
            self.sos = np.array([
                [b0, b1, b2, 1.0, a1, a2],
                [1.0, -2.0, 1.0, 1.0, a1_2, a2_2]
            ])

    def get_mlufs(self, audio):
        """Momentary LUFS sequence."""
        if self.n_start:
            audio = audio[: self.n_start]
        q1, q2 = divmod(audio.shape[0], self.hop)
        pad_len = self.step - self.hop - q2
        if pad_len > 0:
            pad_shape = list(audio.shape)
            pad_shape[0] = pad_len
            audio = np.append(audio, np.zeros(pad_shape), axis=0)
        Mlufs = []
        for i in range(q1):
            segment = audio[i * self.hop : i * self.hop + self.step]
            filtered = signal.sosfilt(self.sos, segment, axis=0)
            z = np.sum(np.mean(np.square(filtered), axis=0))
            if z < self.z_thresh:
                Mlufs.append(float("-inf"))
            else:
                Mlufs.append(-0.691 + 10 * np.log10(z))
        return np.array(Mlufs)

    def get_mlufs_max(self, audio):
        return np.amax(self.get_mlufs(audio))

    def get_ilufs(self, audio):
        """Integrated LUFS value."""
        Mlufs = self.get_mlufs(audio)
        Z = np.power(10, (Mlufs + 0.691) / 10)
        Z0 = Z[Mlufs > -70.0]
        if Z0.size == 0:
            return float("-inf")
        z1 = np.mean(Z0)
        Z = Z[Mlufs > -0.691 + 10 * np.log10(z1) - 10]
        if Z.size == 0:
            return float("-inf")
        z2 = np.mean(Z)
        return -0.691 + 10 * np.log10(z2) if z2 >= self.z_thresh else float("-inf")

    def norm_mlufs_max(self, audio, target=-20.0):
        return audio * db2amp(target - self.get_mlufs_max(audio))

    def norm_ilufs(self, audio, target=-23.0):
        return audio * db2amp(target - self.get_ilufs(audio))

    def print_mlufs_max(self, audio):
        print(f"mlufs_max = {round(self.get_mlufs_max(audio), 4)} LUFS")

    def print_ilufs(self, audio):
        print(f"ilufs = {round(self.get_ilufs(audio), 4)} LUFS")
