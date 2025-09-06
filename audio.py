"""
Shared audio utilities using librosa and related tools.
Includes caching to avoid recalculating features multiple times.
All audio is normalized to [0, 1].

# Caching Notes

1. If the audio file is inside a hidden folder, the cache will be stored in the parent directory.
   This avoids syncing large media files (e.g., VastAI) and only ships small cache files.

2. If the audio file itself is hidden, the cache will be unhidden.
   Cached feature files are small and meant to be accessed regularly.
"""

import logging
from pathlib import Path

import soundfile
import numpy as np
import librosa
import resampy

log = logging.getLogger("audio")

def norm(data):
    """Scale values into [0, 1]."""
    min_val, max_val = np.min(data), np.max(data)
    return (data - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(data)

def zero():
    return np.array([])

def load_audio_cache(filepath, name, fps):
    path = Path(filepath)
    cache = get_audio_cachepath(path, name, fps)
    if cache.exists():
        return np.load(cache.as_posix())
    raise FileNotFoundError(f"Cache file not found: {cache}")

def get_audio_cachepath(filepath, name, fps):
    filepath = Path(filepath)
    if filepath.parent.stem.startswith("."):
        filepath = filepath.parent.parent / filepath.name
    if filepath.name.startswith("."):
        filepath = filepath.with_name(filepath.name[1:])
    return filepath.with_stem(f"{filepath.stem}_{name}_{fps}").with_suffix(".npy")

def save_audio_cache(filepath, name, arr, enabled, fps):
    cache = get_audio_cachepath(filepath, name, fps)
    if enabled:
        np.save(cache.as_posix(), arr)
    return arr

def has_audio_cache(filepath, name, enabled, fps):
    cache = get_audio_cachepath(filepath, name, fps)
    if enabled and not cache.exists():
        log.info(f"audio.{name}({Path(filepath).stem}): no cache found, recalculating...")
    return enabled and cache.exists()

def load_crepe_keyframes(filepath, fps=60):
    import pandas as pd
    df = pd.read_csv(filepath)
    freq = to_keyframes(df["frequency"], len(df["frequency"]) / df["time"].values[-1], fps)
    conf = to_keyframes(df["confidence"], len(df["frequency"]) / df["time"].values[-1], fps)
    return freq, conf

def load_rosa(filepath, fps=60):
    y, sr = soundfile.read(filepath)
    y = librosa.to_mono(y.T)

    duration = librosa.get_duration(y=y, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    orig_fps = len(onset_env) / duration
    onset_resampled = resampy.resample(onset_env, orig_fps, fps)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_marks = np.zeros_like(onset_env)
    beat_marks[beat_frames] = 1
    beat_resampled = resampy.resample(beat_marks, orig_fps, fps)

    y_harm, y_perc = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)[1]
    chroma_resampled = resampy.resample(chroma, orig_fps, fps)

    S = np.abs(librosa.stft(y))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)[1]
    contrast_resampled = resampy.resample(contrast, orig_fps, fps)

    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_resampled = resampy.resample(mfcc[1], orig_fps, fps)

    freqs, times, D = librosa.reassigned_spectrogram(y, fill_nan=True)
    bandwidth = librosa.feature.spectral_bandwidth(S=np.abs(D), freq=freqs)[0]
    bandwidth_resampled = resampy.resample(bandwidth, orig_fps, fps)

    flatness = librosa.feature.spectral_flatness(y=y)[0]
    flatness_resampled = resampy.resample(flatness, orig_fps, fps)

    sentiment = happy_sad(filepath)

    return (
        zero(),
        onset_resampled,
        beat_resampled,
        chroma_resampled,
        contrast_resampled,
        mfcc_resampled,
        flatness_resampled,
        bandwidth_resampled,
        sentiment,
    )

def load_lufs(filepath, caching=True, fps=60):
    import soundfile as sf
    from loudness import lufs_meter
    if not has_audio_cache(filepath, "lufs", caching, fps):
        y, sr = sf.read(filepath)
        meter = lufs_meter(sr, 1/60, overlap=0)
        loudness = meter.get_mlufs(y)
        loudness[np.isinf(loudness)] = 0
        threshold = -5
        loudness = np.where(loudness > threshold, meter.threshold, loudness)
        loudness = norm(loudness)
        loudness = resampy.resample(loudness, 60, fps)
        return save_audio_cache(filepath, "lufs", loudness, caching, fps)
    return load_audio_cache(filepath, "lufs", fps)

def load_pca(filepath, num_components=3, caching=True, fps=60):
    from sklearn.decomposition import PCA
    if not has_audio_cache(filepath, "pca", caching, fps):
        y, sr = librosa.load(filepath)
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, hop_length=int(sr/fps), win_length=int(sr*0.03), n_chroma=12
        )
        pca = PCA(n_components=num_components)
        chroma_pca = pca.fit_transform(chroma.T).T
        duration = librosa.get_duration(y=y, sr=sr)
        orig_fps = len(chroma_pca[0]) / duration
        ret = [resampy.resample(chroma_pca[i], orig_fps, fps) for i in range(num_components)]
        save_audio_cache(filepath, "pca", np.array(ret), caching, fps)
        return tuple(ret)
    arr = load_audio_cache(filepath, "pca", fps)
    return tuple(arr[i] for i in range(num_components))

def load_flatness(filepath, caching=True, fps=60):
    if not has_audio_cache(filepath, "flatness", caching, fps):
        y, sr = librosa.load(filepath)
        flatness = librosa.feature.spectral_flatness(
            y=y, hop_length=int(sr/fps), win_length=int(sr*0.03)
        )[0]
        duration = librosa.get_duration(y=y, sr=sr)
        orig_fps = len(flatness) / duration
        resampled = resampy.resample(flatness, orig_fps, fps)
        return save_audio_cache(filepath, "flatness", resampled, caching, fps)
    return load_audio_cache(filepath, "flatness", fps)

def load_onset(filepath, caching=True, fps=60):
    if not has_audio_cache(filepath, "onset", caching, fps):
        y, sr = librosa.load(filepath)
        onset = librosa.onset.onset_strength(y=y, sr=sr)
        onset = norm(onset)
        duration = librosa.get_duration(y=y, sr=sr)
        orig_fps = len(onset) / duration
        resampled = resampy.resample(onset, orig_fps, fps)
        return save_audio_cache(filepath, "onset", resampled, caching, fps)
    return load_audio_cache(filepath, "onset", fps)

def happy_sad(filepath, fps=60):
    y, sr = soundfile.read(filepath)
    y = librosa.to_mono(y.T)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    major = chroma[[0,4,7],:]
    minor = chroma[[0,3,7],:]
    scores = np.sum(major, axis=0) - np.sum(minor, axis=0)
    scores /= (np.sum(major, axis=0) + np.sum(minor, axis=0))
    duration = librosa.get_duration(y=y, sr=sr)
    orig_fps = len(scores) / duration
    return resampy.resample(scores, orig_fps, fps)

def to_keyframes(dbs, orig_sps, fps=60):
    total_sec = len(dbs) / orig_sps
    frames = int(fps * total_sec)
    dt = np.zeros(frames)
    for i in range(frames):
        t0, t1 = i/fps, (i+1)/fps
        d = dbs[int(t0*orig_sps): int(t1*orig_sps)]
        dt[i] = np.mean(d)
        if np.isinf(dt[i]) or np.isnan(dt[i]):
            dt[i] = dt[i-1]
    return dt
