from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import librosa


@dataclass(frozen=True)
class FeatureConfig:
    target_sr: int = 16000
    n_mels: int = 80
    hop_length: int = 160  # 10ms @16k
    win_length: int = 400  # 25ms @16k
    fmin: int = 50
    fmax: int = 7600
    rms_target: float = 0.06


def rms_normalize(y: np.ndarray, target: float = 0.06) -> np.ndarray:
    y = y.astype(np.float32, copy=False)
    rms = np.sqrt(np.mean(y**2) + 1e-12)
    gain = float(target) / float(rms + 1e-12)
    y = y * gain
    y = np.clip(y, -1.0, 1.0)
    return y.astype(np.float32, copy=False)


def wav_to_logmel(y: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    s = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.target_sr,
        n_mels=cfg.n_mels,
        n_fft=cfg.win_length,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0,
    )
    log_s = librosa.power_to_db(s, ref=np.max).astype(np.float32)  # (n_mels, T)
    return log_s


def rms_db(y: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(y.astype(np.float32))) + 1e-12))
    return float(20.0 * np.log10(rms + 1e-12))


def pad_or_crop_center(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y))).astype(np.float32)
    start = (len(y) - target_len) // 2
    return y[start : start + target_len].astype(np.float32)


def sliding_windows(
    y: np.ndarray, sr: int, win_sec: float, hop_sec: float
) -> Iterable[Tuple[int, int, np.ndarray]]:
    win = int(win_sec * sr)
    hop = int(hop_sec * sr)
    if win <= 0 or hop <= 0:
        raise ValueError("win_sec ve hop_sec > 0 olmalı")
    if len(y) < win:
        return
    for start in range(0, len(y) - win + 1, hop):
        end = start + win
        yield start, end, y[start:end]


def concat_with_silence(
    signals: List[np.ndarray], sr: int, silence_sec: float = 5.0
) -> np.ndarray:
    if not signals:
        return np.zeros((0,), dtype=np.float32)
    silence = np.zeros((int(sr * silence_sec),), dtype=np.float32)
    out = []
    for i, s in enumerate(signals):
        out.append(s.astype(np.float32, copy=False))
        if i != len(signals) - 1:
            out.append(silence)
    return np.concatenate(out, axis=0).astype(np.float32)


def remove_silence_by_rms_sliding(
    y: np.ndarray,
    sr: int,
    win_sec: float,
    hop_sec: float,
    silence_rms_db: float,
    keep_padding_sec: float = 0.05,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Sliding-window RMS dB tabanlı sessizlik kesme.

    - Pencerelerin RMS(dB) < silence_rms_db ise "sessiz" kabul edilir.
    - Sesli pencerelerden oluşan aralıklar birleştirilir (merge).
    - Her sesli aralığın başına/sonuna küçük bir padding eklenir (default 50ms).

    Dönüş:
      trimmed_y: sessizlikleri kesilmiş sinyal
      segments: her segment için mapping bilgisi
        {
          "seg_idx": int,
          "orig_start": int, "orig_end": int,          # sample index (orig y)
          "trim_start": int, "trim_end": int,          # sample index (trimmed_y)
          "orig_start_sec": float, "orig_end_sec": float,
          "trim_start_sec": float, "trim_end_sec": float,
        }
    """
    y = y.astype(np.float32, copy=False)
    win = int(win_sec * sr)
    hop = int(hop_sec * sr)
    pad = int(max(0.0, keep_padding_sec) * sr)
    if win <= 0 or hop <= 0:
        raise ValueError("win_sec ve hop_sec > 0 olmalı")
    if len(y) < win:
        return y.copy(), [
            {
                "seg_idx": 0,
                "orig_start": 0,
                "orig_end": len(y),
                "trim_start": 0,
                "trim_end": len(y),
                "orig_start_sec": 0.0,
                "orig_end_sec": len(y) / sr,
                "trim_start_sec": 0.0,
                "trim_end_sec": len(y) / sr,
            }
        ]

    voiced_ranges: List[Tuple[int, int]] = []
    for start, end, y_win in sliding_windows(y, sr, win_sec, hop_sec):
        if rms_db(y_win) >= silence_rms_db:
            a = max(0, start - pad)
            b = min(len(y), end + pad)
            voiced_ranges.append((a, b))

    if not voiced_ranges:
        return np.zeros((0,), dtype=np.float32), []

    # merge overlaps
    voiced_ranges.sort(key=lambda t: t[0])
    merged: List[Tuple[int, int]] = []
    cur_a, cur_b = voiced_ranges[0]
    for a, b in voiced_ranges[1:]:
        if a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            merged.append((cur_a, cur_b))
            cur_a, cur_b = a, b
    merged.append((cur_a, cur_b))

    # concatenate + mapping
    trimmed_chunks = []
    segments: List[Dict] = []
    trim_cursor = 0
    for i, (a, b) in enumerate(merged):
        chunk = y[a:b]
        trimmed_chunks.append(chunk)
        trim_start = trim_cursor
        trim_end = trim_cursor + len(chunk)
        segments.append(
            {
                "seg_idx": i,
                "orig_start": int(a),
                "orig_end": int(b),
                "trim_start": int(trim_start),
                "trim_end": int(trim_end),
                "orig_start_sec": float(a) / sr,
                "orig_end_sec": float(b) / sr,
                "trim_start_sec": float(trim_start) / sr,
                "trim_end_sec": float(trim_end) / sr,
            }
        )
        trim_cursor = trim_end

    trimmed_y = np.concatenate(trimmed_chunks, axis=0).astype(np.float32)
    return trimmed_y, segments

