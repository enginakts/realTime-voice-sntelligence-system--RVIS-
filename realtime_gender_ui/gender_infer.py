from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

from audio_features import FeatureConfig, rms_db, rms_normalize, wav_to_logmel


@dataclass(frozen=True)
class RealtimeConfig:
    win_sec: float = 1.0
    hop_sec: float = 0.5
    clip_sec: float = 3.0  # model eğitimi 3s sabitleme ile yapıldı
    silence_rms_db: float = -45.0  # daha yüksek => daha agresif filtre


LABELS = {0: "Male", 1: "Female"}


def _to_3sec_center(y: np.ndarray, sr: int, clip_sec: float) -> np.ndarray:
    target = int(sr * clip_sec)
    if len(y) < target:
        return np.pad(y, (0, target - len(y))).astype(np.float32)
    start = (len(y) - target) // 2
    return y[start : start + target].astype(np.float32)


def infer_window(
    model: torch.nn.Module,
    y_win: np.ndarray,
    feat_cfg: FeatureConfig,
    rt_cfg: RealtimeConfig,
    device: str,
) -> Dict:
    """
    Tek pencere için:
    - sessizlik filtreleme (RMS dB)
    - 3sn'e merkezden pad/crop (notebook test mantığı)
    - log-mel -> model -> softmax
    """
    y_win = y_win.astype(np.float32, copy=False)
    db = rms_db(y_win)
    if db < rt_cfg.silence_rms_db:
        return {
            "is_silence": True,
            "rms_db": db,
            "label": None,
            "probs": None,
        }

    y = rms_normalize(y_win, target=feat_cfg.rms_target)
    y = _to_3sec_center(y, feat_cfg.target_sr, rt_cfg.clip_sec)
    feat = wav_to_logmel(y, feat_cfg)  # (80, T)
    x = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0)  # (1,1,80,T)
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy().reshape(-1)
        pred = int(np.argmax(probs))
    return {
        "is_silence": False,
        "rms_db": db,
        "label": LABELS.get(pred, str(pred)),
        "probs": {"Male": float(probs[0]), "Female": float(probs[1])},
    }

