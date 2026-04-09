from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from dataclasses import asdict
from typing import List, Optional

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import numpy as np
import pandas as pd
import streamlit as st
try:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer, AudioProcessorBase
    _HAS_WEBRTC = True
except Exception:  # ImportError dahil
    WebRtcMode = None  # type: ignore
    webrtc_streamer = None  # type: ignore
    AudioProcessorBase = object  # type: ignore
    _HAS_WEBRTC = False

import librosa
import soundfile as sf
import torch

from audio_features import (
    FeatureConfig,
    concat_with_silence,
    remove_silence_by_rms_sliding,
    sliding_windows,
)
from gender_infer import RealtimeConfig, infer_window
from model_gender_cnn_bilstm import load_gender_model


DEFAULT_MODEL_PATH = str((Path(__file__).resolve().parents[1] / "code" / "models" / "model.pt"))

# `av` sadece Senaryo 1 (webrtc audio frame) için gerekli.
try:
    import av  # type: ignore
    _HAS_AV = True
except Exception:
    av = None  # type: ignore
    _HAS_AV = False


st.set_page_config(page_title="Realtime Gender Classifier", layout="wide")
st.title("Realtime Gender Classifier (CNN+BiLSTM)")
st.caption("Senaryo 1: Canlı mikrofon • Senaryo 2: Birleştirilmiş ses ile pencere bazlı analiz")


@st.cache_resource
def _load_model_cached(path: str, device: str):
    return load_gender_model(path, device=device)


def _ensure_float_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    return x.astype(np.float32, copy=False)


class RealtimeGenderAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.cfg = FeatureConfig()
        self.rt = RealtimeConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

        self._buf = np.zeros((0,), dtype=np.float32)
        self._last_emit = 0.0
        self._results: List[dict] = []

        self._win = int(self.cfg.target_sr * self.rt.win_sec)
        self._hop = int(self.cfg.target_sr * self.rt.hop_sec)

    def set_runtime(self, model_path: str, silence_db: float, win_sec: float, hop_sec: float):
        self.rt = RealtimeConfig(
            win_sec=float(win_sec),
            hop_sec=float(hop_sec),
            clip_sec=3.0,
            silence_rms_db=float(silence_db),
        )
        self._win = int(self.cfg.target_sr * self.rt.win_sec)
        self._hop = int(self.cfg.target_sr * self.rt.hop_sec)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = _load_model_cached(model_path, self.device)

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        # streamlit-webrtc: AudioFrame -> numpy
        pcm = frame.to_ndarray()  # shape: (channels, samples) or (samples,)
        if pcm.ndim == 2:
            pcm = pcm.mean(axis=0)
        pcm = pcm.astype(np.float32)

        # normalize int16 PCM if needed
        if np.max(np.abs(pcm)) > 2.0:
            pcm = pcm / 32768.0

        # resample to 16k if frame.sample_rate differs
        sr_in = int(frame.sample_rate)
        if sr_in != self.cfg.target_sr:
            pcm = librosa.resample(pcm, orig_sr=sr_in, target_sr=self.cfg.target_sr)

        # buffer
        self._buf = np.concatenate([self._buf, pcm], axis=0)

        # emit predictions while we have enough audio
        while len(self._buf) >= self._win:
            y_win = self._buf[: self._win]
            self._buf = self._buf[self._hop :]

            if self.model is None:
                break
            out = infer_window(self.model, y_win, self.cfg, self.rt, self.device)
            out["t"] = time.time()
            self._results.append(out)
            # keep last N
            if len(self._results) > 200:
                self._results = self._results[-200:]

        return frame

    def pop_results(self) -> List[dict]:
        r = self._results[:]
        self._results.clear()
        return r


with st.sidebar:
    st.header("Ayarlar")
    model_path = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
    silence_db = st.slider("Sessizlik eşiği (RMS dB)", min_value=-80.0, max_value=-10.0, value=-45.0, step=1.0)
    win_sec = st.selectbox("Pencere (sn)", options=[0.5, 1.0, 1.5, 2.0], index=1)
    hop_sec = st.selectbox("Hop (sn)", options=[0.25, 0.5, 1.0], index=1)
    st.divider()
    st.write("Not: Çok küçük hop gecikmeyi azaltır ama CPU yükünü artırır.")


tabs = st.tabs(["Senaryo 1 • Canlı Mikrofon", "Senaryo 2 • Birleştirilmiş Ses Dosyası"])


with tabs[0]:
    st.subheader("Senaryo 1 — Canlı Mikrofon ile Sürekli Tahmin")
    st.write(
        "Mikrofondan gelen akış 16 kHz'e çevrilir. Kaydırmalı pencerede sessiz pencereler filtrelenir; "
        "sesli pencerelerde model tahmini anlık gösterilir."
    )

    if not _HAS_WEBRTC or not _HAS_AV:
        st.error(
            "Bu ortamda mikrofon için gerekli paketler bulunamadı (`streamlit-webrtc` ve/veya `av`). "
            "Mikrofon senaryosu devre dışı.\n\n"
            "Çözüm (Anaconda):\n"
            "- `conda activate <env>`\n"
            "- `pip install streamlit-webrtc av aiortc`\n\n"
            "Sonra tekrar: `streamlit run app.py`"
        )
        st.stop()

    colA, colB = st.columns([2, 1], gap="large")
    with colA:
        ctx = webrtc_streamer(
            key="gender-mic",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=RealtimeGenderAudioProcessor,
            media_stream_constraints={"audio": True, "video": False},
        )

        if ctx.audio_processor:
            ctx.audio_processor.set_runtime(model_path, silence_db, win_sec, hop_sec)

    with colB:
        st.markdown("#### Anlık çıktı")
        placeholder = st.empty()
        history = st.session_state.get("mic_history", [])

        if ctx.audio_processor:
            new = ctx.audio_processor.pop_results()
            if new:
                history.extend(new)
                history = history[-200:]
                st.session_state["mic_history"] = history

        if history:
            # show last 15
            rows = []
            for r in history[-15:][::-1]:
                if r["is_silence"]:
                    rows.append({"label": "Silence", "rms_db": r["rms_db"], "p_female": None, "p_male": None})
                else:
                    rows.append(
                        {
                            "label": r["label"],
                            "rms_db": r["rms_db"],
                            "p_male": r["probs"]["Male"],
                            "p_female": r["probs"]["Female"],
                        }
                    )
            df = pd.DataFrame(rows)
            placeholder.dataframe(df, use_container_width=True, height=420)
        else:
            placeholder.info("Mikrofonu başlatınca burada tahminler görünecek.")


with tabs[1]:
    st.subheader("Senaryo 2 — Birleştirilmiş Ses Dosyası ile Pencere Bazlı Analiz")
    st.write(
        "10 kadın + 10 erkek ses dosyasını yükle. Her dosya uç uca eklenir, araya 5s sessizlik koyulur. "
        "Tek akış üzerinde gerçek-zamanlı pencere analizi yapılır."
    )

    up = st.file_uploader(
        "Ses dosyaları (wav/mp3/m4a/...)",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        accept_multiple_files=True,
    )
    silence_sec = st.number_input("Aradaki sessizlik (sn)", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
    keep_padding_sec = st.number_input(
        "Sessizlik keserken segment padding (sn)",
        min_value=0.0,
        max_value=0.5,
        value=0.05,
        step=0.01,
        help="Sesli segmentlerin başına/sonuna küçük bir pay ekler (keskin kesmeyi azaltır).",
    )

    run = st.button("Birleştir ve Analiz Et", type="primary", disabled=not up)

    if run:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = _load_model_cached(model_path, device)
        feat_cfg = FeatureConfig()
        rt_cfg = RealtimeConfig(win_sec=float(win_sec), hop_sec=float(hop_sec), clip_sec=3.0, silence_rms_db=float(silence_db))

        signals = []
        names = []
        for f in up:
            data, sr = librosa.load(f, sr=feat_cfg.target_sr, mono=True)
            signals.append(data.astype(np.float32))
            names.append(f.name)

        merged = concat_with_silence(signals, sr=feat_cfg.target_sr, silence_sec=float(silence_sec))

        st.success(f"Birleştirilmiş sinyal: {len(merged)/feat_cfg.target_sr:.1f} sn • {len(up)} dosya")

        # export merged wav
        tmp_path = os.path.join(os.path.dirname(__file__), "_merged_temp.wav")
        sf.write(tmp_path, merged, feat_cfg.target_sr)
        with open(tmp_path, "rb") as f:
            st.download_button("Birleştirilmiş WAV indir", data=f.read(), file_name="merged.wav", mime="audio/wav")

        # REAL cut: remove silence from merged and export trimmed
        trimmed, segs = remove_silence_by_rms_sliding(
            merged,
            sr=feat_cfg.target_sr,
            win_sec=float(win_sec),
            hop_sec=float(hop_sec),
            silence_rms_db=float(silence_db),
            keep_padding_sec=float(keep_padding_sec),
        )

        if len(trimmed) == 0:
            st.error("Trim sonrası ses kalmadı. Sessizlik eşiğini düşürmeyi dene (daha negatif).")
            st.stop()

        st.info(
            f"Sessizlik kesildi: {len(trimmed)/feat_cfg.target_sr:.1f} sn (orig {len(merged)/feat_cfg.target_sr:.1f} sn) • "
            f"{len(segs)} sesli segment"
        )

        tmp_trim = os.path.join(os.path.dirname(__file__), "_merged_trimmed.wav")
        sf.write(tmp_trim, trimmed, feat_cfg.target_sr)
        with open(tmp_trim, "rb") as f:
            st.download_button(
                "Sessizlik kesilmiş WAV indir",
                data=f.read(),
                file_name="merged_trimmed.wav",
                mime="audio/wav",
            )

        # mapping table
        if segs:
            st.markdown("#### Trim mapping (orijinal ↔ trimmed segmentler)")
            st.dataframe(pd.DataFrame(segs), use_container_width=True, height=220)

        # analyze sliding windows
        rows = []
        win = rt_cfg.win_sec
        hop = rt_cfg.hop_sec
        idx = 0

        # helper: map trimmed sample index to original sample index (only if within one segment)
        def _map_trim_to_orig(sample_idx: int) -> Optional[int]:
            for s in segs:
                if s["trim_start"] <= sample_idx < s["trim_end"]:
                    return int(s["orig_start"] + (sample_idx - s["trim_start"]))
            return None

        for start, end, y_win in sliding_windows(trimmed, feat_cfg.target_sr, win, hop):
            out = infer_window(model, y_win, feat_cfg, rt_cfg, device)
            orig_s = _map_trim_to_orig(start)
            orig_e = _map_trim_to_orig(end - 1)
            crosses = False
            if orig_s is None or orig_e is None:
                crosses = True
            else:
                # if window spans two trimmed segments, the mapping is ambiguous
                orig_e = orig_e + 1
                if orig_e < orig_s:
                    crosses = True
            rows.append(
                {
                    "idx": idx,
                    "t_start_trim": start / feat_cfg.target_sr,
                    "t_end_trim": end / feat_cfg.target_sr,
                    "t_start_orig": None if orig_s is None else orig_s / feat_cfg.target_sr,
                    "t_end_orig": None if orig_e is None else orig_e / feat_cfg.target_sr,
                    "crosses_segments": crosses,
                    "is_silence": out["is_silence"],
                    "rms_db": out["rms_db"],
                    "label": out["label"] if out["label"] else "Silence",
                    "p_male": None if out["is_silence"] else out["probs"]["Male"],
                    "p_female": None if out["is_silence"] else out["probs"]["Female"],
                }
            )
            idx += 1

        df = pd.DataFrame(rows)
        st.markdown("#### Pencere sonuçları")
        st.dataframe(df, use_container_width=True, height=420)

        # plot timeline
        st.markdown("#### Zaman çizelgesi (Female olasılığı)")
        df2 = df[~df["is_silence"]].copy()
        if len(df2) == 0:
            st.warning("Sesli pencere bulunamadı (sessizlik eşiğini düşürmeyi dene).")
        else:
            st.line_chart(df2.set_index("t_start_trim")[["p_female", "p_male"]])

