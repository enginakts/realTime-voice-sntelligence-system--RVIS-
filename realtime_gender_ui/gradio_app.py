from __future__ import annotations

import os
import socket
import tempfile
import threading
import time
import queue
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from audio_features import (
    FeatureConfig,
    concat_with_silence,
    remove_silence_by_rms_sliding,
    rms_db,
    sliding_windows,
)
from gender_infer import RealtimeConfig, infer_window
from model_gender_cnn_bilstm import load_gender_model


_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = str(_ROOT / "code" / "models" / "model.pt")

try:
    import sounddevice as sd  # type: ignore

    _HAS_SOUNDDEVICE = True
except Exception:
    sd = None  # type: ignore
    _HAS_SOUNDDEVICE = False


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(path: str) -> torch.nn.Module:
    dev = _device()
    return load_gender_model(path, device=dev)


def _fig_timeline(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 3))
    if len(df) == 0:
        ax.text(0.5, 0.5, "No voiced windows", ha="center", va="center")
        ax.axis("off")
        return fig
    ax.plot(df["t_start_trim"], df["p_female"], label="p_female")
    ax.plot(df["t_start_trim"], df["p_male"], label="p_male")
    ax.set_xlabel("time (trimmed, s)")
    ax.set_ylabel("probability")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    return fig


# ---------------------------
# Senaryo 1: microphone (server-side) realtime
# ---------------------------


# Canlı akış: kısa çerçevelerde VAD → sessizlik modele hiç yazılmadan düşer
_VAD_FRAME_SAMPLES = 160  # 10 ms @ 16 kHz
_VAD_HANGOVER_FRAMES = 10  # ~100 ms kelime sonu
_VAD_MIN_SILENCE_FRAMES = 15  # ~150 ms tam sessizlik: yetersiz kuyruk temizlenir


class MicEngine:
    def __init__(self):
        self.cfg = FeatureConfig()
        self.rt = RealtimeConfig()
        self.device = _device()
        self.model_path: str = DEFAULT_MODEL_PATH
        self.model: Optional[torch.nn.Module] = None
        self.is_running: bool = False

        # Ham mikrofon parçaları (VAD çerçevesine bölünür)
        self._vad_carry = np.zeros((0,), dtype=np.float32)
        # Sadece "konuşma" sayılan örnekler (boşluklar çıkarılmış akış)
        self._strip_buf = np.zeros((0,), dtype=np.float32)
        # UI için ham dalga formu (son N sn)
        self._raw_buf = np.zeros((0,), dtype=np.float32)
        self._raw_keep_sec = 2.0
        self._last_raw_db: float = -120.0
        self._hangover_frames = 0
        self._silence_run_frames = 0
        # Çıktı zaman ekseni: sıkıştırılmış konuşma süresi (sessizlik yok)
        self._speech_timeline_samples = 0

        self._q: "queue.Queue[dict]" = queue.Queue()
        self._stream = None
        self._lock = threading.Lock()

    def _reset_vad_pipeline(self):
        self._vad_carry = np.zeros((0,), dtype=np.float32)
        self._strip_buf = np.zeros((0,), dtype=np.float32)
        self._raw_buf = np.zeros((0,), dtype=np.float32)
        self._last_raw_db = -120.0
        self._hangover_frames = 0
        self._silence_run_frames = 0
        self._speech_timeline_samples = 0

    def configure(self, model_path: str, silence_db: float, win_sec: float, hop_sec: float):
        with self._lock:
            self.model_path = model_path
            self.device = _device()
            self.rt = RealtimeConfig(
                win_sec=float(win_sec),
                hop_sec=float(hop_sec),
                clip_sec=3.0,
                silence_rms_db=float(silence_db),
            )
            self.model = load_gender_model(model_path, device=self.device)

    def start(self):
        if not _HAS_SOUNDDEVICE:
            raise RuntimeError("sounddevice yok")

        if self._stream is not None:
            return

        self._reset_vad_pipeline()

        win = int(self.cfg.target_sr * self.rt.win_sec)
        hop = int(self.cfg.target_sr * self.rt.hop_sec)
        f = _VAD_FRAME_SAMPLES
        thr = float(self.rt.silence_rms_db)

        def callback(indata, frames, time_info, status):  # noqa: ARG001
            y = np.asarray(indata, dtype=np.float32).reshape(-1)
            # raw buffer for waveform UI
            with self._lock:
                self._raw_buf = np.concatenate([self._raw_buf, y], axis=0)
                keep = int(self._raw_keep_sec * self.cfg.target_sr)
                if len(self._raw_buf) > keep:
                    self._raw_buf = self._raw_buf[-keep:]
                self._last_raw_db = float(rms_db(y))

            self._vad_carry = np.concatenate([self._vad_carry, y], axis=0)

            while len(self._vad_carry) >= f:
                frame = self._vad_carry[:f]
                self._vad_carry = self._vad_carry[f:]
                db = rms_db(frame)
                voiced = db >= thr

                if voiced:
                    self._hangover_frames = _VAD_HANGOVER_FRAMES
                    self._silence_run_frames = 0
                    self._strip_buf = np.concatenate([self._strip_buf, frame], axis=0)
                elif self._hangover_frames > 0:
                    self._hangover_frames -= 1
                    self._silence_run_frames = 0
                    self._strip_buf = np.concatenate([self._strip_buf, frame], axis=0)
                else:
                    self._silence_run_frames += 1
                    if (
                        self._silence_run_frames >= _VAD_MIN_SILENCE_FRAMES
                        and len(self._strip_buf) > 0
                        and len(self._strip_buf) < win
                    ):
                        self._strip_buf = np.zeros((0,), dtype=np.float32)

                while len(self._strip_buf) >= win and self.model is not None:
                    y_win = self._strip_buf[:win]
                    self._strip_buf = self._strip_buf[hop:]
                    t_sec = self._speech_timeline_samples / self.cfg.target_sr
                    self._speech_timeline_samples += hop
                    out = infer_window(self.model, y_win, self.cfg, self.rt, self.device)
                    if not out["is_silence"]:
                        self._q.put({"sec": round(float(t_sec), 2), "label": out["label"]})

        # 16k mono float32 stream
        self._stream = sd.InputStream(
            samplerate=self.cfg.target_sr,
            channels=1,
            dtype="float32",
            callback=callback,
            blocksize=0,
        )
        self._stream.start()
        self.is_running = True

    def stop(self):
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None
            self._reset_vad_pipeline()
            self.is_running = False
            # queue flush
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except Exception:
                    break

    def drain(self, max_items: int = 50) -> List[dict]:
        out = []
        for _ in range(max_items):
            try:
                out.append(self._q.get_nowait())
            except Exception:
                break
        return out

    def snapshot_waveform(self) -> Tuple[np.ndarray, float]:
        with self._lock:
            return self._raw_buf.copy(), float(self._last_raw_db)


MIC = MicEngine()


def mic_ui_init_state() -> Dict[str, Any]:
    return {"history": []}


def mic_start(model_path: str, silence_db: float, win_sec: float, hop_sec: float) -> str:
    if not _HAS_SOUNDDEVICE:
        return "sounddevice yok. Kur: `pip install sounddevice`"
    MIC.configure(model_path, silence_db, win_sec, hop_sec)
    MIC.start()
    return "✅ Mikrofon başladı. Konuşmaya başla; tahminler ve dalga formu aşağıda akacak."


def mic_stop() -> str:
    MIC.stop()
    return "⏹️ Mikrofon durduruldu."


def _fig_wave(y: np.ndarray, sr: int):
    fig, ax = plt.subplots(figsize=(10, 2))
    if y is None or len(y) == 0:
        ax.text(0.5, 0.5, "No audio yet", ha="center", va="center")
        ax.axis("off")
        return fig
    t = np.arange(len(y)) / float(sr)
    ax.plot(t, y, linewidth=0.8)
    ax.set_xlabel("time (s, last ~2s)")
    ax.set_ylabel("amp")
    ax.grid(True, alpha=0.25)
    return fig


def mic_poll(state: Dict[str, Any]) -> Tuple[pd.DataFrame, Any, str, Dict[str, Any]]:
    if state is None:
        state = mic_ui_init_state()
    history: List[Dict[str, Any]] = state.get("history", [])
    new = MIC.drain(100)
    if new:
        history.extend(new)
        history = history[-300:]
    state["history"] = history
    df = pd.DataFrame(history[-30:][::-1], columns=["sec", "label"])
    y, db = MIC.snapshot_waveform()
    fig = _fig_wave(y, MIC.cfg.target_sr)
    level = "Mic: OFF" if not MIC.is_running else f"Mic: ON • last chunk RMS ≈ {db:.1f} dB"
    return df, fig, level, state


# -----------------------------------------
# Senaryo 2: merge -> real trim -> analyze
# -----------------------------------------


def scenario2_run(
    files: List[str],
    model_path: str,
    silence_between_sec: float,
    silence_db: float,
    win_sec: float,
    hop_sec: float,
    keep_padding_sec: float,
):
    if not files:
        return None, None, None, None, None, "No files provided"

    feat_cfg = FeatureConfig()
    rt_cfg = RealtimeConfig(win_sec=float(win_sec), hop_sec=float(hop_sec), clip_sec=3.0, silence_rms_db=float(silence_db))

    dev = _device()
    model = _load_model(model_path)

    signals = []
    for p in files:
        y, _ = librosa.load(p, sr=feat_cfg.target_sr, mono=True)
        signals.append(y.astype(np.float32))

    merged = concat_with_silence(signals, sr=feat_cfg.target_sr, silence_sec=float(silence_between_sec))

    trimmed, segs = remove_silence_by_rms_sliding(
        merged,
        sr=feat_cfg.target_sr,
        win_sec=float(win_sec),
        hop_sec=float(hop_sec),
        silence_rms_db=float(silence_db),
        keep_padding_sec=float(keep_padding_sec),
    )

    if len(trimmed) == 0:
        return None, None, None, None, None, "Trim sonrası ses kalmadı. Sessizlik eşiğini düşür."

    # export wavs
    out_dir = tempfile.mkdtemp(prefix="gender_ui_")
    merged_path = os.path.join(out_dir, "merged.wav")
    trimmed_path = os.path.join(out_dir, "merged_trimmed.wav")
    sf.write(merged_path, merged, feat_cfg.target_sr)
    sf.write(trimmed_path, trimmed, feat_cfg.target_sr)

    # mapping helper
    def map_trim_to_orig(sample_idx: int) -> Optional[int]:
        for s in segs:
            if s["trim_start"] <= sample_idx < s["trim_end"]:
                return int(s["orig_start"] + (sample_idx - s["trim_start"]))
        return None

    rows = []
    idx = 0
    for start, end, y_win in sliding_windows(trimmed, feat_cfg.target_sr, rt_cfg.win_sec, rt_cfg.hop_sec):
        out = infer_window(model, y_win, feat_cfg, rt_cfg, dev)
        orig_s = map_trim_to_orig(start)
        orig_e = map_trim_to_orig(end - 1)
        crosses = False
        if orig_s is None or orig_e is None:
            crosses = True
        else:
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
    seg_df = pd.DataFrame(segs)

    voiced = df[~df["is_silence"]].copy()
    fig = _fig_timeline(voiced)

    msg = (
        f"Merged: {len(merged)/feat_cfg.target_sr:.1f}s • "
        f"Trimmed: {len(trimmed)/feat_cfg.target_sr:.1f}s • "
        f"Segments: {len(segs)} • Windows: {len(df)}"
    )
    return merged_path, trimmed_path, seg_df, df, fig, msg


with gr.Blocks(title="Realtime Gender Classifier") as demo:
    gr.Markdown("## Realtime Gender Classifier (Gradio)\nSenaryo 1: Mikrofon • Senaryo 2: Birleştirilmiş dosya + gerçek sessizlik kesme")

    with gr.Row():
        model_path = gr.Textbox(label="Model path", value=DEFAULT_MODEL_PATH, scale=3)
        silence_db = gr.Slider(label="Sessizlik eşiği (RMS dB)", minimum=-80, maximum=-10, value=-45, step=1, scale=2)
        win_sec = gr.Dropdown(label="Pencere (sn)", choices=[0.5, 1.0, 1.5, 2.0], value=1.0, scale=1)
        hop_sec = gr.Dropdown(label="Hop (sn)", choices=[0.25, 0.5, 1.0], value=0.5, scale=1)

    with gr.Tabs():
        with gr.Tab("Senaryo 1 • Mikrofon"):
            gr.Markdown(
                "**Mikrofon (bilgisayar, sounddevice):** Önce kısa pencerelerde RMS (dB) ile **sessizlik çıkarılır**; "
                "yalnızca konuşma sayılan örnekler modele gider. `sec` ekseni **boşluksuz konuşma süresidir**. "
                "Eşik = üstteki *Sessizlik eşiği (RMS dB)*."
            )
            mic_state = gr.State(mic_ui_init_state())
            with gr.Row():
                start_btn = gr.Button("Start", variant="primary", visible=True)
                stop_btn = gr.Button("Stop", visible=False)
                status = gr.Markdown()

            mic_level = gr.Markdown()
            # Kayıt animasyonu: mikrofon aktifken görünür (CSS keyframes ile otomatik hareket)
            rec_anim = gr.HTML(
                value='<div style=\"display:flex;align-items:center;gap:12px;\"><div style=\"font-weight:600;\">Recording…</div><div class=\"vu-wrap\" aria-label=\"recording-indicator\"><span class=\"vu-bar\"></span><span class=\"vu-bar\"></span><span class=\"vu-bar\"></span><span class=\"vu-bar\"></span><span class=\"vu-bar\"></span><span class=\"vu-bar\"></span><span class=\"vu-bar\"></span><span class=\"vu-bar\"></span></div></div><style>.vu-wrap{display:flex;align-items:flex-end;gap:4px;height:18px;}.vu-bar{width:4px;background:#f97316;border-radius:2px;animation:vu 0.9s ease-in-out infinite;opacity:0.9;}.vu-bar:nth-child(1){animation-delay:0.00s}.vu-bar:nth-child(2){animation-delay:0.10s}.vu-bar:nth-child(3){animation-delay:0.20s}.vu-bar:nth-child(4){animation-delay:0.30s}.vu-bar:nth-child(5){animation-delay:0.15s}.vu-bar:nth-child(6){animation-delay:0.25s}.vu-bar:nth-child(7){animation-delay:0.05s}.vu-bar:nth-child(8){animation-delay:0.35s}@keyframes vu{0%{height:4px}20%{height:16px}50%{height:7px}80%{height:18px}100%{height:5px}}</style>',
                visible=False,
            )
            mic_wave = gr.Plot(label="Waveform (last ~2s)")
            mic_table = gr.Dataframe(headers=["sec", "label"], datatype=["number", "str"], interactive=False)
            timer = gr.Timer(0.5)

            # Start: butonu gizle, Stop+animasyonu göster
            def _ui_on_start(model_path: str, silence_db: float, win_sec: float, hop_sec: float):
                msg = mic_start(model_path, silence_db, win_sec, hop_sec)
                return (
                    msg,
                    gr.update(visible=False),  # start_btn
                    gr.update(visible=True),   # stop_btn
                    gr.update(visible=True),   # rec_anim
                )

            # Stop: Start geri gelsin, Stop+animasyon gizlensin
            def _ui_on_stop():
                msg = mic_stop()
                return (
                    msg,
                    gr.update(visible=True),   # start_btn
                    gr.update(visible=False),  # stop_btn
                    gr.update(visible=False),  # rec_anim
                )

            start_btn.click(fn=_ui_on_start, inputs=[model_path, silence_db, win_sec, hop_sec], outputs=[status, start_btn, stop_btn, rec_anim])
            stop_btn.click(fn=_ui_on_stop, inputs=[], outputs=[status, start_btn, stop_btn, rec_anim])
            timer.tick(fn=mic_poll, inputs=[mic_state], outputs=[mic_table, mic_wave, mic_level, mic_state])

        with gr.Tab("Senaryo 2 • Birleştir + Trim + Analiz"):
            files = gr.File(file_count="multiple", label="Ses dosyaları (wav/mp3/m4a/flac/ogg)")
            silence_between_sec = gr.Slider(label="Dosyalar arası sessizlik (sn)", minimum=0, maximum=10, value=5, step=0.5)
            keep_padding_sec = gr.Slider(label="Trim padding (sn)", minimum=0, maximum=0.5, value=0.05, step=0.01)
            run_btn = gr.Button("Birleştir ve Analiz Et", variant="primary")

            msg = gr.Markdown()
            out_merged = gr.File(label="Merged WAV")
            out_trimmed = gr.File(label="Trimmed WAV (silence removed)")
            seg_table = gr.Dataframe(label="Trim mapping (segments)", interactive=False)
            win_table = gr.Dataframe(label="Window results", interactive=False)
            plot = gr.Plot(label="Timeline (trimmed time)")

            run_btn.click(
                fn=scenario2_run,
                inputs=[files, model_path, silence_between_sec, silence_db, win_sec, hop_sec, keep_padding_sec],
                outputs=[out_merged, out_trimmed, seg_table, win_table, plot, msg],
            )


if __name__ == "__main__":
    # Streaming için queue gerekli (aksi halde bazı tarayıcılarda sonuçlar kayıt sonunda gelebilir)
    demo.queue()

    # Port çakışmalarını önlemek için boş port seç
    env_port = os.getenv("GRADIO_SERVER_PORT")

    def _find_free_port(preferred: int = 7860, span: int = 50) -> int:
        for p in range(preferred, preferred + span):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(("127.0.0.1", p))
                    return p
                except OSError:
                    continue
        # fallback: OS assigns a free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])

    port = int(env_port) if env_port else _find_free_port(7860, 50)
    demo.launch(server_name="127.0.0.1", server_port=port)

