# Copyright (c) 2026 MeanVC2 AudioSeal Benchmark.
"""
Robustness attack suite and evaluation metrics matching VALL-E & NeuMark validation protocols.
Includes DSP audio degradations and neural audio codecs (Encodec, DAC, SNAC),
ROC-AUC / TPR@FPR calculations, and formatted benchmark table generation.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple

import julius
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

try:
    from sklearn.metrics import roc_auc_score, roc_curve
except ImportError:
    roc_auc_score, roc_curve = None, None

try:
    from encodec import EncodecModel
except ImportError:
    EncodecModel = None

try:
    import dac
except ImportError:
    dac = None

try:
    from snac import SNAC
except ImportError:
    SNAC = None


def compute_auc_and_tpr_at_fpr(y_true, y_scores, target_fpr: float = 0.001) -> Tuple[float, float]:
    """
    Compute ROC-AUC and TPR at target False Positive Rate (default: 0.1% = 0.001).
    """
    if roc_auc_score is None or roc_curve is None or len(y_true) == 0:
        return 0.5, 0.0
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0
    try:
        auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        auc = 0.5

    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        valid_mask = fpr <= target_fpr
        if np.any(valid_mask):
            tpr_at_target = float(np.max(tpr[valid_mask]))
        else:
            tpr_at_target = float(tpr[0])
    except Exception:
        tpr_at_target = 0.0

    return auc, tpr_at_target


def _match_audio_length(audio: torch.Tensor, target_len: int) -> torch.Tensor:
    if audio.shape[-1] > target_len:
        return audio[..., :target_len]
    elif audio.shape[-1] < target_len:
        return F.pad(audio, (0, target_len - audio.shape[-1]))
    return audio


# -----------------------------------------------------------------------------
# 1. DSP Audio Effects
# -----------------------------------------------------------------------------
class AudioEffects:
    @staticmethod
    def identity(wav: torch.Tensor) -> torch.Tensor:
        return wav

    @staticmethod
    def random_noise(wav: torch.Tensor, noise_std: float = 0.001) -> torch.Tensor:
        return wav + torch.randn_like(wav) * noise_std

    @staticmethod
    def pink_noise(wav: torch.Tensor, noise_std: float = 0.01) -> torch.Tensor:
        length = wav.shape[-1]
        num_rows = 16
        array = torch.randn(num_rows, length // num_rows + 1, device=wav.device)
        reshaped = torch.cumsum(array, dim=1).reshape(-1)[:length]
        pink = reshaped / torch.max(torch.abs(reshaped) + 1e-8)
        return wav + pink * noise_std

    @staticmethod
    def lowpass_filter(wav: torch.Tensor, cutoff_freq: float = 5000, sample_rate: int = 16000) -> torch.Tensor:
        return julius.lowpass_filter(wav, cutoff=cutoff_freq / sample_rate, fft=False)

    @staticmethod
    def highpass_filter(wav: torch.Tensor, cutoff_freq: float = 500, sample_rate: int = 16000) -> torch.Tensor:
        return julius.highpass_filter(wav, cutoff=cutoff_freq / sample_rate, fft=False)

    @staticmethod
    def bandpass_filter(
        wav: torch.Tensor, cutoff_freq_low: float = 300, cutoff_freq_high: float = 8000, sample_rate: int = 16000
    ) -> torch.Tensor:
        return julius.bandpass_filter(
            wav,
            cutoff_low=cutoff_freq_low / sample_rate,
            cutoff_high=cutoff_freq_high / sample_rate,
            fft=False,
        )

    @staticmethod
    def echo(wav: torch.Tensor, volume: float = 0.3, duration: float = 0.2, sample_rate: int = 16000) -> torch.Tensor:
        delay = int(sample_rate * duration)
        if delay >= wav.shape[-1]:
            return wav
        echo_sig = torch.zeros_like(wav)
        echo_sig[..., delay:] = wav[..., :-delay] * volume
        return wav + echo_sig

    @staticmethod
    def smooth(wav: torch.Tensor, window_size: int = 5) -> torch.Tensor:
        kernel = torch.ones(1, 1, window_size, device=wav.device, dtype=wav.dtype) / window_size
        pad_len = window_size - 1
        padded = F.pad(wav, (pad_len, 0))
        smoothed = F.conv1d(padded, kernel)
        tmp = torch.zeros_like(wav)
        tmp[..., : smoothed.shape[-1]] = smoothed[..., : wav.shape[-1]]
        return tmp

    @staticmethod
    def boost_audio(wav: torch.Tensor, amount: float = 10) -> torch.Tensor:
        gain = 10 ** (amount / 20)
        return wav * gain

    @staticmethod
    def duck_audio(wav: torch.Tensor, amount: float = 10) -> torch.Tensor:
        gain = 10 ** (-amount / 20)
        return wav * gain

    @staticmethod
    def updownresample(wav: torch.Tensor, sample_rate: int = 16000, intermediate_freq: int = 32000) -> torch.Tensor:
        resampled = julius.resample_frac(wav, sample_rate, intermediate_freq)
        return julius.resample_frac(resampled, intermediate_freq, sample_rate)

    @staticmethod
    def speed(wav: torch.Tensor, speed_factor: float = 1.1, sample_rate: int = 16000) -> torch.Tensor:
        orig_len = wav.shape[-1]
        stretched = julius.resample_frac(wav, int(sample_rate * speed_factor), sample_rate)
        return _match_audio_length(stretched, orig_len)


# -----------------------------------------------------------------------------
# 2. Neural Audio Codec Attacks
# -----------------------------------------------------------------------------
class EncodecAttack:
    _model = None

    @classmethod
    def get_model(cls, device):
        if cls._model is None:
            if EncodecModel is None:
                raise ImportError("encodec not installed")
            model = EncodecModel.encodec_model_24khz()
            model.set_target_bandwidth(6.0)
            model.eval().to(device)
            for p in model.parameters():
                p.requires_grad = False
            cls._model = model
        return cls._model

    @classmethod
    def compress(cls, wav: torch.Tensor, bandwidth: float, sample_rate: int = 16000) -> torch.Tensor:
        orig_len = wav.shape[-1]
        device = wav.device
        model = cls.get_model(device)
        model.set_target_bandwidth(bandwidth)

        with torch.no_grad():
            wav_24k = julius.resample_frac(wav, sample_rate, 24000) if sample_rate != 24000 else wav
            if wav_24k.ndim == 2:
                wav_24k = wav_24k.unsqueeze(1)
            frames = model.encode(wav_24k)
            decoded = model.decode(frames)
            out = julius.resample_frac(decoded, 24000, sample_rate) if sample_rate != 24000 else decoded
        return _match_audio_length(out, orig_len)

    @classmethod
    def release(cls):
        cls._model = None


class DACAttack:
    _models = {}

    @classmethod
    def get_model(cls, model_type: str, device):
        if model_type not in cls._models:
            if dac is None:
                raise ImportError("descript-audio-codec (dac) not installed")
            model_path = dac.utils.download(model_type=model_type)
            model = dac.DAC.load(model_path)
            model.eval().to(device)
            for p in model.parameters():
                p.requires_grad = False
            cls._models[model_type] = model
        return cls._models[model_type]

    @classmethod
    def compress(cls, wav: torch.Tensor, model_type: str, sample_rate: int = 16000) -> torch.Tensor:
        orig_len = wav.shape[-1]
        device = wav.device
        target_sr = 16000 if model_type == "16khz" else (24000 if model_type == "24khz" else 44100)
        model = cls.get_model(model_type, device)

        with torch.no_grad():
            wav_target = julius.resample_frac(wav, sample_rate, target_sr) if sample_rate != target_sr else wav
            if wav_target.ndim == 2:
                wav_target = wav_target.unsqueeze(1)
            x = model.preprocess(wav_target, target_sr)
            z, codes, latents, _, _ = model.encode(x)
            decoded = model.decode(z)
            out = julius.resample_frac(decoded, target_sr, sample_rate) if sample_rate != target_sr else decoded
        return _match_audio_length(out, orig_len)

    @classmethod
    def release(cls):
        cls._models.clear()


class SNACAttack:
    _models = {}

    @classmethod
    def get_model(cls, model_type: str, device):
        if model_type not in cls._models:
            if SNAC is None:
                raise ImportError("snac not installed")
            model_map = {
                "24khz": "hubertsiuzdak/snac_24khz",
                "32khz": "hubertsiuzdak/snac_32khz",
                "44khz": "hubertsiuzdak/snac_44khz",
            }
            model = SNAC.from_pretrained(model_map[model_type]).eval().to(device)
            for p in model.parameters():
                p.requires_grad = False
            cls._models[model_type] = model
        return cls._models[model_type]

    @classmethod
    def compress(cls, wav: torch.Tensor, model_type: str, sample_rate: int = 16000) -> torch.Tensor:
        orig_len = wav.shape[-1]
        device = wav.device
        target_sr = 24000 if model_type == "24khz" else (32000 if model_type == "32khz" else 44100)
        model = cls.get_model(model_type, device)

        with torch.no_grad():
            wav_target = julius.resample_frac(wav, sample_rate, target_sr) if sample_rate != target_sr else wav
            if wav_target.ndim == 2:
                wav_target = wav_target.unsqueeze(1)
            codes = model.encode(wav_target)
            decoded = model.decode(codes)
            out = julius.resample_frac(decoded, target_sr, sample_rate) if sample_rate != target_sr else decoded
        return _match_audio_length(out, orig_len)

    @classmethod
    def release(cls):
        cls._models.clear()


# -----------------------------------------------------------------------------
# 3. Full Validation Attack Suite
# -----------------------------------------------------------------------------
def get_validation_attack_suite(sample_rate: int = 16000) -> List[Tuple[str, str, str, Callable[[torch.Tensor], torch.Tensor]]]:
    """
    Returns list of (Category, Family, Bitrate/Detail, AttackFunction)
    """
    return [
        # DSP Attacks (12)
        ("DSP", "Clean (Identity)", "", lambda wav: AudioEffects.identity(wav)),
        ("DSP", "Gaussian Noise", "", lambda wav: AudioEffects.random_noise(wav, 0.001)),
        ("DSP", "Pink Noise", "", lambda wav: AudioEffects.pink_noise(wav, 0.01)),
        ("DSP", "Lowpass Filter (5k)", "", lambda wav: AudioEffects.lowpass_filter(wav, 5000, sample_rate)),
        ("DSP", "Highpass Filter (500)", "", lambda wav: AudioEffects.highpass_filter(wav, 500, sample_rate)),
        ("DSP", "Bandpass Filter", "", lambda wav: AudioEffects.bandpass_filter(wav, 300, 8000, sample_rate)),
        ("DSP", "Echo/Reverb", "", lambda wav: AudioEffects.echo(wav, 0.3, 0.2, sample_rate)),
        ("DSP", "Smooth (Moving Avg)", "", lambda wav: AudioEffects.smooth(wav, 5)),
        ("DSP", "Resampling (32k-16k)", "", lambda wav: AudioEffects.updownresample(wav, sample_rate, 32000)),
        ("DSP", "Volume Boost (+10%)", "", lambda wav: AudioEffects.boost_audio(wav, 10)),
        ("DSP", "Volume Duck (-10%)", "", lambda wav: AudioEffects.duck_audio(wav, 10)),
        ("DSP", "Speed (0.8x-1.2x)", "", lambda wav: AudioEffects.speed(wav, 1.1, sample_rate)),

        # Codec Attacks (9)
        ("Codec", "Encodec", "3 kbps", lambda wav: EncodecAttack.compress(wav, 3.0, sample_rate)),
        ("Codec", "Encodec", "6 kbps", lambda wav: EncodecAttack.compress(wav, 6.0, sample_rate)),
        ("Codec", "Encodec", "12 kbps", lambda wav: EncodecAttack.compress(wav, 12.0, sample_rate)),
        ("Codec", "DAC", "6 kbps", lambda wav: DACAttack.compress(wav, "16khz", sample_rate)),
        ("Codec", "DAC", "8 kbps", lambda wav: DACAttack.compress(wav, "44khz", sample_rate)),
        ("Codec", "DAC", "24 kbps", lambda wav: DACAttack.compress(wav, "24khz", sample_rate)),
        ("Codec", "SNAC", "0.98 kbps", lambda wav: SNACAttack.compress(wav, "24khz", sample_rate)),
        ("Codec", "SNAC", "1.9 kbps", lambda wav: SNACAttack.compress(wav, "32khz", sample_rate)),
        ("Codec", "SNAC", "2.6 kbps", lambda wav: SNACAttack.compress(wav, "44khz", sample_rate)),
    ]


# -----------------------------------------------------------------------------
# 4. Formatted Validation Table Report
# -----------------------------------------------------------------------------
def format_full_validation_table(
    step: str | int,
    results: Dict[str, Dict[str, float]],
    quality_metrics: Optional[Dict[str, float]] = None,
) -> str:
    hdr_line = "=" * 125
    div_line = "-" * 125
    lines = [
        hdr_line,
        f"  Benchmark Validation Report (Model / Evaluation Target: {step})",
        hdr_line,
        f"{'Attack Type':<32} | {'Detect ACC':<11} | {'Det ROC-AUC':<11} | {'Det TPR@0.1%':<12} | {'WM Bit Acc':<11} | {'WM ROC-AUC':<11} | {'WM TPR@0.1%':<12}",
        div_line,
    ]

    dsp_det_accs, dsp_det_aucs, dsp_det_tprs = [], [], []
    dsp_wm_accs, dsp_wm_aucs, dsp_wm_tprs = [], [], []
    codec_det_accs, codec_det_aucs, codec_det_tprs = [], [], []
    codec_wm_accs, codec_wm_aucs, codec_wm_tprs = [], [], []

    # 1. Print DSP Section
    for name, stats in results.items():
        if stats.get("category") == "DSP":
            det_acc = stats.get("detect_acc", 0.0)
            det_auc = stats.get("det_roc_auc", stats.get("roc_auc", 0.5))
            det_tpr = stats.get("det_tpr_at_001_fpr", stats.get("tpr_at_001_fpr", 0.0))
            wm_bit = stats.get("bit_acc", 0.0)
            wm_auc = stats.get("wm_roc_auc", 0.5)
            wm_tpr = stats.get("wm_tpr_at_001_fpr", 0.0)

            dsp_det_accs.append(det_acc)
            dsp_det_aucs.append(det_auc)
            dsp_det_tprs.append(det_tpr)
            dsp_wm_accs.append(wm_bit)
            dsp_wm_aucs.append(wm_auc)
            dsp_wm_tprs.append(wm_tpr)

            lines.append(
                f"{name:<32} | {det_acc:<11.4f} | {det_auc:<11.4f} | {det_tpr:<12.4f} | {wm_bit:<11.4f} | {wm_auc:<11.4f} | {wm_tpr:<12.4f}"
            )

    if dsp_det_accs:
        avg_d_acc = sum(dsp_det_accs) / len(dsp_det_accs)
        avg_d_auc = sum(dsp_det_aucs) / len(dsp_det_aucs)
        avg_d_tpr = sum(dsp_det_tprs) / len(dsp_det_tprs)
        avg_w_acc = sum(dsp_wm_accs) / len(dsp_wm_accs)
        avg_w_auc = sum(dsp_wm_aucs) / len(dsp_wm_aucs)
        avg_w_tpr = sum(dsp_wm_tprs) / len(dsp_wm_tprs)
        lines.append(
            f"{'DSP Avg.':<32} | {avg_d_acc:<11.4f} | {avg_d_auc:<11.4f} | {avg_d_tpr:<12.4f} | {avg_w_acc:<11.4f} | {avg_w_auc:<11.4f} | {avg_w_tpr:<12.4f}"
        )
        lines.append(div_line)

    # 2. Print Codec Section (Grouped by Family)
    codec_families = ["Encodec", "DAC", "SNAC"]
    for family in codec_families:
        first = True
        for name, stats in results.items():
            if stats.get("category") == "Codec" and stats.get("family") == family:
                det_acc = stats.get("detect_acc", 0.0)
                det_auc = stats.get("det_roc_auc", stats.get("roc_auc", 0.5))
                det_tpr = stats.get("det_tpr_at_001_fpr", stats.get("tpr_at_001_fpr", 0.0))
                wm_bit = stats.get("bit_acc", 0.0)
                wm_auc = stats.get("wm_roc_auc", 0.5)
                wm_tpr = stats.get("wm_tpr_at_001_fpr", 0.0)

                codec_det_accs.append(det_acc)
                codec_det_aucs.append(det_auc)
                codec_det_tprs.append(det_tpr)
                codec_wm_accs.append(wm_bit)
                codec_wm_aucs.append(wm_auc)
                codec_wm_tprs.append(wm_tpr)

                bitrate = stats.get("bitrate", "")
                label = f"{family:<14} {bitrate}" if first else f"{'':<14} {bitrate}"
                first = False
                lines.append(
                    f"{label:<32} | {det_acc:<11.4f} | {det_auc:<11.4f} | {det_tpr:<12.4f} | {wm_bit:<11.4f} | {wm_auc:<11.4f} | {wm_tpr:<12.4f}"
                )
        lines.append("")

    if lines[-1] == "":
        lines.pop()

    if codec_det_accs:
        avg_c_d_acc = sum(codec_det_accs) / len(codec_det_accs)
        avg_c_d_auc = sum(codec_det_aucs) / len(codec_det_aucs)
        avg_c_d_tpr = sum(codec_det_tprs) / len(codec_det_tprs)
        avg_c_w_acc = sum(codec_wm_accs) / len(codec_wm_accs)
        avg_c_w_auc = sum(codec_wm_aucs) / len(codec_wm_aucs)
        avg_c_w_tpr = sum(codec_wm_tprs) / len(codec_wm_tprs)
        lines.append(
            f"{'Codec Avg.':<32} | {avg_c_d_acc:<11.4f} | {avg_c_d_auc:<11.4f} | {avg_c_d_tpr:<12.4f} | {avg_c_w_acc:<11.4f} | {avg_c_w_auc:<11.4f} | {avg_c_w_tpr:<12.4f}"
        )
        lines.append(div_line)

    all_d_accs = dsp_det_accs + codec_det_accs
    all_d_aucs = dsp_det_aucs + codec_det_aucs
    all_d_tprs = dsp_det_tprs + codec_det_tprs
    all_w_accs = dsp_wm_accs + codec_wm_accs
    all_w_aucs = dsp_wm_aucs + codec_wm_aucs
    all_w_tprs = dsp_wm_tprs + codec_wm_tprs

    if all_d_accs:
        tot_d_acc = sum(all_d_accs) / len(all_d_accs)
        tot_d_auc = sum(all_d_aucs) / len(all_d_aucs)
        tot_d_tpr = sum(all_d_tprs) / len(all_d_tprs)
        tot_w_acc = sum(all_w_accs) / len(all_w_accs)
        tot_w_auc = sum(all_w_aucs) / len(all_w_aucs)
        tot_w_tpr = sum(all_w_tprs) / len(all_w_tprs)
        lines.append(
            f"{'Overall Avg.':<32} | {tot_d_acc:<11.4f} | {tot_d_auc:<11.4f} | {tot_d_tpr:<12.4f} | {tot_w_acc:<11.4f} | {tot_w_auc:<11.4f} | {tot_w_tpr:<12.4f}"
        )

    if quality_metrics:
        lines.append("=" * 125)
        lines.append("  Speech Quality & Fidelity Degradation (Clean VC vs. Watermarked VC):")
        lines.append("-" * 125)
        lines.append(f"{'Metric':<28} | {'Clean VC':<12} | {'Watermarked':<12} | {'Delta (WM - Clean)':<18}")
        lines.append("-" * 125)

        # PESQ
        p_val = quality_metrics.get("pesq_wb", quality_metrics.get("pesq", None))
        if p_val is not None:
            lines.append(f"{'PESQ (WB 16kHz)':<28} | {'N/A (Ref)':<12} | {p_val:<12.4f} | {'-':<18}")

        # STOI
        s_val = quality_metrics.get("stoi", quality_metrics.get("stoi_val", None))
        if s_val is not None:
            lines.append(f"{'STOI (Intelligibility)':<28} | {'1.0000':<12} | {s_val:<12.4f} | {s_val - 1.0:+12.4f}")

        # SNR
        snr_val = quality_metrics.get("snr", quality_metrics.get("snr_db", None))
        if snr_val is not None:
            lines.append(f"{'SNR (dB, Signal-to-Noise)':<28} | {'N/A (Ref)':<12} | {snr_val:<12.2f} dB | {'-':<18}")

        # SIM
        c_sim = quality_metrics.get("clean_sim", 0.0)
        w_sim = quality_metrics.get("wm_sim", 0.0)
        d_sim = w_sim - c_sim
        lines.append(f"{'SIM (Speaker Cosine Sim)':<28} | {c_sim:<12.4f} | {w_sim:<12.4f} | {d_sim:+12.4f}")

        lines.append("-" * 125)
        # Efficiency
        emb_ms = quality_metrics.get("embed_overhead_ms_per_sec", 0.0)
        emb_rtf = quality_metrics.get("embed_rtf", emb_ms / 1000.0)
        det_ms = quality_metrics.get("detect_latency_ms_per_sec", 0.0)
        det_rtf = quality_metrics.get("detect_rtf", det_ms / 1000.0)
        det_mode = quality_metrics.get("detect_mode", "Streaming")
        det_chunk_ms = quality_metrics.get("detect_chunk_ms", 40)
        det_chunk_lat = quality_metrics.get("detect_chunk_latency_ms", None)

        lines.append("  Efficiency & Computational Latency:")
        lines.append("-" * 125)
        lines.append(f"{'Embedding Overhead (ms/s)':<28} | {emb_ms:<12.2f} ms per second of audio  (RTF: {emb_rtf:.4f})")
        lines.append(f"{'Detection Mode':<28} | {det_mode} (Chunk: {det_chunk_ms} ms)")
        if det_chunk_lat is not None:
            lines.append(f"{'Detection Latency (per chunk)':<28} | {det_chunk_lat:<12.2f} ms per {det_chunk_ms}ms chunk")
        lines.append(f"{'Detection Throughput (ms/s)':<28} | {det_ms:<12.2f} ms per second of audio  (RTF: {det_rtf:.4f})")

    lines.append("=" * 125)
    return "\n".join(lines)
