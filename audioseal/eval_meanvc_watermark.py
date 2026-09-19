#!/usr/bin/env python3
# Copyright (c) 2026 MeanVC2 AudioSeal Integration. All rights reserved.
"""
Comprehensive 1,000-sample streaming voice conversion (MeanVC2) +
real-time AudioSeal watermark embedding evaluation benchmark.
Evaluates watermark robustness under 21 attacks (DSP + neural codecs)
and audio quality degradation (PESQ, STOI, SNR, SIM) matching VALL-E Neumark standard.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

# Configure paths
SCRIPT_DIR = Path(__file__).resolve().parent
MEANVC_ROOT = SCRIPT_DIR.parent
RUNTIME_DIR = MEANVC_ROOT / "runtime"
RUNTIME_SRC_DIR = RUNTIME_DIR / "src"

for p in [str(SCRIPT_DIR), str(MEANVC_ROOT), str(RUNTIME_DIR), str(RUNTIME_SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Prevent PyTorch Dynamo JIT compile issues on GCC 8.5
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PYTHONUNBUFFERED"] = "1"

from eval_attacks import (
    compute_auc_and_tpr_at_fpr,
    format_full_validation_table,
    get_validation_attack_suite,
)
from run_rt import DEFAULT_TARGET_WAV, VCRunner
from src.speaker import extract_embedding
from watermarker import AudioSealStreamWatermarker

try:
    from pesq import pesq
except ImportError:
    pesq = None

try:
    from pystoi import stoi
except ImportError:
    stoi = None


def resolve_path(p: str | None) -> str:
    if p is None:
        return ""
    if os.path.exists(p):
        return str(Path(p).resolve())
    alt = MEANVC_ROOT / p
    if alt.exists():
        return str(alt.resolve())
    return str(Path(p).resolve())


def collect_libritts_samples(
    dataset_dir: Path,
    num_samples: int = 1000,
    min_duration: float = 2.0,
    max_duration: float = 10.0,
    seed: int = 42,
) -> List[Tuple[str, str, float, str]]:
    """
    Collect audio samples from LibriTTS directory.
    Returns list of (sample_id, wav_path, duration_sec, transcript).
    """
    print(f"[Dataset] Scanning for .wav files in: {dataset_dir} ...")
    all_wavs = []
    for root, _, files in os.walk(str(dataset_dir)):
        for f in files:
            if f.endswith(".wav"):
                all_wavs.append(os.path.join(root, f))

    all_wavs.sort()
    rng = np.random.RandomState(seed)
    rng.shuffle(all_wavs)

    selected = []
    print(f"[Dataset] Found {len(all_wavs)} candidate files. Filtering for [{min_duration:.1f}s, {max_duration:.1f}s]...")

    for wav_p in all_wavs:
        try:
            info = sf.info(wav_p)
            dur = info.duration
            if min_duration <= dur <= max_duration:
                sid = Path(wav_p).stem
                # Check for transcript file
                txt_p = Path(wav_p).with_suffix(".normalized.txt")
                if not txt_p.exists():
                    txt_p = Path(wav_p).with_suffix(".txt")
                text = txt_p.read_text().strip() if txt_p.exists() else ""
                selected.append((sid, wav_p, dur, text))
                if len(selected) >= num_samples:
                    break
        except Exception:
            continue

    print(f"[Dataset] Successfully selected {len(selected)} samples for evaluation.")
    return selected


def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    noise = noisy - clean
    p_signal = np.sum(clean**2)
    p_noise = np.sum(noise**2)
    if p_noise <= 1e-10:
        return 60.0
    if p_signal <= 1e-10:
        return 0.0
    return float(10.0 * np.log10(p_signal / p_noise))


def run_streaming_detection(
    detector,
    audio_tensor: torch.Tensor,
    chunk_samples: int = 640,
) -> Tuple[float, torch.Tensor, float, float]:
    """
    Run chunk-by-chunk streaming detection.
    Args:
        detector: AudioSealDetector model
        audio_tensor: [1, 1, T] audio waveform
        chunk_samples: number of samples per chunk (640 for 40ms, 2560 for 160ms)
    Returns:
        mean_score: float mean probability of watermarked class
        avg_msg: [1, 16] sigmoid message probabilities
        total_elapsed_sec: total forward time in seconds
        avg_chunk_latency_ms: average latency per chunk in milliseconds
    """
    T = audio_tensor.shape[-1]
    scores = []
    msgs = []
    chunk_latencies = []
    t0 = time.time()
    for i in range(0, T, chunk_samples):
        chk = audio_tensor[..., i : i + chunk_samples]
        tc0 = time.time()
        with torch.no_grad():
            res, msg = detector(chk)
        chunk_latencies.append((time.time() - tc0) * 1000.0)
        scores.append(res[:, 1, :])
        msgs.append(msg)
    elapsed = time.time() - t0
    all_scores = torch.cat(scores, dim=-1)
    mean_score = float(all_scores.mean().item())
    avg_msg = torch.stack(msgs).mean(dim=0)
    avg_chunk_latency = float(np.mean(chunk_latencies)) if chunk_latencies else 0.0
    return mean_score, avg_msg, elapsed, avg_chunk_latency


def main():
    parser = argparse.ArgumentParser(description="MeanVC2 + AudioSeal 1000-Sample Evaluation Benchmark")
    parser.add_argument(
        "--libritts-dir",
        type=str,
        default="/home/pj25001109/ku60000344/dataset/libriTTS/LibriTTS/train-clean-100",
        help="Path to LibriTTS dataset directory",
    )
    parser.add_argument(
        "--target-wav",
        type=str,
        default=DEFAULT_TARGET_WAV,
        help="Path to target speaker reference audio",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to evaluate (default: 1000)",
    )
    parser.add_argument(
        "--model",
        choices=["40ms", "120ms"],
        default="40ms",
        help="MeanVC2 model config (default: 40ms)",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        choices=[40, 160],
        default=160,
        help="MeanVC2 streaming I/O chunk size in ms (default: 160ms = 2560 samples; 40ms = 640 samples)",
    )
    parser.add_argument(
        "--detect-mode",
        choices=["streaming", "offline"],
        default="streaming",
        help="Detection execution mode (default: streaming)",
    )
    parser.add_argument(
        "--detect-chunk-ms",
        type=int,
        choices=[40, 80, 160],
        default=40,
        help="Streaming detection chunk size in ms (default: 40ms = 640 samples)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (default: cuda)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="AudioSeal watermark embedding strength (default: 1.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exp/eval_meanvc_1000",
        help="Output directory for benchmark results",
    )
    parser.add_argument(
        "--save-audio-count",
        type=int,
        default=50,
        help="Number of sample audio pairs to save to disk (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    # Create directories
    out_dir = SCRIPT_DIR / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    audio_dir = out_dir / "synthesized_audios"
    tmp_dir = out_dir / "tmp_parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    target_path = resolve_path(args.target_wav)

    print("=" * 100)
    print("  MeanVC2 (Streaming VC) + AudioSeal (Real-time Watermark) Benchmark")
    print("=" * 100)
    print(f"Device:             {args.device}")
    print(f"MeanVC2 Model:      {args.model}")
    print(f"MeanVC2 Chunk Size: {args.chunk_ms} ms")
    print(f"Watermark Alpha:    {args.alpha}")
    print(f"Detection Mode:     {args.detect_mode} ({args.detect_chunk_ms} ms chunks)")
    print(f"Target Speaker:     {target_path}")
    print(f"Dataset Dir:        {args.libritts_dir}")
    print(f"Num Samples:        {args.num_samples}")
    print(f"Save Audio Count:   {args.save_audio_count}")
    print(f"Output Directory:   {out_dir}")
    print("-" * 100)

    # 1. Initialize Pipeline
    print("[Pipeline] Initializing MeanVC2 Streaming VCRunner...")
    vc_runner = VCRunner(target_wav=target_path, device=args.device, model=args.model)

    print(f"[Pipeline] Initializing AudioSeal streaming watermarker on {args.device}...")
    watermarker = AudioSealStreamWatermarker(
        device=args.device,
        alpha=args.alpha,
        sample_rate=16000,
    )

    # Pre-extract target speaker embedding for SIM metric
    print("[Pipeline] Pre-extracting target speaker embedding...")
    target_spk_emb = extract_embedding(vc_runner.spk_model, target_path, device=args.device)

    # Pre-load attack suite
    print("[Pipeline] Initializing 21 robustness attack functions...")
    val_attacks = get_validation_attack_suite(sample_rate=16000)

    # 2. Collect Dataset Samples
    samples = collect_libritts_samples(
        Path(args.libritts_dir),
        num_samples=args.num_samples,
        min_duration=2.0,
        max_duration=10.0,
        seed=args.seed,
    )

    if len(samples) == 0:
        raise RuntimeError(f"No audio samples found in {args.libritts_dir}")

    # Check for existing checkpointed parts
    existing_parts = set()
    for p in tmp_dir.glob("sample_*.json"):
        try:
            sid = int(p.stem.split("_")[1])
            existing_parts.add(sid)
        except Exception:
            pass

    print(f"[Pipeline] Found {len(existing_parts)} already completed samples. Resuming from checkpoint...")

    # Aggregators for attacks and metrics
    attack_raw_records = defaultdict(lambda: {
        "category": "",
        "family": "",
        "bitrate": "",
        "bit_matches": 0,
        "total_bits": 0,
        "pos_matches": 0,
        "neg_matches": 0,
        "total_samples": 0,
        "wm_scores": [],
        "clean_scores": [],
        "gt_bits_all": [],
        "pred_bit_probs_all": [],
    })

    quality_records = {
        "pesq": [],
        "stoi": [],
        "snr": [],
        "clean_sim": [],
        "wm_sim": [],
        "embed_overhead_ms_per_sec": [],
        "detect_latency_ms_per_sec": [],
        "detect_chunk_latency_ms": [],
    }

    saved_samples_meta = []
    t_pipeline_start = time.time()

    # 3. Main Evaluation Loop
    for idx_0, (cut_id, wav_path, dur, text) in enumerate(samples):
        sample_idx = idx_0 + 1
        tmp_json = tmp_dir / f"sample_{sample_idx:04d}.json"

        if sample_idx in existing_parts and tmp_json.exists():
            with open(tmp_json, "r", encoding="utf-8") as f:
                r_data = json.load(f)
            # Replay into aggregators
            for k, v in r_data["attack_stats"].items():
                rec = attack_raw_records[k]
                rec["category"] = v["category"]
                rec["family"] = v["family"]
                rec["bitrate"] = v["bitrate"]
                rec["bit_matches"] += v["bit_matches"]
                rec["total_bits"] += v["total_bits"]
                rec["pos_matches"] += v["pos_match"]
                rec["neg_matches"] += v["neg_match"]
                rec["total_samples"] += 1
                rec["wm_scores"].append(v["wm_score"])
                rec["clean_scores"].append(v["clean_score"])
                rec["gt_bits_all"].extend(v["gt_bits"])
                rec["pred_bit_probs_all"].extend(v["pred_probs"])

            for qk in quality_records:
                if qk in r_data["quality"]:
                    quality_records[qk].append(r_data["quality"][qk])
            if "saved_paths" in r_data:
                saved_samples_meta.append(r_data["saved_paths"])
            continue

        # Load audio
        wav, sr = sf.read(wav_path)
        if wav.ndim == 2:
            wav = np.mean(wav, axis=1)
        if sr != 16000:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            sr = 16000
        wav = wav.astype(np.float32)

        # Generate unique random 16-bit message for this sample
        torch.manual_seed(args.seed * 10000 + sample_idx)
        message = torch.randint(0, 2, (1, 16), dtype=torch.int32, device=args.device)
        gt_bits_list = message.squeeze().cpu().tolist()
        msg_hex = "".join(str(b) for b in gt_bits_list)

        # Reset streaming caches
        vc_runner._init_cache()
        watermarker.reset()

        clean_chunks = []
        wm_chunks = []
        pos = 0
        total_audio_sec = len(wav) / 16000.0

        t_embed_start = time.time()
        # Streaming loop: audio chunk size (default vc_runner.CHUNK = 2560 = 160ms; or 640 = 40ms)
        chunk_samples_io = int(16000 * (args.chunk_ms / 1000.0))
        while pos < len(wav):
            end = min(pos + chunk_samples_io, len(wav))
            chunk = wav[pos:end]
            clean_chunk = vc_runner.process_chunk(chunk)
            if clean_chunk is not None and len(clean_chunk) > 0:
                clean_chunks.append(clean_chunk)
                wm_chunk = watermarker.process_chunk(clean_chunk, alpha=args.alpha, message=message)
                if wm_chunk is not None and len(wm_chunk) > 0:
                    wm_chunks.append(wm_chunk)
            pos = end

        # Drain tail vocoder overlap if any
        if vc_runner.last_wav is not None:
            tail_c = vc_runner.last_wav
            clean_chunks.append(tail_c)
            tail_w = watermarker.process_chunk(tail_c, alpha=args.alpha, message=message)
            if tail_w is not None and len(tail_w) > 0:
                wm_chunks.append(tail_w)

        embed_dur = time.time() - t_embed_start
        embed_ms_per_sec = (embed_dur / max(0.1, total_audio_sec)) * 1000.0

        clean_wav = np.concatenate(clean_chunks) if clean_chunks else np.zeros(1600, dtype=np.float32)
        wm_wav = np.concatenate(wm_chunks) if wm_chunks else np.zeros(1600, dtype=np.float32)

        # Match exact length
        min_len = min(len(clean_wav), len(wm_wav))
        clean_wav = clean_wav[:min_len]
        wm_wav = wm_wav[:min_len]

        # Prevent peak clipping using joint normalization (prevents amplitude scale mismatch)
        peak = max(float(np.max(np.abs(clean_wav))), float(np.max(np.abs(wm_wav))))
        if peak > 0.99:
            clean_wav = (clean_wav / peak) * 0.95
            wm_wav = (wm_wav / peak) * 0.95

        # 4. Calculate Speech Quality Metrics
        # PESQ
        sample_pesq = 0.0
        if pesq is not None:
            try:
                sample_pesq = float(pesq(16000, clean_wav, wm_wav, "wb"))
            except Exception:
                sample_pesq = 0.0

        # STOI
        sample_stoi = 0.0
        if stoi is not None:
            try:
                sample_stoi = float(stoi(clean_wav, wm_wav, 16000, extended=False))
            except Exception:
                sample_stoi = 0.0

        # SNR
        sample_snr = compute_snr(clean_wav, wm_wav)

        # SIM (Speaker Cosine Sim)
        emb_c = extract_embedding(vc_runner.spk_model, clean_wav, device=args.device)
        emb_w = extract_embedding(vc_runner.spk_model, wm_wav, device=args.device)
        sim_clean = float(F.cosine_similarity(emb_c, target_spk_emb).item())
        sim_wm = float(F.cosine_similarity(emb_w, target_spk_emb).item())

        sample_quality = {
            "pesq": sample_pesq,
            "stoi": sample_stoi,
            "snr": sample_snr,
            "clean_sim": sim_clean,
            "wm_sim": sim_wm,
            "embed_overhead_ms_per_sec": embed_ms_per_sec,
            "detect_latency_ms_per_sec": 0.0,
            "detect_chunk_latency_ms": 0.0,
        }

        # 5. Evaluate Robustness Attacks
        det_chunk_samples = int(16000 * (args.detect_chunk_ms / 1000.0))
        t_det_start = time.time()
        # Convert to torch tensor [1, 1, T]
        t_clean = torch.from_numpy(clean_wav).float().unsqueeze(0).unsqueeze(0).to(args.device)
        t_wm = torch.from_numpy(wm_wav).float().unsqueeze(0).unsqueeze(0).to(args.device)

        sample_attack_stats = {}
        sample_chunk_lats = []
        for cat, name, detail, atk_fn in val_attacks:
            key = name if cat == "DSP" else f"{name} {detail}"

            # 1. Attacked Watermarked audio
            try:
                atk_wm = atk_fn(t_wm)
            except Exception:
                atk_wm = t_wm

            # 2. Attacked Clean audio (negative baseline for ROC-AUC & Det ACC)
            try:
                atk_clean = atk_fn(t_clean)
            except Exception:
                atk_clean = t_clean

            if args.detect_mode == "streaming":
                wm_score, pred_msg, t_wm_dur, wm_lat = run_streaming_detection(
                    watermarker.detector, atk_wm, chunk_samples=det_chunk_samples
                )
                clean_score, _, t_cl_dur, _ = run_streaming_detection(
                    watermarker.detector, atk_clean, chunk_samples=det_chunk_samples
                )
                sample_chunk_lats.append(wm_lat)
            else:
                with torch.no_grad():
                    res_wm, pred_msg = watermarker.detector(atk_wm)
                    res_clean, _ = watermarker.detector(atk_clean)
                wm_score = float(res_wm[:, 1, :].mean().item())
                clean_score = float(res_clean[:, 1, :].mean().item())
                sample_chunk_lats.append(0.0)

            # Predicted bits from sigmoid message
            pred_probs = pred_msg.squeeze().cpu().tolist()
            if isinstance(pred_probs, float):
                pred_probs = [pred_probs]
            pred_bits = [int(p > 0.5) for p in pred_probs]

            bit_matches = sum(int(pb == gb) for pb, gb in zip(pred_bits, gt_bits_list))
            pos_match = int(wm_score > 0.5)
            neg_match = int(clean_score <= 0.5)

            sample_attack_stats[key] = {
                "category": cat,
                "family": name,
                "bitrate": detail,
                "bit_matches": bit_matches,
                "total_bits": 16,
                "pos_match": pos_match,
                "neg_match": neg_match,
                "wm_score": wm_score,
                "clean_score": clean_score,
                "gt_bits": gt_bits_list,
                "pred_probs": pred_probs,
            }

            # Accumulate into running records
            rec = attack_raw_records[key]
            rec["category"] = cat
            rec["family"] = name
            rec["bitrate"] = detail
            rec["bit_matches"] += bit_matches
            rec["total_bits"] += 16
            rec["pos_matches"] += pos_match
            rec["neg_matches"] += neg_match
            rec["total_samples"] += 1
            rec["wm_scores"].append(wm_score)
            rec["clean_scores"].append(clean_score)
            rec["gt_bits_all"].extend(gt_bits_list)
            rec["pred_bit_probs_all"].extend(pred_probs)

        det_dur = time.time() - t_det_start
        detect_ms_per_sec = (det_dur / (len(val_attacks) * max(0.1, total_audio_sec))) * 1000.0
        sample_quality["detect_latency_ms_per_sec"] = detect_ms_per_sec
        sample_quality["detect_chunk_latency_ms"] = float(np.mean(sample_chunk_lats)) if sample_chunk_lats else 0.0

        for qk in quality_records:
            if qk in sample_quality:
                quality_records[qk].append(sample_quality[qk])

        # 6. Save Audio Pair if within quota
        saved_paths = None
        if sample_idx <= args.save_audio_count:
            p_clean = audio_dir / f"sample_{sample_idx:04d}_{cut_id}_clean.wav"
            p_wm = audio_dir / f"sample_{sample_idx:04d}_{cut_id}_wm.wav"
            sf.write(str(p_clean), clean_wav, 16000)
            sf.write(str(p_wm), wm_wav, 16000)
            saved_paths = {
                "sample_idx": sample_idx,
                "cut_id": cut_id,
                "clean_wav": str(p_clean),
                "wm_wav": str(p_wm),
                "duration": total_audio_sec,
                "watermark_hex": msg_hex,
            }
            saved_samples_meta.append(saved_paths)

        # Write checkpoint part
        sample_res = {
            "sample_idx": sample_idx,
            "cut_id": cut_id,
            "duration": total_audio_sec,
            "text": text,
            "watermark_hex": msg_hex,
            "quality": sample_quality,
            "attack_stats": sample_attack_stats,
        }
        if saved_paths:
            sample_res["saved_paths"] = saved_paths

        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(sample_res, f)

        # Progress log
        if sample_idx % 10 == 0 or sample_idx == len(samples):
            clean_sim_avg = np.mean(quality_records["clean_sim"])
            wm_sim_avg = np.mean(quality_records["wm_sim"])
            pesq_avg = np.mean(quality_records["pesq"])
            snr_avg = np.mean(quality_records["snr"])
            clean_atk_acc = (
                attack_raw_records["Clean (Identity)"]["bit_matches"]
                / max(1, attack_raw_records["Clean (Identity)"]["total_bits"])
            )
            print(
                f"[{sample_idx:04d}/{len(samples)}] Dur: {total_audio_sec:.2f}s | "
                f"Clean WM Bit Acc: {clean_atk_acc * 100:.1f}% | "
                f"PESQ: {pesq_avg:.3f} | SNR: {snr_avg:.1f}dB | "
                f"SIM: {clean_sim_avg:.3f}->{wm_sim_avg:.3f}",
                flush=True,
            )

    # 7. Aggregate Full Results & Format Benchmark Report
    total_eval_time = time.time() - t_pipeline_start
    print("\n[Aggregating] Computing overall ROC-AUC, TPR@0.1% FPR, and summarizing report...")

    summary_table_data = {}
    for key, rec in attack_raw_records.items():
        total_samples = rec["total_samples"]
        # Detection Accuracy
        det_acc = (rec["pos_matches"] + rec["neg_matches"]) / max(1, 2 * total_samples)

        # Detection ROC-AUC & TPR@0.1%
        y_det_true = [1] * len(rec["wm_scores"]) + [0] * len(rec["clean_scores"])
        y_det_scores = rec["wm_scores"] + rec["clean_scores"]
        det_auc, det_tpr = compute_auc_and_tpr_at_fpr(y_det_true, y_det_scores, target_fpr=0.001)

        # Message Bit Accuracy
        bit_acc = rec["bit_matches"] / max(1, rec["total_bits"])

        # Message ROC-AUC & TPR@0.1%
        wm_auc, wm_tpr = compute_auc_and_tpr_at_fpr(rec["gt_bits_all"], rec["pred_bit_probs_all"], target_fpr=0.001)

        summary_table_data[key] = {
            "category": rec["category"],
            "family": rec["family"],
            "bitrate": rec["bitrate"],
            "detect_acc": det_acc,
            "det_roc_auc": det_auc,
            "det_tpr_at_001_fpr": det_tpr,
            "bit_acc": bit_acc,
            "wm_roc_auc": wm_auc,
            "wm_tpr_at_001_fpr": wm_tpr,
        }

    overall_quality = {
        "pesq": float(np.mean(quality_records["pesq"])),
        "stoi": float(np.mean(quality_records["stoi"])),
        "snr": float(np.mean(quality_records["snr"])),
        "clean_sim": float(np.mean(quality_records["clean_sim"])),
        "wm_sim": float(np.mean(quality_records["wm_sim"])),
        "embed_overhead_ms_per_sec": float(np.mean(quality_records["embed_overhead_ms_per_sec"])),
        "detect_latency_ms_per_sec": float(np.mean(quality_records["detect_latency_ms_per_sec"])),
        "embed_rtf": float(np.mean(quality_records["embed_overhead_ms_per_sec"])) / 1000.0,
        "detect_rtf": float(np.mean(quality_records["detect_latency_ms_per_sec"])) / 1000.0,
        "detect_mode": f"Streaming ({args.detect_chunk_ms}ms chunk)" if args.detect_mode == "streaming" else "Offline",
        "detect_chunk_ms": args.detect_chunk_ms,
        "detect_chunk_latency_ms": float(np.mean(quality_records["detect_chunk_latency_ms"])) if quality_records["detect_chunk_latency_ms"] else 0.0,
    }

    report_table = format_full_validation_table(
        step=f"MeanVC2-{args.model} + AudioSeal (1000 LibriTTS train-clean)",
        results=summary_table_data,
        quality_metrics=overall_quality,
    )

    print("\n" + report_table + "\n", flush=True)

    # Save to disk
    report_file = out_dir / "validation_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_table + "\n")

    summary_json_file = out_dir / "eval_summary.json"
    with open(summary_json_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "alpha": args.alpha,
            "device": args.device,
            "target_wav": target_path,
            "num_evaluated_samples": len(samples),
            "total_runtime_seconds": round(total_eval_time, 2),
            "overall_quality_metrics": overall_quality,
            "robustness_summary": summary_table_data,
            "saved_audios_count": len(saved_samples_meta),
            "saved_audios": saved_samples_meta,
        }, f, indent=4)

    print("=" * 100)
    print(f"[Done] Benchmark complete in {total_eval_time / 60.0:.2f} minutes.")
    print(f"Report written to:       {report_file}")
    print(f"Summary JSON written to: {summary_json_file}")
    print(f"Sample WAVs saved in:    {audio_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
