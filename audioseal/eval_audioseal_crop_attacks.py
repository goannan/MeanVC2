#!/usr/bin/env python3
# Copyright (c) 2026 MeanVC2 AudioSeal Integration. All rights reserved.
"""
AudioSeal Streaming Watermark Robustness Benchmark:
1. Temporal Cropping Evaluation: Random cropping across 1, 2, 5, 10, 20 frames.
2. Robustness Attack Evaluation on 20-frame crops: 21 DSP & Neural Codec attacks
   matching VALL-E Neumark standard (eval_valle_test_neumark.pjm).
3. Evaluates on 1,000 utterances from LibriTTS test-clean.
4. Voice Conversion using MeanVC2 160ms model (120ms VC + 160ms ASR chunk).
5. Strictly causal, streaming chunk-by-chunk embedding.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
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


def collect_libritts_test_samples(
    dataset_dir: Path,
    num_samples: int = 1000,
    min_duration: float = 2.0,
    max_duration: float = 10.0,
    seed: int = 42,
) -> List[Tuple[str, str, float, str]]:
    """
    Collect audio samples from LibriTTS test-clean directory.
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


def format_cropping_table(
    crop_stats: Dict[int, Dict[str, float]],
    target_label: str = "AudioSeal-Streaming (MeanVC2 160ms)",
) -> str:
    """Format a summary table for random temporal cropping results."""
    hdr_line = "=" * 125
    div_line = "-" * 125
    lines = [
        hdr_line,
        f"  Temporal Cropping Benchmark Report (Target: {target_label})",
        hdr_line,
        f"{'Crop Frames':<14} | {'Duration (ms)':<14} | {'Samples':<9} | {'Detect ACC':<11} | {'Det ROC-AUC':<11} | {'Det TPR@0.1%':<12} | {'WM Bit Acc':<11} | {'WM ROC-AUC':<11} | {'WM TPR@0.1%':<12}",
        div_line,
    ]

    for frames in sorted(crop_stats.keys()):
        st = crop_stats[frames]
        dur_ms = st["duration_ms"]
        n_samples = st["samples"]
        d_acc = st["detect_acc"]
        d_auc = st["det_roc_auc"]
        d_tpr = st["det_tpr_at_001_fpr"]
        w_bit = st["bit_acc"]
        w_auc = st["wm_roc_auc"]
        w_tpr = st["wm_tpr_at_001_fpr"]

        lbl = f"{frames} frames" if frames > 1 else f"{frames} frame"
        dur_str = f"{dur_ms:.1f} ms"
        lines.append(
            f"{lbl:<14} | {dur_str:<14} | {n_samples:<9} | {d_acc:<11.4f} | {d_auc:<11.4f} | {d_tpr:<12.4f} | {w_bit:<11.4f} | {w_auc:<11.4f} | {w_tpr:<12.4f}"
        )

    lines.append(hdr_line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="AudioSeal Streaming Watermark Robustness Benchmark: Cropping + 20-Frame Attacks"
    )
    parser.add_argument(
        "--libritts-dir",
        type=str,
        default="/home/pj25001109/ku60000344/dataset/libriTTS/LibriTTS/test-clean",
        help="Path to LibriTTS test-clean dataset directory",
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
        help="Number of test samples to evaluate (default: 1000)",
    )
    parser.add_argument(
        "--model",
        choices=["40ms", "120ms"],
        default="120ms",
        help="MeanVC2 model config (default: 120ms, which uses the 160ms ASR chunk model)",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        choices=[40, 160],
        default=160,
        help="MeanVC2 streaming I/O chunk size in ms (default: 160ms = 2560 samples)",
    )
    parser.add_argument(
        "--crop-frames",
        nargs="+",
        type=int,
        default=[1, 2, 5, 10, 20],
        help="Frame counts to evaluate random temporal cropping (default: 1 2 5 10 20)",
    )
    parser.add_argument(
        "--frame-samples",
        type=int,
        default=160,
        help="Number of audio samples per speech frame at 16kHz (default: 160 = 10ms)",
    )
    parser.add_argument(
        "--attack-crop-frames",
        type=int,
        default=20,
        help="Crop size in frames used for the 21 robustness attacks suite (default: 20 frames = 3200 samples = 200ms)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (default: cuda if available else cpu)",
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
        default="exp/eval_audioseal_crop_attacks",
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
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Worker rank index for multi-GPU evaluation (default: 0)",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Total worker count / GPUs for multi-GPU evaluation (default: 1)",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip worker inference; aggregate all tmp_parts/sample_*.json into final report",
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
    print("  AudioSeal Streaming Watermark Robustness Benchmark (Cropping & 20-Frame Attacks)")
    print("==================================================================================")
    print(f"Device:             {args.device}")
    print(f"MeanVC2 Model:      {args.model} (160ms ASR chunk / 120ms lookahead)")
    print(f"MeanVC2 Chunk Size: {args.chunk_ms} ms ({int(16000 * (args.chunk_ms / 1000.0))} samples)")
    print(f"Watermark Alpha:    {args.alpha}")
    print(f"Cropping Frames:    {args.crop_frames} (Frame hop: {args.frame_samples} samples = {args.frame_samples/16.0:.1f}ms)")
    print(f"Attack Crop Frames: {args.attack_crop_frames} frames ({args.attack_crop_frames * args.frame_samples} samples = {args.attack_crop_frames * args.frame_samples / 16.0:.1f}ms)")
    print(f"Target Speaker:     {target_path}")
    print(f"Dataset Dir:        {args.libritts_dir}")
    print(f"Num Samples:        {args.num_samples}")
    print(f"Save Audio Count:   {args.save_audio_count}")
    print(f"Output Directory:   {out_dir}")
    print("-" * 100)

    # 1. Initialize Pipeline (skip if aggregating only)
    if not args.aggregate_only:
        print(f"[Pipeline] Initializing MeanVC2 Streaming VCRunner (160ms model) on rank {args.rank}...")
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
        print("[Pipeline] Initializing 21 robustness attack functions matching VALL-E Neumark standard...")
        val_attacks = get_validation_attack_suite(sample_rate=16000)
    else:
        vc_runner = None
        watermarker = None
        target_spk_emb = None
        val_attacks = None

    # 2. Collect Dataset Samples
    samples = collect_libritts_test_samples(
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

    # Aggregators for Cropping Evaluation
    # Key: frames (e.g. 1, 2, 5, 10, 20)
    crop_raw_records = {
        f: {
            "frames": f,
            "duration_ms": f * (args.frame_samples / 16.0),
            "samples": f * args.frame_samples,
            "bit_matches": 0,
            "total_bits": 0,
            "pos_matches": 0,
            "neg_matches": 0,
            "total_samples": 0,
            "wm_scores": [],
            "clean_scores": [],
            "gt_bits_all": [],
            "pred_bit_probs_all": [],
        }
        for f in args.crop_frames
    }

    # Aggregators for 20-frame Attacks
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
    }

    saved_samples_meta = []
    t_pipeline_start = time.time()
    chunk_samples_io = int(16000 * (args.chunk_ms / 1000.0))

    # 3. Main Evaluation Loop
    for idx_0, (cut_id, wav_path, dur, text) in enumerate(samples):
        sample_idx = idx_0 + 1
        tmp_json = tmp_dir / f"sample_{sample_idx:04d}.json"

        # In multi-worker mode, only process assigned samples unless in aggregate-only mode
        if not args.aggregate_only and (sample_idx - 1) % args.world_size != args.rank:
            continue

        # Checkpoint replay
        if sample_idx in existing_parts and tmp_json.exists():
            with open(tmp_json, "r", encoding="utf-8") as f:
                r_data = json.load(f)

            # Replay cropping stats
            for str_f, v in r_data["crop_stats"].items():
                f_int = int(str_f)
                if f_int in crop_raw_records:
                    rec = crop_raw_records[f_int]
                    rec["bit_matches"] += v["bit_matches"]
                    rec["total_bits"] += v["total_bits"]
                    rec["pos_matches"] += v["pos_match"]
                    rec["neg_matches"] += v["neg_match"]
                    rec["total_samples"] += 1
                    rec["wm_scores"].append(v["wm_score"])
                    rec["clean_scores"].append(v["clean_score"])
                    rec["gt_bits_all"].extend(v["gt_bits"])
                    rec["pred_bit_probs_all"].extend(v["pred_probs"])

            # Replay attack stats
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

        if args.aggregate_only:
            # If aggregate-only mode, skip samples not yet computed
            continue

        # Load audio waveform
        wav, sr = sf.read(wav_path)
        if wav.ndim == 2:
            wav = np.mean(wav, axis=1)
        if sr != 16000:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            sr = 16000
        wav = wav.astype(np.float32)

        # Unique 16-bit binary payload for this sample
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
        # Strictly causal, chunk-by-chunk streaming conversion & watermark embedding
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

        # Drain tail vocoder overlap
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

        min_len = min(len(clean_wav), len(wm_wav))
        clean_wav = clean_wav[:min_len]
        wm_wav = wm_wav[:min_len]

        # Joint peak normalization to avoid amplitude clipping
        peak = max(float(np.max(np.abs(clean_wav))), float(np.max(np.abs(wm_wav))))
        if peak > 0.99:
            clean_wav = (clean_wav / peak) * 0.95
            wm_wav = (wm_wav / peak) * 0.95

        # 4. Speech Quality Metrics on Full Audio
        sample_pesq = 0.0
        if pesq is not None:
            try:
                sample_pesq = float(pesq(16000, clean_wav, wm_wav, "wb"))
            except Exception:
                sample_pesq = 0.0

        sample_stoi = 0.0
        if stoi is not None:
            try:
                sample_stoi = float(stoi(clean_wav, wm_wav, 16000, extended=False))
            except Exception:
                sample_stoi = 0.0

        sample_snr = compute_snr(clean_wav, wm_wav)

        emb_c = extract_embedding(vc_runner.spk_model, clean_wav, device=args.device)
        emb_w = extract_embedding(vc_runner.spk_model, wm_wav, device=args.device)
        sim_clean = float(F.cosine_similarity(emb_c, target_spk_emb).item())
        sim_wm = float(F.cosine_similarity(emb_w, target_spk_emb).item())

        # 5. Part 1: Temporal Cropping Evaluation (1, 2, 5, 10, 20 frames)
        sample_crop_stats = {}
        # Ensure repeatable random cropping per sample
        crop_rng = random.Random(args.seed * 10000 + sample_idx)

        # Cache extracted 20-frame crop for Part 2
        crop_20f_wm = None
        crop_20f_clean = None

        for f_count in args.crop_frames:
            crop_len = f_count * args.frame_samples
            if len(wm_wav) < crop_len:
                # If audio is shorter than crop_len, pad with zeros
                pad_amt = crop_len - len(wm_wav)
                c_wm = np.pad(wm_wav, (0, pad_amt))
                c_clean = np.pad(clean_wav, (0, pad_amt))
            else:
                start_max = len(wm_wav) - crop_len
                start_pos = crop_rng.randint(0, start_max)
                c_wm = wm_wav[start_pos : start_pos + crop_len]
                c_clean = clean_wav[start_pos : start_pos + crop_len]

            if f_count == args.attack_crop_frames:
                crop_20f_wm = c_wm.copy()
                crop_20f_clean = c_clean.copy()

            # Run detection on crop
            prob_wm, dec_bits_wm = watermarker.detect(c_wm)
            prob_clean, _ = watermarker.detect(c_clean)

            # Get raw predicted bit probabilities by running forward feature through detector
            t_wm_crop = torch.from_numpy(c_wm).float().unsqueeze(0).unsqueeze(0).to(args.device)
            with torch.no_grad():
                _, pred_msg_tensor = watermarker.detector(t_wm_crop)
            pred_probs = pred_msg_tensor.squeeze().cpu().tolist()
            if isinstance(pred_probs, float):
                pred_probs = [pred_probs]

            bit_matches = sum(int(pb == gb) for pb, gb in zip(dec_bits_wm, gt_bits_list))
            pos_match = int(prob_wm > 0.5)
            neg_match = int(prob_clean <= 0.5)

            sample_crop_stats[str(f_count)] = {
                "frames": f_count,
                "duration_ms": f_count * (args.frame_samples / 16.0),
                "samples": crop_len,
                "bit_matches": bit_matches,
                "total_bits": 16,
                "pos_match": pos_match,
                "neg_match": neg_match,
                "wm_score": prob_wm,
                "clean_score": prob_clean,
                "gt_bits": gt_bits_list,
                "pred_probs": pred_probs,
            }

            # Accumulate into running crop records
            rec = crop_raw_records[f_count]
            rec["bit_matches"] += bit_matches
            rec["total_bits"] += 16
            rec["pos_matches"] += pos_match
            rec["neg_matches"] += neg_match
            rec["total_samples"] += 1
            rec["wm_scores"].append(prob_wm)
            rec["clean_scores"].append(prob_clean)
            rec["gt_bits_all"].extend(gt_bits_list)
            rec["pred_bit_probs_all"].extend(pred_probs)

        # Fallback if 20-frame crop was not in crop_frames
        if crop_20f_wm is None:
            c_len_20 = args.attack_crop_frames * args.frame_samples
            if len(wm_wav) < c_len_20:
                crop_20f_wm = np.pad(wm_wav, (0, c_len_20 - len(wm_wav)))
                crop_20f_clean = np.pad(clean_wav, (0, c_len_20 - len(clean_wav)))
            else:
                sp = crop_rng.randint(0, len(wm_wav) - c_len_20)
                crop_20f_wm = wm_wav[sp : sp + c_len_20]
                crop_20f_clean = clean_wav[sp : sp + c_len_20]

        # 6. Part 2: Robustness Attacks on 20-frame Crop (3200 samples)
        t_det_start = time.time()
        t_20f_wm = torch.from_numpy(crop_20f_wm).float().unsqueeze(0).unsqueeze(0).to(args.device)
        t_20f_clean = torch.from_numpy(crop_20f_clean).float().unsqueeze(0).unsqueeze(0).to(args.device)

        sample_attack_stats = {}
        for cat, name, detail, atk_fn in val_attacks:
            key = name if cat == "DSP" else f"{name} {detail}"

            try:
                atk_wm = atk_fn(t_20f_wm)
            except Exception:
                atk_wm = t_20f_wm

            try:
                atk_clean = atk_fn(t_20f_clean)
            except Exception:
                atk_clean = t_20f_clean

            # Run detector on attacked 20-frame crops
            with torch.no_grad():
                res_wm, pred_msg_atk = watermarker.detector(atk_wm)
                res_clean, _ = watermarker.detector(atk_clean)

            wm_score = float(res_wm[:, 1, :].mean().item())
            clean_score = float(res_clean[:, 1, :].mean().item())

            pred_probs = pred_msg_atk.squeeze().cpu().tolist()
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

            # Accumulate into attack records
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
        detect_ms_per_sec = (det_dur / (len(val_attacks) * max(0.1, len(crop_20f_wm) / 16000.0))) * 1000.0

        sample_quality = {
            "pesq": sample_pesq,
            "stoi": sample_stoi,
            "snr": sample_snr,
            "clean_sim": sim_clean,
            "wm_sim": sim_wm,
            "embed_overhead_ms_per_sec": embed_ms_per_sec,
            "detect_latency_ms_per_sec": detect_ms_per_sec,
        }

        for qk in quality_records:
            if qk in sample_quality:
                quality_records[qk].append(sample_quality[qk])

        # 7. Save Sample Audios if within quota
        saved_paths = None
        if sample_idx <= args.save_audio_count:
            p_clean = audio_dir / f"sample_{sample_idx:04d}_{cut_id}_clean.wav"
            p_wm = audio_dir / f"sample_{sample_idx:04d}_{cut_id}_wm.wav"
            p_crop20 = audio_dir / f"sample_{sample_idx:04d}_{cut_id}_wm_20f.wav"
            sf.write(str(p_clean), clean_wav, 16000)
            sf.write(str(p_wm), wm_wav, 16000)
            sf.write(str(p_crop20), crop_20f_wm, 16000)
            saved_paths = {
                "sample_idx": sample_idx,
                "cut_id": cut_id,
                "clean_wav": str(p_clean),
                "wm_wav": str(p_wm),
                "crop_20f_wav": str(p_crop20),
                "duration": total_audio_sec,
                "watermark_hex": msg_hex,
            }
            saved_samples_meta.append(saved_paths)

        # 8. Write Checkpoint Part
        checkpoint_entry = {
            "sample_idx": sample_idx,
            "cut_id": cut_id,
            "duration": total_audio_sec,
            "text": text,
            "watermark_hex": msg_hex,
            "crop_stats": sample_crop_stats,
            "attack_stats": sample_attack_stats,
            "quality": sample_quality,
        }
        if saved_paths:
            checkpoint_entry["saved_paths"] = saved_paths

        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(checkpoint_entry, f)

        # Progress log
        if sample_idx % 10 == 0 or sample_idx == len(samples) or sample_idx <= 5:
            elapsed_tot = time.time() - t_pipeline_start
            ips = sample_idx / max(0.1, elapsed_tot)
            eta_sec = (len(samples) - sample_idx) / max(0.01, ips)
            print(
                f"[Rank {args.rank}][{sample_idx:04d}/{len(samples)}] Dur: {total_audio_sec:.2f}s | "
                f"20f Clean: {clean_score:.3f}, WM: {wm_score:.3f} | "
                f"PESQ: {sample_pesq:.2f}, SIM: {sim_wm:.3f} | "
                f"Speed: {ips:.2f} samples/s, ETA: {eta_sec/60.0:.1f}m",
                flush=True,
            )

    # In multi-worker execution, workers terminate here once their shard is completed
    if args.world_size > 1 and not args.aggregate_only:
        print(f"[Worker Rank {args.rank}/{args.world_size}] Shard completed successfully!", flush=True)
        return

    # -------------------------------------------------------------------------
    # 4. Aggregation and Final Statistics
    # -------------------------------------------------------------------------
    print("\n[Pipeline] Aggregating benchmark results across all evaluated samples...")

    # A. Aggregate Cropping Stats
    crop_final_summary = {}
    for f_count, stats in crop_raw_records.items():
        n_samples = stats["total_samples"]
        if n_samples == 0:
            continue

        bit_acc = stats["bit_matches"] / max(1, stats["total_bits"])
        pos_acc = stats["pos_matches"] / max(1, n_samples)
        neg_acc = stats["neg_matches"] / max(1, n_samples)
        det_acc = (stats["pos_matches"] + stats["neg_matches"]) / max(1, 2 * n_samples)

        y_true_det = [1] * len(stats["wm_scores"]) + [0] * len(stats["clean_scores"])
        y_scores_det = stats["wm_scores"] + stats["clean_scores"]
        det_roc_auc, det_tpr_at_001 = compute_auc_and_tpr_at_fpr(y_true_det, y_scores_det, target_fpr=0.001)

        wm_roc_auc, wm_tpr_at_001 = compute_auc_and_tpr_at_fpr(
            stats["gt_bits_all"], stats["pred_bit_probs_all"], target_fpr=0.001
        )

        crop_final_summary[f_count] = {
            "frames": f_count,
            "duration_ms": stats["duration_ms"],
            "samples": stats["samples"],
            "num_evaluated": n_samples,
            "bit_acc": bit_acc,
            "pos_acc": pos_acc,
            "neg_acc": neg_acc,
            "detect_acc": det_acc,
            "det_roc_auc": det_roc_auc,
            "det_tpr_at_001_fpr": det_tpr_at_001,
            "wm_roc_auc": wm_roc_auc,
            "wm_tpr_at_001_fpr": wm_tpr_at_001,
        }

    # B. Aggregate 20-frame Attack Stats
    attack_final_summary = {}
    for key, stats in attack_raw_records.items():
        n_samples = stats["total_samples"]
        if n_samples == 0:
            continue

        bit_acc = stats["bit_matches"] / max(1, stats["total_bits"])
        pos_acc = stats["pos_matches"] / max(1, n_samples)
        neg_acc = stats["neg_matches"] / max(1, n_samples)
        det_acc = (stats["pos_matches"] + stats["neg_matches"]) / max(1, 2 * n_samples)

        y_true_det = [1] * len(stats["wm_scores"]) + [0] * len(stats["clean_scores"])
        y_scores_det = stats["wm_scores"] + stats["clean_scores"]
        det_roc_auc, det_tpr_at_001 = compute_auc_and_tpr_at_fpr(y_true_det, y_scores_det, target_fpr=0.001)

        wm_roc_auc, wm_tpr_at_001 = compute_auc_and_tpr_at_fpr(
            stats["gt_bits_all"], stats["pred_bit_probs_all"], target_fpr=0.001
        )

        attack_final_summary[key] = {
            "category": stats["category"],
            "family": stats["family"],
            "bitrate": stats["bitrate"],
            "bit_acc": bit_acc,
            "pos_acc": pos_acc,
            "neg_acc": neg_acc,
            "detect_acc": det_acc,
            "det_roc_auc": det_roc_auc,
            "det_tpr_at_001_fpr": det_tpr_at_001,
            "wm_roc_auc": wm_roc_auc,
            "wm_tpr_at_001_fpr": wm_tpr_at_001,
        }

    # C. Average Speech Quality
    avg_quality = {}
    for qk, vals in quality_records.items():
        avg_quality[qk] = float(np.mean(vals)) if vals else 0.0

    # -------------------------------------------------------------------------
    # 5. Format & Print Reports
    # -------------------------------------------------------------------------
    crop_report = format_cropping_table(
        crop_final_summary,
        target_label=f"AudioSeal-Streaming (MeanVC2 {args.model}, 160ms chunk)",
    )

    attack_report = format_full_validation_table(
        step=f"AudioSeal Streaming on 20-Frame Crop (MeanVC2 {args.model})",
        results=attack_final_summary,
        quality_metrics={
            "pesq": avg_quality.get("pesq", 0.0),
            "stoi": avg_quality.get("stoi", 0.0),
            "snr": avg_quality.get("snr", 0.0),
            "clean_sim": avg_quality.get("clean_sim", 0.0),
            "wm_sim": avg_quality.get("wm_sim", 0.0),
            "embed_overhead_ms_per_sec": avg_quality.get("embed_overhead_ms_per_sec", 0.0),
            "detect_latency_ms_per_sec": avg_quality.get("detect_latency_ms_per_sec", 0.0),
        },
    )

    full_report = (
        "=============================================================================================================================\n"
        "  AUDIOSEAL STREAMING WATERMARK EVALUATION REPORT\n"
        f"  Model: MeanVC2 ({args.model} / 160ms ASR chunk) | Dataset: LibriTTS test-clean ({len(samples)} samples)\n"
        "=============================================================================================================================\n\n"
        + crop_report
        + "\n\n"
        + attack_report
    )

    print("\n" + full_report + "\n", flush=True)

    # Save to disk
    report_file = out_dir / "validation_report.txt"
    report_file.write_text(full_report, encoding="utf-8")

    summary_file = out_dir / "eval_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meanvc_model": args.model,
                "chunk_ms": args.chunk_ms,
                "dataset_dir": args.libritts_dir,
                "num_samples": len(samples),
                "crop_frames": args.crop_frames,
                "frame_samples": args.frame_samples,
                "attack_crop_frames": args.attack_crop_frames,
                "alpha": args.alpha,
                "cropping_summary": crop_final_summary,
                "attack_robustness_summary": attack_final_summary,
                "average_quality_metrics": avg_quality,
                "saved_audio_samples": saved_samples_meta,
            },
            f,
            indent=4,
        )

    print(f"[Done] All evaluations completed successfully!")
    print(f"Validation Report: {report_file}")
    print(f"Summary JSON:      {summary_file}")
    print(f"Synthesized Audio: {audio_dir}")


if __name__ == "__main__":
    main()
