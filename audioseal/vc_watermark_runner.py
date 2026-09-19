# Copyright (c) 2026 MeanVC2 AudioSeal Integration. All rights reserved.
"""
Integrated MeanVC2 Streaming Voice Conversion with AudioSeal Watermarking.

Combines MeanVC2 VCRunner (ASR -> DiT Flow Matching -> Vocos) with
AudioSeal real-time streaming causal watermarking.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

# Ensure MeanVC2 root and runtime/src are in sys.path
_CURRENT_DIR = Path(__file__).resolve().parent
_MEANVC_ROOT = _CURRENT_DIR.parent
_RUNTIME_DIR = _MEANVC_ROOT / "runtime"
_RUNTIME_SRC_DIR = _RUNTIME_DIR / "src"

for p in [str(_MEANVC_ROOT), str(_RUNTIME_DIR), str(_RUNTIME_SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from watermarker import AudioSealStreamWatermarker

_import_err = None
# Attempt to import VCRunner from MeanVC2 runtime
try:
    from run_rt import VCRunner, DEFAULT_TARGET_WAV, MODEL_PATHS
except Exception as e:
    VCRunner = None
    DEFAULT_TARGET_WAV = None
    MODEL_PATHS = None
    _import_err = e


class WatermarkedVCRunner:
    """
    Wrapper for MeanVC2 VCRunner with integrated AudioSeal real-time streaming watermark.
    """

    def __init__(
        self,
        target_wav: str,
        device: str = "cpu",
        model: str = "120ms",
        alpha: float = 1.0,
        secret_message: Optional[Union[torch.Tensor, list[int]]] = None,
        watermark_device: Optional[str] = None,
    ):
        """
        Initialize the Watermarked VC Pipeline.

        Args:
            target_wav: Path to target reference speaker WAV.
            device: Compute device for MeanVC2 ('cpu' or 'cuda').
            model: MeanVC2 model config ('120ms' or '40ms').
            alpha: Watermark embedding gain (default: 1.0).
            secret_message: 16-bit binary message to embed (list of 16 ints or Tensor).
            watermark_device: Compute device for AudioSeal (defaults to same as device).
        """
        if VCRunner is None:
            raise ImportError(
                f"Could not import VCRunner from MeanVC2/runtime/run_rt.py: {_import_err}. "
                "Ensure MeanVC2 dependencies and model checkpoints are set up."
            )

        print("[WatermarkedVC] Initializing MeanVC2 VCRunner...")
        self.vc = VCRunner(target_wav=target_wav, device=device, model=model)

        wm_dev = watermark_device if watermark_device is not None else device
        print(f"[WatermarkedVC] Initializing AudioSeal streaming watermarker on {wm_dev}...")
        self.watermarker = AudioSealStreamWatermarker(
            device=wm_dev,
            default_message=secret_message,
            alpha=alpha,
            sample_rate=16000,
        )

        self.alpha = alpha
        self.secret_message = self.watermarker.default_message

    def _init_cache(self):
        """Reset internal caches for both MeanVC2 and AudioSeal."""
        self.vc._init_cache()
        self.watermarker.reset()

    def reset(self):
        """Reset conversation session (clean transition to next utterance)."""
        self._init_cache()

    def process_chunk(self, samples: np.ndarray) -> np.ndarray | None:
        """
        Processes a single input audio chunk through the entire pipeline:
        Microphone Audio -> MeanVC2 (ASR -> DiT -> Vocos) -> AudioSeal Watermark -> Output

        Args:
            samples: 16kHz float32 input microphone/audio chunk.

        Returns:
            Watermarked converted audio chunk as 1D np.ndarray (or None if accumulating frames).
        """
        # 1. Forward through MeanVC2
        converted_chunk = self.vc.process_chunk(samples)
        if converted_chunk is None or len(converted_chunk) == 0:
            return None

        # 2. Forward through AudioSeal causal streaming watermarker
        watermarked_chunk = self.watermarker.process_chunk(
            converted_chunk, alpha=self.alpha, message=self.secret_message
        )

        return watermarked_chunk

    def process_file(
        self,
        input_path: str,
        output_path: str,
        verify_watermark: bool = True,
        seed: int = 42,
    ):
        """
        Process an entire WAV file chunk-by-chunk using the streaming pipeline,
        embed watermarks in real-time, and optionally verify extraction accuracy.
        """
        import soundfile as sf

        wav, sr = sf.read(input_path)
        if wav.ndim == 2:
            wav = np.mean(wav, axis=1)
        if sr != 16000:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            sr = 16000
        wav = wav.astype(np.float32)

        self._init_cache()
        torch.manual_seed(seed)

        output_parts = []
        pos = 0
        total_duration = len(wav) / sr
        print(f"[WatermarkedVC] Streaming {total_duration:.2f}s file with chunk size {self.vc.CHUNK}...")
        t_start = time.time()

        while pos < len(wav):
            end = min(pos + self.vc.CHUNK, len(wav))
            chunk = wav[pos:end]
            out_chunk = self.process_chunk(chunk)
            if out_chunk is not None and len(out_chunk) > 0:
                output_parts.append(out_chunk)
            pos = end

        # Drain remaining buffer in MeanVC2 vocoder if any
        # (hand-off final trailing vocoder frames to watermarker)
        if self.vc.last_wav is not None:
            tail_wm = self.watermarker.process_chunk(self.vc.last_wav, alpha=self.alpha)
            if tail_wm is not None and len(tail_wm) > 0:
                output_parts.append(tail_wm)

        final_wav = np.concatenate(output_parts)
        sf.write(output_path, final_wav, 16000)
        elapsed = time.time() - t_start
        rtf = elapsed / total_duration
        print(f"[WatermarkedVC] Finished. Output saved to: {output_path}")
        print(f"[WatermarkedVC] Audio: {total_duration:.2f}s, Processed in: {elapsed:.2f}s, RTF: {rtf:.3f}")

        # Verification step
        if verify_watermark:
            print("[WatermarkedVC] Verifying watermark extraction from output file...")
            detect_score, detected_bits = self.watermarker.detect(final_wav)
            gt_bits = self.secret_message.squeeze().tolist()
            matched = sum(int(a == b) for a, b in zip(detected_bits, gt_bits))
            acc = matched / len(gt_bits)
            print(f"  -> Watermark Detected Probability: {detect_score * 100:.2f}%")
            print(f"  -> Extracted Message: {detected_bits}")
            print(f"  -> Ground Truth Msg:  {gt_bits}")
            print(f"  -> Bit Accuracy:      {acc * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="MeanVC2 + AudioSeal Real-Time Stream")
    parser.add_argument("--mode", choices=["file", "test"], default="test",
                        help="'file' to convert audio file, 'test' to run self-test")
    parser.add_argument("--input", type=str, default="test_audio/test.wav", help="Input WAV path")
    parser.add_argument("--output", type=str, default="watermarked_output.wav", help="Output WAV path")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET_WAV, help="Target speaker WAV")
    parser.add_argument("--model", choices=["120ms", "40ms"], default="120ms", help="MeanVC2 model")
    parser.add_argument("--alpha", type=float, default=1.0, help="Watermark strength alpha")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device")

    args = parser.parse_args()
    def resolve_path(p: str | None) -> str | None:
        if p is None:
            return None
        if os.path.exists(p):
            return p
        alt = _MEANVC_ROOT / p
        if alt.exists():
            return str(alt)
        return p

    target_path = resolve_path(args.target)
    input_path = resolve_path(args.input)

    if args.mode == "file":
        runner = WatermarkedVCRunner(
            target_wav=target_path,
            device=args.device,
            model=args.model,
            alpha=args.alpha,
        )
        runner.process_file(input_path, args.output)
    else:
        print("Running AudioSeal standalone stream self-test...")
        from test_watermark_stream import run_stream_test
        run_stream_test()


if __name__ == "__main__":
    main()
