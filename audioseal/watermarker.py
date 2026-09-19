# Copyright (c) 2026 MeanVC2 AudioSeal Integration. All rights reserved.
"""
AudioSeal Real-Time Streaming Watermark Processor.

Designed for seamless integration with real-time speech generation and conversion
pipelines (e.g. MeanVC2, CosyVoice, VALL-E).
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Optional, Tuple, Union

# 1. Protection against g++ -std=c++20 compiler issues on legacy host GCC
os.environ["NO_TORCH_COMPILE"] = "1"

import numpy as np
import torch

try:
    import torch._dynamo
    torch._dynamo.config.disable = True
except Exception:
    pass

# 2. Safely import installed AudioSeal package without shadowing by local folder name
_local_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_local_dir)
_removed_paths = []
for p in list(sys.path):
    abs_p = os.path.abspath(p) if p else os.path.abspath(os.getcwd())
    if abs_p in (os.path.abspath(_local_dir), os.path.abspath(_parent_dir)):
        _removed_paths.append(p)
        sys.path.remove(p)

_local_mod = sys.modules.pop("audioseal", None)
try:
    import audioseal as _official_audioseal
    from audioseal.models import AudioSealWM, AudioSealDetector
    AudioSeal = _official_audioseal.AudioSeal
finally:
    for p in reversed(_removed_paths):
        sys.path.insert(0, p)
    if _local_mod is not None:
        sys.modules["audioseal"] = _local_mod


class AudioSealStreamWatermarker:
    """
    Real-time streaming audio watermark embedder and extractor.

    Maintains a continuous causal convolution state buffer across audio chunks,
    ensuring zero-lookahead latency, seamless chunk boundary transitions,
    and constant O(1) memory usage during continuous real-time streaming.
    """

    def __init__(
        self,
        model_card: str = "audioseal_wm_streaming",
        detector_card: str = "audioseal_detector_streaming",
        device: str | torch.device = "cpu",
        default_message: Optional[Union[torch.Tensor, list[int]]] = None,
        alpha: float = 1.0,
        sample_rate: int = 16000,
    ):
        """
        Initialize the streaming watermarker.

        Args:
            model_card: AudioSeal generator model card ('audioseal_wm_streaming').
            detector_card: AudioSeal detector model card ('audioseal_detector_streaming').
            device: Computing device ('cpu' or 'cuda').
            default_message: Default 16-bit binary message (shape: [1, 16] or 16 ints).
            alpha: Default watermark embedding strength (default 1.0).
            sample_rate: Audio sampling rate (AudioSeal models are 16kHz native).
        """
        self.device = torch.device(device)
        self.alpha = float(alpha)
        self.sample_rate = sample_rate

        # Prevent OpenMP thread-pool synchronization lock contention on high-core servers (e.g. 120 cores)
        if self.device.type == "cpu" and torch.get_num_threads() > 4:
            torch.set_num_threads(4)

        # Load streaming generator
        print(f"[AudioSeal] Loading streaming generator: {model_card} on {self.device}...")
        self.generator: AudioSealWM = AudioSeal.load_generator(model_card)
        self.generator.to(self.device)
        self.generator.eval()

        # Load streaming detector (for verification / inline detection)
        print(f"[AudioSeal] Loading streaming detector: {detector_card} on {self.device}...")
        self.detector: AudioSealDetector = AudioSeal.load_detector(detector_card)
        self.detector.to(self.device)
        self.detector.eval()

        # Secret message handling (16 bits)
        self.nbits = getattr(self.generator, "nbits", 16)
        if default_message is None:
            # Default to an identifiable pseudo-random or fixed 16-bit key
            # e.g., 0xA55A: [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
            self.default_message = torch.tensor(
                [[1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0]],
                dtype=torch.int32,
                device=self.device,
            )
        else:
            if isinstance(default_message, list):
                self.default_message = torch.tensor(
                    [default_message], dtype=torch.int32, device=self.device
                )
            else:
                self.default_message = default_message.to(
                    dtype=torch.int32, device=self.device
                )
                if self.default_message.dim() == 1:
                    self.default_message = self.default_message.unsqueeze(0)

        # Initialize persistent streaming state
        self._start_persistent_stream()

    def _start_persistent_stream(self):
        """Enable persistent streaming buffer in encoder."""
        if hasattr(self.generator.encoder, "streaming_forever"):
            self.generator.encoder.streaming_forever(batch_size=1)
        elif hasattr(self.generator.encoder, "_start_streaming"):
            self.generator.encoder._start_streaming(batch_size=1)

    def reset(self):
        """
        Reset internal streaming buffer.
        MUST be called when switching to a new utterance, new speaker,
        or starting a new conversation session to prevent boundary artifact contamination.
        """
        if hasattr(self.generator.encoder, "reset_streaming"):
            self.generator.encoder.reset_streaming()
        elif hasattr(self.generator.encoder, "_init_streaming_state"):
            self._start_persistent_stream()

    @contextmanager
    def streaming_session(self):
        """Context manager for an isolated streaming session (auto-reset on exit)."""
        self.reset()
        try:
            yield self
        finally:
            self.reset()

    def process_chunk(
        self,
        chunk: Union[np.ndarray, torch.Tensor],
        alpha: Optional[float] = None,
        message: Optional[Union[torch.Tensor, list[int]]] = None,
    ) -> np.ndarray:
        """
        Process and embed watermark into a single incoming audio chunk in real time.

        Args:
            chunk: Audio chunk waveform (16kHz float32).
                   Can be 1D numpy array [T], 2D [1, T], or PyTorch Tensor.
            alpha: Custom watermark strength for this chunk. Defaults to self.alpha.
            message: Custom secret message for this chunk. Defaults to self.default_message.

        Returns:
            Watermarked audio chunk as 1D numpy array (float32).
        """
        if chunk is None or (isinstance(chunk, np.ndarray) and chunk.size == 0):
            return np.zeros(0, dtype=np.float32)

        # 1. Format input to [batch=1, channels=1, time=T]
        is_numpy = isinstance(chunk, np.ndarray)
        if is_numpy:
            t_chunk = torch.from_numpy(chunk).float().to(self.device)
        else:
            t_chunk = chunk.float().to(self.device)

        if t_chunk.dim() == 1:
            t_chunk = t_chunk.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        elif t_chunk.dim() == 2:
            t_chunk = t_chunk.unsqueeze(1)               # [1, 1, T]

        # 2. Prepare message
        if message is None:
            msg_t = self.default_message
        elif isinstance(message, list):
            msg_t = torch.tensor([message], dtype=torch.int32, device=self.device)
        else:
            msg_t = message.to(dtype=torch.int32, device=self.device)
            if msg_t.dim() == 1:
                msg_t = msg_t.unsqueeze(0)

        eff_alpha = float(alpha if alpha is not None else self.alpha)

        # 3. Generate watermark
        with torch.no_grad():
            wm = self.generator.get_watermark(t_chunk, sample_rate=self.sample_rate, message=msg_t)
            watermarked = t_chunk + eff_alpha * wm

        # 4. Return as 1D float32 numpy array (matching MeanVC2 convention)
        out_np = watermarked.squeeze().detach().cpu().numpy().astype(np.float32)
        return out_np

    def detect(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        detection_threshold: float = 0.5,
        message_threshold: float = 0.5,
    ) -> Tuple[float, list[int]]:
        """
        Detect watermark and decode 16-bit message from an audio slice.

        Args:
            audio: Audio waveform slice (16kHz).
            detection_threshold: Threshold for watermark presence detection.
            message_threshold: Threshold for bit probability rounding (0 or 1).

        Returns:
            Tuple of:
              - detect_probability: float between 0.0 and 1.0.
              - decoded_message: list of 16 binary ints [0 or 1].
        """
        if isinstance(audio, np.ndarray):
            t_audio = torch.from_numpy(audio).float().to(self.device)
        else:
            t_audio = audio.float().to(self.device)

        if t_audio.dim() == 1:
            t_audio = t_audio.unsqueeze(0).unsqueeze(0)
        elif t_audio.dim() == 2:
            t_audio = t_audio.unsqueeze(0)

        with torch.no_grad():
            score, dec_msg = self.detector.detect_watermark(
                t_audio,
                sample_rate=self.sample_rate,
                detection_threshold=detection_threshold,
                message_threshold=message_threshold,
            )

        prob = float(score.item()) if score.numel() == 1 else float(score[0].item())
        bits = dec_msg.squeeze().cpu().tolist()
        if isinstance(bits, int):
            bits = [bits]
        return prob, bits
