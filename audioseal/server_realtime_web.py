#!/usr/bin/env python3
"""
MeanVC2 + AudioSeal Real-Time Web Streaming Server
Combines:
1. aiohttp HTTP Server (serves Web UI static assets)
2. aiohttp WebSocket Server (/ws/stream) for real-time 16kHz PCM audio streaming
3. MeanVC2 VCRunner (streaming chunk voice conversion)
4. AudioSealStreamWatermarker (strictly causal 16-bit watermark embed & extract)
"""

import os
import sys
import time
import json
import asyncio
import argparse

# Lazy import holder
VCRunner = None
AudioSealStreamWatermarker = None
extract_embedding = None
torch = None
np = None

# Set project roots
_AUDIOROOT = os.path.dirname(os.path.abspath(__file__))
_MEANVC_ROOT = os.path.dirname(_AUDIOROOT)
if _AUDIOROOT not in sys.path:
    sys.path.insert(0, _AUDIOROOT)
if _MEANVC_ROOT not in sys.path:
    sys.path.insert(0, _MEANVC_ROOT)
if os.path.join(_MEANVC_ROOT, "runtime") not in sys.path:
    sys.path.insert(0, os.path.join(_MEANVC_ROOT, "runtime"))

STATIC_DIR = os.path.join(_AUDIOROOT, "web")
DEFAULT_TARGET_REL = "test_audio/common_voice_en_10119832.wav"
DEFAULT_TARGET_PATH = os.path.join(_MEANVC_ROOT, DEFAULT_TARGET_REL)

# Predefined 16-bit payload secret key
DEFAULT_KEY = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1]

# Speaker Presets for Voice Conversion
SPEAKER_PRESETS = {
    "default_male": {
        "label": "CommonVoice Male (Default)",
        "path": DEFAULT_TARGET_PATH,
        "aliases": ["default_male", "common_voice_en_10119832.wav", "default", "male_default"]
    },
    "libritts_female_150": {
        "label": "LibriTTS Female (Spk 150 · Warm)",
        "path": "/home/pj25001109/ku60000344/dataset/libriTTS/LibriTTS/train-clean-100/150/126107/150_126107_000026_000000.wav",
        "aliases": ["libritts_female_150", "libritts_female", "female", "female_150"]
    },
    "libritts_female_19": {
        "label": "LibriTTS Female (Spk 19 · Clear)",
        "path": "/home/pj25001109/ku60000344/dataset/libriTTS/LibriTTS/train-clean-100/19/227/19_227_000005_000000.wav",
        "aliases": ["libritts_female_19", "female_19"]
    },
    "libritts_male_1081": {
        "label": "LibriTTS Male (Spk 1081 · Deep)",
        "path": "/home/pj25001109/ku60000344/dataset/libriTTS/LibriTTS/train-clean-100/1081/128618/1081_128618_000024_000003.wav",
        "aliases": ["libritts_male_1081", "libritts_male", "male_1081"]
    }
}


def _lazy_init_modules():
    global VCRunner, AudioSealStreamWatermarker, torch, np, extract_embedding
    if torch is None:
        print("[Init] Loading PyTorch and NumPy...")
        import torch as _torch
        import numpy as _np
        torch = _torch
        np = _np

    if VCRunner is None:
        print("[Init] Loading MeanVC2 runtime and speaker module...")
        from runtime.run_rt import VCRunner as _VCRunner
        from src.speaker import extract_embedding as _extract_embedding
        VCRunner = _VCRunner
        extract_embedding = _extract_embedding

    if AudioSealStreamWatermarker is None:
        print("[Init] Loading AudioSeal stream watermarker...")
        from watermarker import AudioSealStreamWatermarker as _AudioSealStreamWatermarker
        AudioSealStreamWatermarker = _AudioSealStreamWatermarker


class RealtimeServer:
    def __init__(self, device: str = "cpu", model: str = "40ms", target_spk: str = DEFAULT_TARGET_PATH):
        _lazy_init_modules()

        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model_type = model
        self.target_spk = target_spk

        print(f"[Init 1/3] Loading MeanVC2 ({self.model_type}) on {self.device}...")
        self.vc = VCRunner(target_wav=self.target_spk, device=self.device, model=self.model_type)

        print(f"[Init 2/3] Precomputing target speaker embeddings & GTM KVs...")
        self.spk_caches = {}
        for spk_key, info in SPEAKER_PRESETS.items():
            wav_path = info["path"]
            if os.path.exists(wav_path):
                emb = extract_embedding(self.vc.spk_model, wav_path, device=self.device)
                with torch.no_grad():
                    gtm_kv = self.vc.vc.gtm(emb)
                self.spk_caches[spk_key] = {
                    "emb": emb,
                    "gtm_kv": gtm_kv,
                    "label": info["label"],
                    "aliases": info["aliases"]
                }
                print(f"  -> Cached [{spk_key}]: {info['label']}")
            else:
                print(f"  -> WARNING: Path not found for [{spk_key}]: {wav_path}")

        print(f"[Init 3/3] Loading AudioSeal streaming watermarker on {self.device}...")
        self.wm = AudioSealStreamWatermarker(
            device=self.device,
            default_message=DEFAULT_KEY,
            alpha=0.50,
            sample_rate=16000,
        )

        self.wm_strength = 0.50
        self.is_detecting = False
        if self.device == "cpu":
            # Enable multi-threading for CPU to speed up DiT and Vocos
            torch.set_num_threads(min(8, os.cpu_count() or 4))
            print(f"[Init] Set CPU PyTorch threads to: {torch.get_num_threads()}")

        print(f"[Ready] Pipeline active on {self.device}. Ready for live streaming.")

    def set_speaker(self, requested_key: str):
        requested_key_clean = str(requested_key).strip().lower()
        matched_entry = None
        matched_key = None

        for key, entry in self.spk_caches.items():
            if key.lower() == requested_key_clean or any(a.lower() == requested_key_clean for a in entry["aliases"]):
                matched_entry = entry
                matched_key = key
                break

        if matched_entry is None:
            print(f"[Speaker] Unknown speaker '{requested_key}', available: {list(self.spk_caches.keys())}")
            return False, requested_key

        self.vc.vc_spk_emb = matched_entry["emb"]
        self.vc.vc_gtm_kv = matched_entry["gtm_kv"]
        self.vc._init_cache()
        print(f"[Speaker] Switched active speaker timbre to: {matched_entry['label']} ({matched_key})")
        return True, matched_entry["label"]

    async def _detect_and_report(self, ws, chunk_id, wm_out_np, target_bits_list, proc_ms, chunk_ms, rtf):
        if getattr(self, "is_detecting", False):
            return
        self.is_detecting = True
        try:
            loop = asyncio.get_running_loop()
            prob, extracted_bits = await loop.run_in_executor(None, self.wm.detect, wm_out_np)
            matches = sum(t == e for t, e in zip(target_bits_list, extracted_bits))
            bit_acc = matches / 16.0
            meta_payload = {
                "type": "stats",
                "chunk_id": chunk_id,
                "proc_ms": round(proc_ms, 2),
                "chunk_ms": round(chunk_ms, 1),
                "rtf": round(rtf, 3),
                "wm_prob": round(prob, 4),
                "bit_acc": round(bit_acc, 4),
                "target_bits": target_bits_list,
                "extracted_bits": extracted_bits
            }
            if not ws.closed:
                await ws.send_str(json.dumps(meta_payload))
        except Exception:
            pass
        finally:
            self.is_detecting = False

    async def index_handler(self, request):
        from aiohttp import web
        index_path = os.path.join(STATIC_DIR, "index.html")
        return web.FileResponse(index_path)

    async def websocket_handler(self, request):
        from aiohttp import web, WSMsgType
        ws = web.WebSocketResponse(max_msg_size=1024 * 1024)
        await ws.prepare(request)
        peer = request.remote
        print(f"\n[WS] Client connected from {peer}")

        # Reset VC and AudioSeal state for new connection
        self.vc._init_cache()
        self.wm.reset()
        chunk_id = 0
        wm_alpha = self.wm_strength
        target_bits_list = DEFAULT_KEY
        silent_chunks = 0

        # Leaky audio queue: capacity of 2 strictly prevents latency queue accumulation
        audio_queue = asyncio.Queue(maxsize=2)

        async def audio_worker():
            nonlocal chunk_id, wm_alpha, target_bits_list, silent_chunks
            loop = asyncio.get_running_loop()
            while True:
                raw_bytes = await audio_queue.get()
                try:
                    samples = np.frombuffer(raw_bytes, dtype=np.float32)
                    if len(samples) == 0:
                        continue

                    # VAD: When user pauses, refresh DiT attention cache to keep sequence length short
                    max_amp = float(np.max(np.abs(samples)))
                    if max_amp < 0.008:
                        silent_chunks += 1
                        if silent_chunks >= 3 and getattr(self.vc, "vc_offset", 0) > 40:
                            self.vc.vc_kv_cache = None
                            self.vc.vc_offset = 0
                    else:
                        silent_chunks = 0

                    t0 = time.perf_counter()

                    # 1. MeanVC2 Real-Time Voice Conversion (run in worker thread)
                    out_vc = await loop.run_in_executor(None, self.vc.process_chunk, samples)

                    if out_vc is not None and len(out_vc) > 0:
                        chunk_id += 1

                        # 2. AudioSeal Real-Time Streaming Embedding (run in worker thread)
                        wm_out_np = await loop.run_in_executor(
                            None, self.wm.process_chunk, out_vc, wm_alpha, target_bits_list
                        )

                        t_audio = time.perf_counter()
                        proc_ms = (t_audio - t0) * 1000.0
                        chunk_ms = (len(samples) / 16000.0) * 1000.0
                        rtf = (t_audio - t0) / (len(samples) / 16000.0)

                        # Send watermarked audio back IMMEDIATELY (zero lag for ear-return)
                        if not ws.closed:
                            await ws.send_bytes(wm_out_np.astype(np.float32).tobytes())

                        # 3. AudioSeal Streaming Detection in background task
                        if not getattr(self, "is_detecting", False):
                            asyncio.create_task(
                                self._detect_and_report(
                                    ws, chunk_id, wm_out_np, target_bits_list, proc_ms, chunk_ms, rtf
                                )
                            )
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"[Worker] Audio process error: {e}")
                finally:
                    audio_queue.task_done()

        worker_task = asyncio.create_task(audio_worker())

        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    # Drop older chunk if processing cannot keep up, keeping fresh real-time audio
                    if audio_queue.full():
                        try:
                            audio_queue.get_nowait()
                            audio_queue.task_done()
                        except (asyncio.QueueEmpty, ValueError):
                            pass
                    await audio_queue.put(msg.data)

                elif msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        if data.get("type") == "config":
                            if "wm_strength" in data:
                                wm_alpha = float(data["wm_strength"])
                                print(f"[WS] Updated watermark strength: {wm_alpha}")
                            if "target_spk" in data:
                                spk_key = str(data["target_spk"])
                                print(f"[WS] Requested target speaker: {spk_key}")
                                ok, spk_label = self.set_speaker(spk_key)
                                if ok and not ws.closed:
                                    await ws.send_str(json.dumps({
                                        "type": "log",
                                        "message": f"Active target speaker: {spk_label}",
                                        "log_type": "system"
                                    }))
                    except Exception as e:
                        print(f"[WS] Config parse error: {e}")

                elif msg.type == WSMsgType.ERROR:
                    print(f"[WS] WebSocket error: {ws.exception()}")

        except asyncio.CancelledError:
            pass
        finally:
            worker_task.cancel()
            print(f"[WS] Client disconnected from {peer}")

        return ws


def create_app(device: str = "cpu", model: str = "40ms", target_spk: str = DEFAULT_TARGET_PATH):
    from aiohttp import web
    server = RealtimeServer(device=device, model=model, target_spk=target_spk)
    app = web.Application()

    app.router.add_get("/", server.index_handler)
    app.router.add_get("/ws/stream", server.websocket_handler)
    app.router.add_static("/", path=STATIC_DIR, show_index=False)

    return app


def main():
    parser = argparse.ArgumentParser(description="MeanVC2 + AudioSeal Real-Time Web Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8998, help="Port (default: 8998)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--model", type=str, default="40ms", choices=["40ms", "120ms"])
    parser.add_argument("--target-spk", type=str, default=DEFAULT_TARGET_PATH)
    args = parser.parse_args()

    print("=" * 60)
    print("MeanVC2 + AudioSeal Real-Time Web Studio")
    print(f"  URL:    http://{args.host}:{args.port}")
    print(f"  Device: {args.device}")
    print(f"  Model:  {args.model}")
    print(f"  Target: {args.target_spk}")
    print("=" * 60)
    print(f"[Instructions for Local Laptop Browser & Microphone]:")
    print(f"  1. Run this SSH command on your local computer:")
    print(f"     ssh -L {args.port}:localhost:{args.port} <user>@<server>")
    print(f"  2. Open this link in Chrome or Edge:")
    print(f"     http://localhost:{args.port}")
    print("=" * 60)

    from aiohttp import web
    app = create_app(device=args.device, model=args.model, target_spk=args.target_spk)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
