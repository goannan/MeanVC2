/**
 * MeanVC2 + AudioSeal Real-Time Streaming Web Client
 * Handles:
 * - 16kHz microphone capture and 40ms chunk slicing
 * - WebSocket streaming of raw PCM Float32
 * - Jitter-buffered audio playback via Web Audio API
 * - Real-time 16-bit watermark display and metrics visualization
 * - Dual-layer oscilloscope visualizer
 */

// State
let ws = null;
let audioCtx = null;
let micStream = null;
let micSourceNode = null;
let processorNode = null;
let analyserInput = null;
let analyserOutput = null;
let gainNode = null;

let isRecording = false;
let nextPlayTime = 0;
let chunkCounter = 0;
let pcmBuffer = []; // Accumulator for 40ms chunks (640 samples @ 16kHz)

let currentChunkSamples = 2560; // Default 160ms (CPU friendly)
const TARGET_SR = 16000;

// DOM Elements
const wsStatusBadge = document.getElementById("ws-status-badge");
const wsStatusText = document.getElementById("ws-status-text");
const btnMic = document.getElementById("btn-mic-toggle");
const micBtnLabel = document.getElementById("mic-btn-label");
const selectChunkSize = document.getElementById("select-chunk-size");
const selectTargetSpk = document.getElementById("select-target-spk");
const sliderWmStrength = document.getElementById("slider-wm-strength");
const wmStrengthVal = document.getElementById("wm-strength-val");
const sliderOutVolume = document.getElementById("slider-out-volume");
const outVolumeVal = document.getElementById("out-volume-val");

if (selectChunkSize) {
    currentChunkSamples = parseInt(selectChunkSize.value);
    selectChunkSize.addEventListener("change", (e) => {
        currentChunkSamples = parseInt(e.target.value);
        const ms = Math.round(currentChunkSamples / 16);
        addLog(`Switched streaming buffer step: ${ms}ms`, "system");
        const valChunkMs = document.getElementById("val-chunk-ms");
        if (valChunkMs) valChunkMs.textContent = ms;
    });
}

const targetBitsContainer = document.getElementById("target-bits-container");
const extractedBitsContainer = document.getElementById("extracted-bits-container");
const valDetectProb = document.getElementById("val-detect-prob");
const barDetectProb = document.getElementById("bar-detect-prob");
const valBitAcc = document.getElementById("val-bit-acc");
const barBitAcc = document.getElementById("bar-bit-acc");
const valProcLatency = document.getElementById("val-proc-latency");
const valRtf = document.getElementById("val-rtf");
const valChunkCount = document.getElementById("val-chunk-count");
const statusRtfPill = document.getElementById("status-rtf-pill");
const logTerminal = document.getElementById("log-terminal");
const btnClearLog = document.getElementById("btn-clear-log");
const canvasVisualizer = document.getElementById("canvas-visualizer");
const canvasCtx = canvasVisualizer.getContext("2d");

// ---------------------------------------------------------------------------
// Logging helper
// ---------------------------------------------------------------------------
function addLog(msg, type = "normal") {
    const entry = document.createElement("div");
    entry.className = `log-entry ${type}`;
    const timeStr = new Date().toLocaleTimeString();
    entry.textContent = `[${timeStr}] ${msg}`;
    logTerminal.appendChild(entry);
    logTerminal.scrollTop = logTerminal.scrollHeight;
}

btnClearLog.addEventListener("click", () => {
    logTerminal.innerHTML = "";
});

// ---------------------------------------------------------------------------
// UI Initialization: 16-Bit Grid
// ---------------------------------------------------------------------------
function initBitContainers() {
    targetBitsContainer.innerHTML = "";
    extractedBitsContainer.innerHTML = "";
    for (let i = 0; i < 16; i++) {
        const boxT = document.createElement("div");
        boxT.id = `bit-t-${i}`;
        boxT.className = "bit-box val-0";
        boxT.textContent = "-";
        targetBitsContainer.appendChild(boxT);

        const boxE = document.createElement("div");
        boxE.id = `bit-e-${i}`;
        boxE.className = "bit-box val-0";
        boxE.textContent = "-";
        extractedBitsContainer.appendChild(boxE);
    }
}
initBitContainers();

// Sliders listener
sliderWmStrength.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value).toFixed(2);
    wmStrengthVal.textContent = val;
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "config", wm_strength: parseFloat(val) }));
    }
});

sliderOutVolume.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    outVolumeVal.textContent = `${Math.round(val * 100)}%`;
    if (gainNode) {
        gainNode.gain.value = val;
    }
});

selectTargetSpk.addEventListener("change", (e) => {
    const spk = e.target.value;
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "config", target_spk: spk }));
        addLog(`Switched target speaker timbre: ${spk}`, "system");
    }
});

// ---------------------------------------------------------------------------
// WebSocket Connection
// ---------------------------------------------------------------------------
function initWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/stream`;

    ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
        wsStatusBadge.className = "status-badge connected";
        wsStatusText.textContent = "Connected";
        addLog("WebSocket link established. Server ready.", "system");
        
        // Sync initial config
        ws.send(JSON.stringify({
            type: "config",
            wm_strength: parseFloat(sliderWmStrength.value),
            target_spk: selectTargetSpk.value
        }));
    };

    ws.onclose = () => {
        wsStatusBadge.className = "status-badge disconnected";
        wsStatusText.textContent = "Disconnected";
        addLog("WebSocket disconnected. Attempting reconnection...", "error");
        if (isRecording) stopRecording();
        setTimeout(initWebSocket, 2000);
    };

    ws.onerror = (err) => {
        console.error("WS error:", err);
    };

    ws.onmessage = (event) => {
        if (typeof event.data === "string") {
            // JSON metadata with watermark detection stats
            try {
                const meta = JSON.parse(event.data);
                handleServerMetadata(meta);
            } catch (err) {
                console.error("Metadata parse error:", err);
            }
        } else if (event.data instanceof ArrayBuffer) {
            // Binary audio chunk (Float32 PCM)
            playAudioChunk(new Float32Array(event.data));
        }
    };
}

// ---------------------------------------------------------------------------
// Metadata Handler (Updates 16-Bit Matrix & Metrics)
// ---------------------------------------------------------------------------
function handleServerMetadata(meta) {
    if (meta.type === "log") {
        addLog(meta.message, meta.log_type || "system");
        return;
    }
    if (meta.type !== "stats") return;

    chunkCounter = meta.chunk_id;
    valChunkCount.textContent = chunkCounter;

    // Detect probability
    const probPct = Math.min(100, Math.max(0, (meta.wm_prob * 100))).toFixed(1);
    valDetectProb.textContent = probPct;
    barDetectProb.style.width = `${probPct}%`;

    // Bit accuracy
    const accPct = (meta.bit_acc * 100).toFixed(1);
    valBitAcc.textContent = accPct;
    barBitAcc.style.width = `${accPct}%`;

    // Latency & RTF
    valProcLatency.textContent = meta.proc_ms.toFixed(1);
    valRtf.textContent = meta.rtf.toFixed(2);

    if (meta.rtf <= 0.35) {
        statusRtfPill.className = "badge-pill ok";
        statusRtfPill.textContent = "Ultra Fast";
    } else if (meta.rtf <= 1.0) {
        statusRtfPill.className = "badge-pill ok";
        statusRtfPill.textContent = "RT Steady";
    } else {
        statusRtfPill.className = "badge-pill warn";
        statusRtfPill.textContent = "Buffer Lag";
    }

    // Update 16-bit boxes
    if (meta.target_bits && meta.extracted_bits) {
        for (let i = 0; i < 16; i++) {
            const tBit = meta.target_bits[i];
            const eBit = meta.extracted_bits[i];

            const boxT = document.getElementById(`bit-t-${i}`);
            const boxE = document.getElementById(`bit-e-${i}`);

            if (boxT) {
                boxT.textContent = tBit;
                boxT.className = `bit-box val-${tBit}`;
            }

            if (boxE) {
                boxE.textContent = eBit;
                const isMatch = (tBit === eBit);
                boxE.className = `bit-box val-${eBit} ${isMatch ? 'match' : 'mismatch'}`;
            }
        }
    }

    // Periodic log
    if (chunkCounter % 50 === 0) {
        addLog(`[Chunk ${chunkCounter}] Confidence: ${probPct}% | Bit Acc: ${accPct}% | RTF: ${meta.rtf.toFixed(2)}`, "watermark");
    }
}

// ---------------------------------------------------------------------------
// Audio Playback Pipeline (Continuous Jitter-Free Web Audio Buffer)
// ---------------------------------------------------------------------------
function playAudioChunk(float32Data) {
    if (!audioCtx || float32Data.length === 0) return;

    // Send through output analyser for visualizer
    const buffer = audioCtx.createBuffer(1, float32Data.length, TARGET_SR);
    buffer.copyToChannel(float32Data, 0, 0);

    const source = audioCtx.createBufferSource();
    source.buffer = buffer;

    // Connect source -> gain -> analyser -> speakers
    source.connect(gainNode);

    const now = audioCtx.currentTime;
    // Anti-Lag Catch-Up: If scheduled playback drifts more than 280ms ahead of now,
    // snap back to real-time (now + 20ms) so latency NEVER accumulates!
    if (nextPlayTime < now || (nextPlayTime - now) > 0.28) {
        nextPlayTime = now + 0.02; // 20ms lead-in jitter buffer
    }

    source.start(nextPlayTime);
    nextPlayTime += buffer.duration;
}

// ---------------------------------------------------------------------------
// Microphone Capture & 40ms Slicing
// ---------------------------------------------------------------------------
async function startRecording() {
    try {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: TARGET_SR
            });
        }
        if (audioCtx.state === "suspended") {
            await audioCtx.resume();
        }

        nextPlayTime = audioCtx.currentTime;

        // Setup output nodes
        gainNode = audioCtx.createGain();
        gainNode.gain.value = parseFloat(sliderOutVolume.value);

        analyserOutput = audioCtx.createAnalyser();
        analyserOutput.fftSize = 512;
        gainNode.connect(analyserOutput);
        analyserOutput.connect(audioCtx.destination);

        // Request microphone
        micStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: TARGET_SR,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });

        micSourceNode = audioCtx.createMediaStreamSource(micStream);
        analyserInput = audioCtx.createAnalyser();
        analyserInput.fftSize = 512;
        micSourceNode.connect(analyserInput);

        // Use ScriptProcessor for straightforward PCM chunking
        const bufferSize = 1024;
        processorNode = audioCtx.createScriptProcessor(bufferSize, 1, 1);

        pcmBuffer = [];

        processorNode.onaudioprocess = (e) => {
            if (!isRecording) return;
            const inputData = e.inputBuffer.getChannelData(0);

            // Accumulate into pcmBuffer
            for (let i = 0; i < inputData.length; i++) {
                pcmBuffer.push(inputData[i]);
            }

            // Flush out full chunks (e.g. 160ms = 2560 samples, or 40ms = 640 samples)
            while (pcmBuffer.length >= currentChunkSamples) {
                const chunk = new Float32Array(pcmBuffer.slice(0, currentChunkSamples));
                pcmBuffer = pcmBuffer.slice(currentChunkSamples);

                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(chunk.buffer);
                }
            }
        };

        micSourceNode.connect(processorNode);
        // Connect processor to destination so Chrome keeps firing onaudioprocess
        const dummyGain = audioCtx.createGain();
        dummyGain.gain.value = 0.0;
        processorNode.connect(dummyGain);
        dummyGain.connect(audioCtx.destination);

        isRecording = true;
        btnMic.className = "btn-primary recording";
        micBtnLabel.textContent = "Stop Microphone";
        wsStatusBadge.className = "status-badge streaming";
        wsStatusText.textContent = "Streaming Live";
        addLog("Microphone active. Streaming 40ms chunks to GPU pipeline...", "system");

    } catch (err) {
        console.error("Microphone access error:", err);
        addLog(`Microphone access failed: ${err.message}`, "error");
        stopRecording();
    }
}

function stopRecording() {
    isRecording = false;
    btnMic.className = "btn-primary";
    micBtnLabel.textContent = "Start Microphone";

    if (micStream) {
        micStream.getTracks().forEach(track => track.stop());
        micStream = null;
    }
    if (processorNode) {
        processorNode.disconnect();
        processorNode = null;
    }
    if (micSourceNode) {
        micSourceNode.disconnect();
        micSourceNode = null;
    }

    wsStatusBadge.className = ws && ws.readyState === WebSocket.OPEN ? "status-badge connected" : "status-badge disconnected";
    wsStatusText.textContent = ws && ws.readyState === WebSocket.OPEN ? "Connected" : "Disconnected";
    addLog("Microphone stream stopped.", "system");
}

btnMic.addEventListener("click", () => {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
});

// ---------------------------------------------------------------------------
// Canvas Dual Oscilloscope Visualizer
// ---------------------------------------------------------------------------
function drawVisualizer() {
    requestAnimationFrame(drawVisualizer);

    const width = canvasVisualizer.width;
    const height = canvasVisualizer.height;

    canvasCtx.clearRect(0, 0, width, height);

    // Background grid
    canvasCtx.strokeStyle = "rgba(255, 255, 255, 0.03)";
    canvasCtx.lineWidth = 1;
    for (let y = 0; y < height; y += 30) {
        canvasCtx.beginPath();
        canvasCtx.moveTo(0, y);
        canvasCtx.lineTo(width, y);
        canvasCtx.stroke();
    }
    for (let x = 0; x < width; x += 60) {
        canvasCtx.beginPath();
        canvasCtx.moveTo(x, 0);
        canvasCtx.lineTo(x, height);
        canvasCtx.stroke();
    }

    // 1. Draw Output (Watermarked VC) Waveform (Violet/Purple)
    if (analyserOutput && isRecording) {
        const outData = new Uint8Array(analyserOutput.frequencyBinCount);
        analyserOutput.getByteTimeDomainData(outData);

        canvasCtx.lineWidth = 2.5;
        canvasCtx.strokeStyle = "#c084fc";
        canvasCtx.shadowColor = "rgba(192, 132, 252, 0.6)";
        canvasCtx.shadowBlur = 8;
        canvasCtx.beginPath();

        const sliceWidth = width / outData.length;
        let x = 0;

        for (let i = 0; i < outData.length; i++) {
            const v = outData[i] / 128.0;
            const y = (v * height) / 2;

            if (i === 0) canvasCtx.moveTo(x, y);
            else canvasCtx.lineTo(x, y);

            x += sliceWidth;
        }
        canvasCtx.stroke();
        canvasCtx.shadowBlur = 0;
    }

    // 2. Draw Mic Input Waveform (Neon Cyan)
    if (analyserInput && isRecording) {
        const inData = new Uint8Array(analyserInput.frequencyBinCount);
        analyserInput.getByteTimeDomainData(inData);

        canvasCtx.lineWidth = 2;
        canvasCtx.strokeStyle = "#00f2fe";
        canvasCtx.shadowColor = "rgba(0, 242, 254, 0.6)";
        canvasCtx.shadowBlur = 8;
        canvasCtx.beginPath();

        const sliceWidth = width / inData.length;
        let x = 0;

        for (let i = 0; i < inData.length; i++) {
            const v = inData[i] / 128.0;
            const y = (v * height) / 2;

            if (i === 0) canvasCtx.moveTo(x, y);
            else canvasCtx.lineTo(x, y);

            x += sliceWidth;
        }
        canvasCtx.stroke();
        canvasCtx.shadowBlur = 0;
    } else if (!isRecording) {
        // Flat center line when idle
        canvasCtx.strokeStyle = "rgba(255, 255, 255, 0.15)";
        canvasCtx.lineWidth = 1.5;
        canvasCtx.beginPath();
        canvasCtx.moveTo(0, height / 2);
        canvasCtx.lineTo(width, height / 2);
        canvasCtx.stroke();
    }
}

// Start visualizer loop and websocket
drawVisualizer();
initWebSocket();
