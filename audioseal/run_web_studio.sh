#!/bin/bash
# ==============================================================================
# Launch Script: MeanVC2 + AudioSeal Real-Time Web Streaming Studio
# ==============================================================================

PORT=${1:-8998}
DEVICE=${2:-cuda}
MODEL=${3:-40ms}

PYTHON="/home/pj25001109/ku60000344/miniconda3/envs/meanvc/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_USER=$(whoami)

# Gracefully clean up any previous server instances to prevent [Errno 98] port conflicts
OLD_PIDS=$(pgrep -f "server_realtime_web.py" || true)
if [ -n "$OLD_PIDS" ]; then
    echo "[Notice] Cleaning up previous server instance(s) (PID: $OLD_PIDS)..."
    kill -9 $OLD_PIDS 2>/dev/null || true
    sleep 1
fi

echo "======================================================================"
echo " Starting MeanVC2 + AudioSeal Real-Time Web Studio"
echo "======================================================================"
echo " Port:   $PORT"
echo " Device: $DEVICE"
echo " Model:  $MODEL"
echo " Web UI: http://127.0.0.1:$PORT"
echo "----------------------------------------------------------------------"
echo " [Step 1] Forward port over SSH on your local laptop terminal:"
echo "          ssh -L $PORT:localhost:$PORT $CURRENT_USER@<your_server_ip>"
echo " [Step 2] Open Chrome or Edge on your laptop:"
echo "          http://localhost:$PORT"
echo "======================================================================"

$PYTHON "$SCRIPT_DIR/server_realtime_web.py" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --device "$DEVICE" \
    --model "$MODEL"
