#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# E4B + Serena Agent Bridge — Universal Startup Script
# Works from any project directory that has agent_config.yaml
# ═══════════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BRIDGE_SCRIPT="${SCRIPT_DIR}/agent_bridge.py"
CONFIG_FILE="${SCRIPT_DIR}/agent_config.yaml"

# Fallback to old bridge script name if configurable one missing
if [ ! -f "$BRIDGE_SCRIPT" ]; then
    BRIDGE_SCRIPT="${SCRIPT_DIR}/serena_e4b_bridge.py"
fi

echo "=== E4B + Serena Agent Bridge Startup ==="
echo "  Project: $SCRIPT_DIR"
echo ""

# ─── 1. Stop GUI (frees ~1.8GB RAM on Jetson) ───────────────────
echo "[1/4] Stopping GUI..."
sudo systemctl stop gdm.service 2>/dev/null && echo "  GDM stopped" || echo "  GDM not running"
free -h | head -2

# ─── 2. Verify / Start E4B llama-server ──────────────────────────
echo ""
echo "[2/4] Checking E4B on port 8091..."
if curl -s --max-time 5 http://127.0.0.1:8091/health | grep -q "ok"; then
    MODEL=$(curl -s http://127.0.0.1:8091/v1/models 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin)['models'][0]['model'])" 2>/dev/null || echo "unknown")
    echo "  E4B running: $MODEL"
else
    echo "  Starting E4B llama-server..."
    GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 setsid "$HOME/llama.cpp/build/bin/llama-server" \
        -m "$HOME/models/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf" \
        --alias gemma-4-e4b-qat --host 127.0.0.1 --port 8091 \
        -ngl 99 -c 32768 -ctk q4_0 -ctv q4_0 \
        -b 64 -ub 64 -fa on --jinja --fit off -np 1 \
        > /tmp/e4b_server.log 2>&1 &
    echo "  Waiting for server..."
    for i in $(seq 1 60); do
        curl -s --max-time 5 http://127.0.0.1:8091/health | grep -q "ok" && { echo "  Ready after ${i}s"; break; }
        sleep 1
    done
fi

# ─── 3. Verify Python deps ───────────────────────────────────────
echo ""
echo "[3/4] Checking dependencies..."
python3 -c "import mcp; print('  mcp OK')" 2>/dev/null || { echo "  mcp MISSING — pip install mcp"; exit 1; }
python3 -c "import httpx; print('  httpx OK')" 2>/dev/null || { echo "  httpx MISSING — pip install httpx"; exit 1; }
python3 -c "import yaml; print('  pyyaml OK')" 2>/dev/null || { echo "  pyyaml MISSING — pip install pyyaml"; exit 1; }
which serena >/dev/null 2>&1 && echo "  serena OK" || { echo "  serena MISSING — pip install serena-agent"; exit 1; }

# ─── 4. Launch Agent Bridge ──────────────────────────────────────
echo ""
echo "[4/4] Launching agent bridge..."
echo "  Config: $CONFIG_FILE"
echo ""

cd "$SCRIPT_DIR"
setsid python3 "$BRIDGE_SCRIPT" > /tmp/serena_agent_run.log 2>&1 &
BRIDGE_PID=$!
echo "  Bridge PID: $BRIDGE_PID"
echo "  Log: $SCRIPT_DIR/agent_log.json"
echo "  Stdout: /tmp/serena_agent_run.log"
echo ""
echo "=== Agent running in background ==="
echo "  Monitor: python3 -c \"import json;[print(f'[{d.get(\\\"type\\\")}] turn={d.get(\\\"turn\\\",\\\"?\\\")} {d.get(\\\"summary\\\",\\\"\\\")[:200]}') for l in open('$SCRIPT_DIR/agent_log.json') for d in [json.loads(l)]]\""
echo "  Kill: kill $BRIDGE_PID"