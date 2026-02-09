#!/bin/bash
# Consistent benchmark: 50 concurrent users, 512 in / 512 out, 200 prompts
# Usage: ./run_bench.sh <label>

LABEL=${1:-"baseline"}
RESULT_DIR="/home/amit_h2loop_ai/infernce_challenge/results"
mkdir -p "$RESULT_DIR"

echo "=== Waiting for server to be ready ==="
for i in $(seq 1 120); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 2
done

# Verify server is up
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "ERROR: Server not ready after 240s"
    exit 1
fi

# Get model name from server
MODEL=$(curl -s http://localhost:8000/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "Model: $MODEL"
echo "Label: $LABEL"

echo "=== Running benchmark: $LABEL ==="
vllm bench serve \
  --backend openai \
  --base-url http://localhost:8000/v1 \
  --endpoint /completions \
  --model "$MODEL" \
  --tokenizer Qwen/Qwen2.5-0.5B \
  --max-concurrency 50 \
  --num-prompts 200 \
  --ignore-eos \
  --random-input-len 512 \
  --random-output-len 512 \
  --save-result \
  --result-dir "$RESULT_DIR" \
  --result-filename "${LABEL}.json" \
  --label "$LABEL" \
  2>&1

echo "=== Benchmark $LABEL complete ==="
