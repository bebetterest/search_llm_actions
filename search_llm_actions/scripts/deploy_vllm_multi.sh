MODEL_PATH=./QwQ-32B-Preview
SERVER_MODEL_NAME=local_model
SERVER_PORT_START=19270
DTYPE=bfloat16
SWAP_SPACE=64

GPU_MEMORY_UTILIZATION=0.95

# deploy 1 engine per GPU
GPU_LIST=(
    0 1 2 3 4 5 6 7
)

# deploy 1 engine per 2 GPUs
# GPU_LIST=(
#     0,1 2,3 4,5 6,7
#     0,1 2,3 4,5 6,7
# )

# deploy 2 engines per GPU
# GPU_LIST=(
#     0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7
# )


if command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=gpu_name --format=csv,noheader | grep -q "." ; then
    DEVICE_TYPE="nvidia"
    echo "DEVICE_TYPE: ${DEVICE_TYPE}"
elif command -v rocm-smi > /dev/null && rocm-smi --showproductname | grep -q "." ; then
    DEVICE_TYPE="amd"
    echo "DEVICE_TYPE: ${DEVICE_TYPE}"
else
    echo "No NVIDIA or AMD GPU found"
    exit 1
fi


SERVER_PORTS=()
for i in ${!GPU_LIST[@]}; do
    VISIBLE_DEVICES=${GPU_LIST[$i]}
    if [ $DEVICE_TYPE = "nvidia" ]; then
        export CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}
        echo "CUDA_VISIBLE_DEVICES: ${VISIBLE_DEVICES}"
    elif [ $DEVICE_TYPE = "amd" ]; then
        export HIP_VISIBLE_DEVICES=${VISIBLE_DEVICES}
        echo "HIP_VISIBLE_DEVICES: ${VISIBLE_DEVICES}"
    fi

    SERVER_PORT=$((SERVER_PORT_START + i))
    SERVER_PORTS+=($SERVER_PORT)
    TP_SIZE=$(echo $VISIBLE_DEVICES | tr ',' '\n' | wc -l)
    python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL_PATH \
        --served-model-name $SERVER_MODEL_NAME \
        --port $SERVER_PORT \
        --dtype $DTYPE \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --tensor-parallel-size $TP_SIZE \
        --swap-space $SWAP_SPACE \
        --enable-chunked-prefill \
        > ./vllm_${VISIBLE_DEVICES}.log 2>&1 &
    echo "vllm_${VISIBLE_DEVICES}.log"
done

echo "server_ports.txt: ${SERVER_PORTS[@]}"
echo ${SERVER_PORTS[@]} > server_ports.txt


# curl http://localhost:8000/v1/models
# INFO 11-26 15:52:33 launcher.py:14] Available routes are:
# INFO 11-26 15:52:33 launcher.py:22] Route: /openapi.json, Methods: GET, HEAD
# INFO 11-26 15:52:33 launcher.py:22] Route: /docs, Methods: GET, HEAD
# INFO 11-26 15:52:33 launcher.py:22] Route: /docs/oauth2-redirect, Methods: GET, HEAD
# INFO 11-26 15:52:33 launcher.py:22] Route: /redoc, Methods: GET, HEAD
# INFO 11-26 15:52:33 launcher.py:22] Route: /health, Methods: GET
# INFO 11-26 15:52:33 launcher.py:22] Route: /tokenize, Methods: POST
# INFO 11-26 15:52:33 launcher.py:22] Route: /detokenize, Methods: POST
# INFO 11-26 15:52:33 launcher.py:22] Route: /v1/models, Methods: GET
# INFO 11-26 15:52:33 launcher.py:22] Route: /version, Methods: GET
# INFO 11-26 15:52:33 launcher.py:22] Route: /v1/chat/completions, Methods: POST
# INFO 11-26 15:52:33 launcher.py:22] Route: /v1/completions, Methods: POST
# INFO 11-26 15:52:33 launcher.py:22] Route: /v1/embeddings, Methods: POST
