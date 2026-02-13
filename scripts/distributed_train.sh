#!/bin/bash
set -euo pipefail

# Distributed training entrypoint for OpenPCDet (VoxelRCNN)
# Supports both local (single machine) and GCP (multi-node K8s) modes.
#
# Required env vars (both modes):
#   MODEL_CFG       - Path to model config YAML
#   DATASET_CFG     - Path to dataset config YAML
#
# GCP mode env vars (set by K8s indexed Job + autobaan-training orchestrator):
#   TRAINING_MODE=gcp
#   WORLD_SIZE      - Total number of nodes
#   MASTER_ADDR     - DNS name of pod-0 (headless service)
#   MASTER_PORT     - NCCL master port (default: 29500)
#   JOB_COMPLETION_INDEX - Pod index (0..N-1), becomes node rank
#   GCS_DATASET_PATH    - gs:// path to download dataset from
#   GCS_CHECKPOINT_PATH - gs:// path to upload checkpoints to (master only)

MODE=${TRAINING_MODE:-local}
NUM_GPUS=${NUM_GPUS_PER_NODE:-1}
MODEL_CFG=${MODEL_CFG:-/opt/OpenPCDet/tools/cfgs/autobaans_models/voxel_rcnn.yaml}

echo "=== GPU Diagnostics ==="
nvidia-smi || echo "nvidia-smi not available"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
echo "======================="

if [ "$MODE" = "gcp" ]; then
    RANK=${JOB_COMPLETION_INDEX:-0}
    CKPT_DIR=${CHECKPOINT_DIR:-/checkpoints/${RUN_TAG:-run}}

    echo "[GCP] rank=$RANK | world_size=$WORLD_SIZE | master=$MASTER_ADDR:${MASTER_PORT:-29500}"

    # Download dataset from GCS to local emptyDir volume
    echo "Downloading dataset from $GCS_DATASET_PATH..."
    python /opt/gcs_helper.py download "$GCS_DATASET_PATH" /data

    # Run distributed training (no exec — need to upload checkpoints after)
    torchrun \
        --nnodes="$WORLD_SIZE" \
        --node_rank="$RANK" \
        --nproc_per_node="$NUM_GPUS" \
        --master_addr="$MASTER_ADDR" \
        --master_port="${MASTER_PORT:-29500}" \
        /opt/OpenPCDet/tools/train.py \
        --launcher pytorch \
        --model_cfg "$MODEL_CFG" \
        --dataset_cfg "$DATASET_CFG" \
        --extra_tag "${RUN_TAG:-distributed}" \
        --batch_size "${BATCH_SIZE_PER_GPU:-5}" \
        --epochs "${EPOCHS:-50}" \
        ${EXTRA_ARGS:-}

    # Upload checkpoints to GCS (master pod only)
    if [ "$RANK" = "0" ]; then
        echo "Uploading checkpoints to $GCS_CHECKPOINT_PATH..."
        python /opt/gcs_helper.py upload "$CKPT_DIR" "$GCS_CHECKPOINT_PATH"
    fi
else
    CKPT_DIR=${CHECKPOINT_DIR:-./output}

    echo "[Local] nproc=$NUM_GPUS"

    exec torchrun \
        --nnodes=1 \
        --nproc_per_node="$NUM_GPUS" \
        /opt/OpenPCDet/tools/train.py \
        --launcher pytorch \
        --model_cfg "$MODEL_CFG" \
        --dataset_cfg "$DATASET_CFG" \
        --extra_tag "${RUN_TAG:-local}" \
        --batch_size "${BATCH_SIZE_PER_GPU:-5}" \
        --epochs "${EPOCHS:-50}" \
        ${EXTRA_ARGS:-}
fi
