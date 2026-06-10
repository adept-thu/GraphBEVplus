#!/usr/bin/env bash
# ./tools/noise_test.sh 1 9 7
set -x
CFG=./projects/configs/stage2_e2e/fusion_base_e2e.py #
CKPT=./ckpts/fusion_latest.pth #

GPUS=$1
SEVERITY=$2 # 1~10
GPU_ENV=$3


T=`date +%m%d%H%M`

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT



WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
if [ ! -d ${WORK_DIR}logs_alignment_$SEVERITY ]; then
    mkdir -p ${WORK_DIR}logs_alignment_$SEVERITY
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=${GPU_ENV} python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CFG $CKPT --launcher pytorch --eval bbox \
    --show-dir ${WORK_DIR} --cfg-options model.pts_bbox_head.transformer.encoder.noise_severity=$SEVERITY \
    2>&1 | tee ${WORK_DIR}logs_alignment_$SEVERITY/eval_.$T