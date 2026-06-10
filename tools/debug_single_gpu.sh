#!/usr/bin/env bash
# for single gpu debug without ddp

# python ./tools/test_single_gpu.py \
#  ./projects/configs/stage2_e2e/fusion_base_e2e.py \
#  ./ckpts/fusion_latest.pth --eval bbox --show-dir ./projects/work_dirs/stage2_e2e/fusion_base_e2e/debug


python ./tools/test_single_gpu.py \
 ./projects/configs/stage2_e2e/graph_bev_debug.py \
 ./ckpts/fusion_latest.pth --eval bbox --show-dir ./projects/work_dirs/stage2_e2e/graph_bev/debug

python ./tools/train_single_gpu.py \
     projects/configs/stage1_track_map/fusion_base_track_map.py \
     --work-dir projects/work_dirs/stage1_track_map/debug