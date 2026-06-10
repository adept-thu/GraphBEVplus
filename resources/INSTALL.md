# Installation

This guide describes the environment used by the official GraphBEVplus
implementation. GraphBEVplus is built on UniAD and the OpenMMLab 1.x stack, so
the dependency versions below should be kept pinned.

## Requirements

- Linux
- NVIDIA GPU
- CUDA 11.1
- GCC 5 or newer
- Conda
- Python 3.8

## 1. Clone GraphBEVplus

```bash
git clone https://github.com/adept-thu/GraphBEVplus.git
cd GraphBEVplus
```

## 2. Create the Conda Environment

```bash
conda create -n graphbevpp python=3.8 -y
conda activate graphbevpp
```

## 3. Install PyTorch

The reference environment uses PyTorch 1.9.1 and CUDA 11.1.

```bash
conda install cudatoolkit=11.1.1 -c conda-forge

pip install \
  torch==1.9.1+cu111 \
  torchvision==0.10.1+cu111 \
  torchaudio==0.9.1 \
  -f https://download.pytorch.org/whl/torch_stable.html
```

Verify that PyTorch can access the GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 4. Configure GCC and CUDA

Make sure the compiler and CUDA toolkit are available before installing MMCV,
because several operators are compiled locally.

```bash
gcc --version
nvcc --version

export CUDA_HOME=/path/to/cuda-11.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

If your system GCC is too old, install a compatible compiler in the Conda
environment and export its `bin` directory to `PATH`.

## 5. Install OpenMMLab Dependencies

```bash
pip install mmcv-full==1.4.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

Install the compatible MMDetection3D version from source:

```bash
cd ..
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1

pip install scipy==1.7.3
pip install scikit-image==0.20.0
pip install -v -e .

cd ../GraphBEVplus
```

## 6. Verify the Environment

Run the following command from the GraphBEVplus root:

```bash
python -c "import torch, mmcv, mmdet, mmseg, mmdet3d; \
print('torch:', torch.__version__); \
print('cuda:', torch.version.cuda); \
print('mmcv:', mmcv.__version__); \
print('mmdet:', mmdet.__version__); \
print('mmseg:', mmseg.__version__); \
print('mmdet3d:', mmdet3d.__version__)"
```

Expected core versions:

| Package | Version |
| --- | --- |
| Python | 3.8 |
| PyTorch | 1.9.1 |
| CUDA | 11.1 |
| MMCV-Full | 1.4.0 |
| MMDetection | 2.14.0 |
| MMSegmentation | 0.14.1 |
| MMDetection3D | 0.17.1 |

## Troubleshooting

### MMCV operators fail to import

Confirm that the CUDA version used by PyTorch matches the version used to
compile MMCV:

```bash
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

Reinstall `mmcv-full` after correcting `CUDA_HOME`.

### CUDA out of memory

Reduce the per-GPU batch size in the selected config or use more GPUs. The
official end-to-end training command recommends eight or more GPUs.

### Dependency resolver upgrades pinned packages

Reinstall the exact versions in this guide. OpenMMLab 2.x packages are not
drop-in replacements for the 1.x stack used by GraphBEVplus.

## Next Step

Prepare nuScenes by following [Dataset Preparation](./DATA_PREPARE.md).
