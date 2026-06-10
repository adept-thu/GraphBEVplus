# Dataset Preparation

GraphBEVplus uses the nuScenes full dataset together with its CAN bus and map
extensions. The end-to-end pipeline and temporal data information follow
UniAD.

Before downloading the data, create a nuScenes account and accept the dataset
license on the [nuScenes website](https://www.nuscenes.org/).

## 1. Download nuScenes

Download the following files:

- nuScenes v1.0 full dataset
- nuScenes CAN bus expansion
- nuScenes map expansion v1.3

Place or link the extracted files under `UniAD/data/nuscenes`.

```bash
cd GraphBEVplus
mkdir -p UniAD/data/nuscenes
```

Large datasets can be stored outside the repository and linked into place:

```bash
ln -s /path/to/nuscenes/* UniAD/data/nuscenes/
```

After extraction, the directory should contain at least:

```text
GraphBEVplus/
|-- UniAD/
|   `-- data/
|       `-- nuscenes/
|           |-- can_bus/
|           |-- maps/
|           |-- samples/
|           |-- sweeps/
|           |-- v1.0-test/
|           `-- v1.0-trainval/
|-- projects/
`-- tools/
```

## 2. Prepare Temporal Data Information

Choose one of the following options.

### Option A: Download Pre-generated Information Files

```bash
cd GraphBEVplus
mkdir -p UniAD/data/infos
cd UniAD/data/infos

wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/nuscenes_infos_temporal_train.pkl
wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/nuscenes_infos_temporal_val.pkl
```

### Option B: Generate Information Files Locally

```bash
cd GraphBEVplus/UniAD
mkdir -p data/infos
./tools/uniad_create_data.sh
```

This generates:

```text
UniAD/data/infos/nuscenes_infos_temporal_train.pkl
UniAD/data/infos/nuscenes_infos_temporal_val.pkl
```

The generated pickle files may contain an absolute dataset root. If you use
locally generated files, check the selected config and set `data_root`
accordingly. In some UniAD configurations, `data_root` should be an empty
string because the absolute path is already stored in the pickle file.

## 3. Prepare Motion Anchors

```bash
cd GraphBEVplus
mkdir -p UniAD/data/others
cd UniAD/data/others

wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/motion_anchor_infos_mode6.pkl
```

## 4. Verify the Final Structure

```text
GraphBEVplus/
|-- UniAD/
|   `-- data/
|       |-- nuscenes/
|       |   |-- can_bus/
|       |   |-- maps/
|       |   |-- samples/
|       |   |-- sweeps/
|       |   |-- v1.0-test/
|       |   `-- v1.0-trainval/
|       |-- infos/
|       |   |-- nuscenes_infos_temporal_train.pkl
|       |   `-- nuscenes_infos_temporal_val.pkl
|       `-- others/
|           `-- motion_anchor_infos_mode6.pkl
|-- projects/
`-- tools/
```

Run these checks from the repository root:

```bash
test -d UniAD/data/nuscenes/samples
test -d UniAD/data/nuscenes/sweeps
test -f UniAD/data/infos/nuscenes_infos_temporal_train.pkl
test -f UniAD/data/infos/nuscenes_infos_temporal_val.pkl
test -f UniAD/data/others/motion_anchor_infos_mode6.pkl
echo "Dataset structure looks complete."
```

Once the data is ready, follow the upstream
[training and evaluation guide](https://github.com/adept-thu/GraphBEVplus/blob/main/docs/TRAIN_EVAL.md).
