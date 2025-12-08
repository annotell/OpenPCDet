# KOGNIC OpenPCDet Fork - Training Guide

This document provides information specific to the KOGNIC fork of OpenPCDet, including how training is performed and which config files have been used.

**📚 For detailed training history and model comparisons, see [TRAINING_HISTORY.md](TRAINING_HISTORY.md)**

## Training Overview

The training process in this fork has been customized from the original OpenPCDet implementation. The main training script is located at `tools/train.py`.

### Key Differences from Original OpenPCDet

1. **Separate Model and Dataset Configs**: The training script uses separate `--model_cfg` and `--dataset_cfg` arguments instead of a single `--cfg_file`.
2. **Interactive Class Adjustment**: The training script includes a `define_classes_ux()` function that allows interactive class merging and adjustment during training setup.
3. **Config Saving**: Training configurations are automatically saved to the checkpoint directory with timestamps.
4. **Automatic Checkpoint Resuming**: The script automatically resumes from the latest checkpoint if available.

## Training Process

### Training Script

The main training script is `tools/train.py`. It supports:

- Single GPU training
- Multi-GPU distributed training (using PyTorch distributed)
- Checkpoint resuming
- Mixed precision training (AMP)
- Tensorboard logging

### Training Commands

#### Single GPU Training

```bash
cd tools
python train.py \
    --model_cfg cfgs/autobaans_models/voxel_rcnn.yaml \
    --dataset_cfg cfgs/dataset_configs/autobaans_dataset.yaml \
    --batch_size 5 \
    --epochs 50 \
    --workers 4
```

#### Multi-GPU Distributed Training

```bash
cd tools
bash scripts/dist_train.sh ${NUM_GPUS} \
    --model_cfg cfgs/autobaans_models/voxel_rcnn.yaml \
    --dataset_cfg cfgs/dataset_configs/autobaans_dataset.yaml \
    --batch_size 5 \
    --epochs 50 \
    --workers 4
```

Or using torch.distributed.launch directly:

```bash
cd tools
python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_endpoint=localhost:${PORT} \
    train.py \
    --launcher pytorch \
    --model_cfg cfgs/autobaans_models/voxel_rcnn.yaml \
    --dataset_cfg cfgs/dataset_configs/autobaans_dataset.yaml \
    --workers 12
```

### Training Arguments

Key arguments for `train.py`:

- `--model_cfg`: Path to model configuration file (optional, can be in dataset config)
- `--dataset_cfg`: Path to dataset configuration file (required)
- `--batch_size`: Batch size per GPU (optional, uses config default if not specified)
- `--epochs`: Number of training epochs (optional, uses config default if not specified)
- `--workers`: Number of dataloader workers (default: 4)
- `--extra_tag`: Extra tag for experiment naming (default: "default")
- `--ckpt`: Path to checkpoint to resume from (optional)
- `--pretrained_model`: Path to pretrained model weights (optional)
- `--launcher`: Launcher type - "none", "pytorch", or "slurm" (default: "none")
- `--sync_bn`: Use synchronized batch normalization (for multi-GPU)
- `--fix_random_seed`: Fix random seed for reproducibility
- `--use_amp`: Use automatic mixed precision training
- `--ckpt_save_interval`: Interval for saving checkpoints in epochs (default: 1)
- `--max_ckpt_save_num`: Maximum number of checkpoints to keep (default: 30)

## Config Files Used for Training

### Autobaans Dataset Models

The following model configurations have been created and used for training on the Autobaans dataset:

#### 1. VoxelRCNN (`tools/cfgs/autobaans_models/voxel_rcnn.yaml`)

- **Model**: VoxelRCNN with CenterHead
- **Architecture**:
  - VFE: DynMeanVFE (Dynamic Mean Voxel Feature Encoder)
  - Backbone 3D: VoxelResBackBone8x
  - Backbone 2D: BaseBEVResBackbone
  - Dense Head: CenterHead with IoU regression
  - ROI Head: VoxelRCNNHead
- **Training Parameters**:
  - Batch size per GPU: 5
  - Number of epochs: 50
  - Optimizer: adam_onecycle
  - Learning rate: 0.000025
  - Weight decay: 0.001
- **Class Adjustments**:
  - Car, Van, Truck, Bus, Trailer → Vehicle
  - Motorbike → VulnerableVehicle
  - Person → Pedestrian
- **Data Augmentation**:
  - Random world flip (x, y axes)
  - Random world rotation: [-45°, 45°]
  - Random world scaling: [0.95, 1.05]
  - Random world translation: std=[2.5, 2.5, 2.5]
- **Point Cloud Range**: [-100, -60, -2, 100, 60, 4]
- **Voxel Size**: [0.10, 0.10, 0.15]

#### 2. VoxelRCNN (Old Version) (`tools/cfgs/autobaans_models/voxel_rcnn_old.yaml`)

- **Model**: VoxelRCNN with CenterHead (older configuration)
- **Training Parameters**:
  - Batch size per GPU: 5
  - Number of epochs: 25
  - Optimizer: adam_onecycle
  - Learning rate: 0.000025
- **Note**: This appears to be an earlier version with fewer epochs

#### 3. PV-RCNN++ (`tools/cfgs/autobaans_models/pv_rcnn_plusplus.yaml`)

- **Model**: PV-RCNN++ (Point-Voxel R-CNN Plus Plus)
- **Architecture**:
  - VFE: MeanVFE
  - Backbone 3D: VoxelBackBone8x
  - Backbone 2D: BaseBEVBackbone
  - Dense Head: AnchorHeadSingle
  - PFE: VoxelSetAbstraction
  - Point Head: PointHeadSimple
  - ROI Head: PVRCNNHead
- **Training Parameters**:
  - Batch size per GPU: 8
  - Number of epochs: 25
  - Optimizer: adam_onecycle
  - Learning rate: 0.001
  - Weight decay: 0.001
- **Classes**: Medium, Large, VeryLarge
- **Anchor Sizes**:
  - Medium: [3.8, 1.8, 1.6]
  - Large: [6.5, 2.4, 3.0]
  - VeryLarge: [10.7, 2.7, 3.4]

### Dataset Configuration

#### Autobaans Dataset (`tools/cfgs/dataset_configs/autobaans_dataset.yaml`)

- **Dataset Type**: AutobaansDataset
- **Data Path**: `/mnt/bfd/datasets/autobaans/3dod/cosmos_proj_178`
- **Point Cloud Range**: [-100, -60, -2, 100, 60, 4]
- **Voxel Configuration**:
  - Voxel size: [0.10, 0.10, 0.15]
  - Max points per voxel: 10
  - Max number of voxels: 150,000 (train and test)
- **Point Features**: x, y, z, intensity
- **Data Splits**: 
  - Train: `train` (from train.pickle)
  - Test: `val` (from val.pickle)

## Training Output

### Output Directory Structure

Training outputs are saved to `/root/OpenPCDet/output/` by default (hardcoded in train.py line 293):

```
output/
└── autobaans_models/
    └── voxel_rcnn/
        └── default/
            ├── ckpt/                    # Checkpoints directory
            │   ├── checkpoint_epoch_*.pth
            │   └── autobaans_*.yaml      # Saved config files with timestamps
            ├── tensorboard/              # Tensorboard logs
            ├── train_*.log               # Training logs
            └── voxel_rcnn.yaml           # Copied model config
```

### Checkpoint Management

- Checkpoints are saved every epoch by default (`--ckpt_save_interval`)
- Latest checkpoint is automatically loaded if training is resumed
- Maximum 30 checkpoints are kept by default (`--max_ckpt_save_num`)
- Checkpoints are also saved every 5 minutes (`--ckpt_save_time_interval`)

## Class Extraction and Adjustment System

The KOGNIC fork uses a two-stage class management system:

### Stage 1: Dataset Extraction Config

The **dataset extraction config** (e.g., `dataset_creation/config.yaml`) defines which classes to extract from BigQuery:

```yaml
classes:
  - Car
  - Van
  - Truck
  - Bus
  - Motorbike
  - Person
  - Trailer
```

This determines what annotation classes are downloaded and stored in the dataset. Different projects may have different class names (e.g., `vehicle`, `pedestrian`, `VulnerableVehicle` in cosmos_proj_178 vs `Car`, `Van`, `Truck` in conti3d).

### Stage 2: Training Config Class Adjustments

The **training config** (e.g., `tools/cfgs/autobaans_models/voxel_rcnn.yaml`) uses `CLASS_ADJUSTMENTS` to translate the extracted classes into the **3 main classes** that all models should support:

1. **Vehicle** - All vehicle types (cars, trucks, vans, buses, trailers)
2. **VulnerableVehicle** - Vulnerable road users on vehicles (motorcycles, bicycles, etc.)
3. **Pedestrian** - Pedestrians and people

Example class adjustments from `voxel_rcnn.yaml`:
```yaml
CLASS_ADJUSTMENTS:
  'Car': 'Vehicle'
  'Van': 'Vehicle'
  'Truck': 'Vehicle'
  'Bus': 'Vehicle'
  'Motorbike': 'VulnerableVehicle'
  'Person': 'Pedestrian'
  'Trailer': 'Vehicle'
```

This mapping happens during training initialization, allowing the same model architecture to work with different source datasets that may use different class naming conventions.

### Class Adjustment Feature

During training initialization, the script provides an interactive class adjustment interface:

1. **Automatic Class Mapping**: If `CLASS_ADJUSTMENTS` is defined in the model config, classes are automatically merged according to the mapping.
2. **Interactive Mode**: If you choose not to use automatic mapping, you can manually specify:
   - Main classes for training
   - Which classes to merge into each main class
3. **Config Saving**: The final class configuration is saved to the checkpoint directory with a timestamp.

**Why this two-stage approach?**
- **Flexibility**: Different projects may use different class names, but we want consistent model outputs
- **Standardization**: All models output the same 3 main classes regardless of source data
- **Maintainability**: Easy to add new datasets without changing model architecture

## Training Tips

1. **Distributed Training**: Use `scripts/dist_train.sh` for multi-GPU training. The script automatically finds an available port.

2. **Resuming Training**: Simply run the same training command again - the script will automatically resume from the latest checkpoint.

3. **Monitoring**: Use Tensorboard to monitor training:
   ```bash
   tensorboard --logdir output/autobaans_models/voxel_rcnn/default/tensorboard
   ```

4. **Batch Size**: Ensure batch size is divisible by the number of GPUs when using distributed training.

5. **Mixed Precision**: Enable AMP with `--use_amp` for faster training and reduced memory usage.

6. **Workers**: Adjust `--workers` based on your system. More workers can speed up data loading but use more CPU/memory.

## Testing/Evaluation

To test a trained model:

```bash
cd tools
python test.py \
    --model_cfg cfgs/autobaans_models/voxel_rcnn.yaml \
    --dataset_cfg cfgs/dataset_configs/autobaans_dataset.yaml \
    --ckpt output/autobaans_models/voxel_rcnn/default/ckpt/checkpoint_epoch_50.pth \
    --batch_size 5
```

For multi-GPU testing:

```bash
cd tools
bash scripts/dist_test.sh ${NUM_GPUS} \
    --model_cfg cfgs/autobaans_models/voxel_rcnn.yaml \
    --dataset_cfg cfgs/dataset_configs/autobaans_dataset.yaml \
    --ckpt output/autobaans_models/voxel_rcnn/default/ckpt/checkpoint_epoch_50.pth
```

## Data Fetching and Preparation

### Overview

The training data is fetched from BigQuery and Kognic's file storage system, then processed into a format suitable for training. The process involves several steps:

1. **Fetching metadata from BigQuery** - Retrieves annotation data from the database
2. **Downloading point clouds** - Downloads LiDAR point cloud data from Kognic's file storage
3. **Creating train/val splits** - Splits the data into training and validation sets
4. **Generating pickle files** - Creates the `train.pickle` and `val.pickle` files used during training

### Data Fetching Process

#### Step 1: Fetch Metadata from BigQuery

The `dataset_creation/fetch_table.py` script queries BigQuery to retrieve annotation data:

```bash
cd dataset_creation
python fetch_table.py --config config.yaml
```

**What it does:**
- Queries the `annotell-com.dbt_shapes.shapes_training_cuboid` table in BigQuery
- Filters by project IDs or request IDs (specified in `config.yaml`)
- Filters by shape classes (Vehicle, Pedestrian, VulnerableVehicle)
- Retrieves geometries, timestamps, and sensor information
- Saves the result as a pickle file: `datatable_{id_list_name}_{ids}.pkl`

**Configuration (`dataset_creation/config.yaml`):**
```yaml
table: annotell-com.dbt_shapes.shapes_training_cuboid
projects: [178]  # Project IDs to fetch
requests: []     # Request IDs (if provided, projects are ignored)
classes:
  - Vehicle
  - Pedestrian
  - VulnerableVehicle
```

#### Step 2: Download Point Clouds and Annotations

The `dataset_creation/creator.py` script orchestrates the download process:

```bash
cd dataset_creation
python creator.py --config config.yaml
```

Or use the individual components:

```bash
# Fetch table (if not already done)
python fetch_table.py --config config.yaml

# Download dataset
python download_dataset.py
```

**What it does:**
1. Loads the datatable from the pickle file (or fetches it if not found)
2. Uses `DatasetLoader` to:
   - Download point clouds from Kognic's Potree file storage
   - Extract and transform 3D cuboid annotations
   - Save point clouds as `.npz` files in `{dataset_root}/{dataset_name}/pcs/`
   - Save annotations as `.pickle` files in `{dataset_root}/{dataset_name}/annos/`
3. Handles multi-LiDAR setups by splitting point clouds by sensor
4. Optionally filters point clouds to camera-visible regions (for Orion projects)
5. Transforms coordinates to LiDAR coordinate system (LCS) if configured

**Key features:**
- **Multi-threaded downloads**: Uses ThreadPoolExecutor for parallel downloads (configurable via `max_workers`)
- **Resume capability**: Skips already downloaded files
- **Multi-LiDAR support**: Handles scenes with multiple LiDAR sensors
- **Point cloud filtering**: Can filter to camera-visible regions (enabled with `filter_pc: True`)

**Configuration options:**
```yaml
dataset_root: '/mnt/bfd/datasets/autobaans/3dod'
dataset_name: 'cosmos_proj_178'
max_workers: 8              # Number of parallel download threads
filter_pc: True             # Filter point clouds to camera-visible regions
transform2LCS: False        # Transform coordinates to LiDAR coordinate system
create_splits: False        # Create train/val splits during download (deprecated)
```

#### Step 3: Create Train/Val Splits

The `dataset_creation/split_dataset.py` script creates the train/val split:

```bash
cd dataset_creation
python split_dataset.py --config config.yaml
```

**What it does:**
1. Lists all annotation files in the `annos/` directory
2. Validates that corresponding point cloud files exist and are valid
3. Randomly shuffles the list
4. Splits into train and validation sets based on `train_split` ratio
5. Saves file name lists to `train.pickle` and `val.pickle` in the dataset root

**Split configuration:**
```yaml
train_split: 1000  # If >= 1: number of train samples, if < 1: ratio (e.g., 0.9 = 90% train)
```

**Output:**
- `{dataset_root}/{dataset_name}/train.pickle` - List of training sample filenames (without extension)
- `{dataset_root}/{dataset_name}/val.pickle` - List of validation sample filenames (without extension)

**Note:** The pickle files contain lists of strings (filenames), not the actual data. Each filename corresponds to:
- Point cloud: `{filename}.npy.npz` in `pcs/` directory
- Annotations: `{filename}.pickle` in `annos/` directory

### Dataset Directory Structure

After data preparation, the dataset directory has the following structure:

```
{dataset_root}/{dataset_name}/
├── config.yaml                    # Copy of the dataset creation config
├── train.pickle                   # List of training sample filenames
├── val.pickle                     # List of validation sample filenames
├── pcs/                           # Point cloud files
│   ├── {judgement_id}_{timestamp}_{sensor_id}.npy.npz
│   └── ...
├── annos/                         # Annotation files
│   ├── {judgement_id}_{timestamp}_{sensor_id}.pickle
│   └── ...
└── gt_database/                   # Ground truth database (created during training prep)
    └── ...
```

### Point Cloud Format

Point clouds are stored as compressed NumPy arrays (`.npz` files):
- Shape: `[N, 4]` where N is the number of points
- Columns: `[x, y, z, intensity]`
- Intensity values are normalized (divided by 2^16) during loading

### Annotation Format

Annotations are stored as pickle files containing lists of cuboid dictionaries:
```python
[
    {
        "scale": [width, length, height],
        "coordinates": [x, y, z],
        "rotation": [qx, qy, qz, qw],  # Quaternion
        "class": "Vehicle",  # or "Pedestrian", "VulnerableVehicle", etc.
        "sensor": sensor_id,  # For multi-LiDAR setups
        "judgement_id": ...,
        "timestamp": ...
    },
    ...
]
```

### Data Loading During Training

During training, the `AutobaansDataset` class:

1. **Loads pickle files**: Reads `train.pickle` or `val.pickle` to get the list of sample filenames
2. **Loads point clouds**: For each sample, loads the corresponding `.npz` file from `pcs/`
3. **Loads annotations**: Loads the corresponding `.pickle` file from `annos/`
4. **Converts annotations**: Transforms cuboid format to OpenPCDet's format:
   - Converts quaternion rotation to Euler angle (yaw)
   - Applies class adjustments (e.g., Car → Vehicle)
   - Creates `gt_boxes_lidar` in format: `[x, y, z, length, width, height, yaw]`

### Complete Data Preparation Workflow

```bash
# 1. Fetch metadata from BigQuery
cd dataset_creation
python fetch_table.py --config config.yaml

# 2. Download point clouds and annotations
python creator.py --config config.yaml --yes  # --yes skips confirmation

# 3. Create train/val splits
python split_dataset.py --config config.yaml --yes

# 4. Verify the dataset structure
ls -la /mnt/bfd/datasets/autobaans/3dod/cosmos_proj_178/
# Should see: train.pickle, val.pickle, pcs/, annos/
```

### Troubleshooting

**Issue: Missing annotation files**
- Check that the annotation files exist in the `annos/` directory
- Verify that filenames in `train.pickle`/`val.pickle` match actual files

**Issue: Invalid point clouds**
- The split script validates point clouds and removes invalid ones
- Check logs for "Invalid pointcloud" messages

**Issue: Download failures**
- Check network connectivity to Kognic's file storage
- Verify authentication credentials
- Check that `max_workers` isn't too high (may cause rate limiting)

**Issue: Multi-LiDAR sensor issues**
- Ensure `is_multilidar` is correctly detected in the datatable
- Check that sensor IDs match between point clouds and annotations

## Training History and Lessons Learned

**For complete training history, see [TRAINING_HISTORY.md](TRAINING_HISTORY.md)**

### Current Production Model

The current production model is **`small_translation.pth`** located at:
```
/root/OpenPCDet/output/autobaans_models/voxel_rcnn/z+conti/ckpt/small_translation.pth
```

**Key characteristics:**
- **Dataset**: Merged Zenseact (cosmos_proj_178) + Conti (conti3d, projects 873, 874, 915)
- **Training date**: March 15, 2025
- **Translation augmentation**: [0.5, 0.5, 0.5] (small, appropriate level)
- **Key fixes applied**:
  - ✅ Removed cuboid projection in dataset creation (fixed "hovering boxes" issue)
  - ✅ Merged datasets for better coverage
  - ✅ Started from initial Cosmos model from January

**Known limitations:**
- Works sub-par for truck-mounted lidar
- Works sub-par for some multi-lidar configurations
- Multi-lidar tasks work "okay-ish" but not well on multi-lidar truck data like Kodiak

### Lessons Learned

**What worked:**
- ✅ **Removing cuboid projection** - Fixed hovering boxes issue in training data
- ✅ **Merging datasets** - Combining Zenseact + Conti improved performance
- ✅ **Small translation augmentation** - [0.5, 0.5, 0.5] provided good balance

**What didn't work:**
- ❌ **Larger translation augmentation** - Increasing to [2.5, 2.5, 2.5] (5x larger) reduced overall object count detected
- ❌ **Fine-tuning on smaller datasets** - Adding Kodiak data via fine-tuning made things worse
- ❌ **Fine-tuning approach** - Fine-tuning on specific datasets has not worked well; may need entirely new training merging datasets

**Recommendations for future training:**
- Start entirely new training merging datasets (not fine-tuning) to address imbalance issues
- Consider dataset balance when merging multiple sources
- Avoid excessive augmentation increases without careful validation

## Notes

- The output directory is currently hardcoded in `train.py` (line 293) to `/root/OpenPCDet/output/autobaans_models/voxel_rcnn/default`. This may need to be adjusted for different model configs.
- The training script expects the dataset to have `train.pickle` and `val.pickle` files in the dataset root directory.
- Class names are dynamically adjusted during training initialization, allowing flexible class merging strategies.
- Point cloud files are loaded with memory mapping (`mmap_mode="r"`) for efficient memory usage.
- The dataset automatically handles multi-LiDAR setups by splitting point clouds by sensor ID.
- **See [TRAINING_HISTORY.md](TRAINING_HISTORY.md) for complete training history, model comparisons, and detailed explanations of why certain models performed better or worse.**

