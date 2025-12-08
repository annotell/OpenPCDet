# Training History and Model Inventory

## Overview

This document lists all training attempts found in `/root/OpenPCDet/output/autobaans_models/voxel_rcnn/` and maps them to their source datasets and projects.

**Note**: 
- **Training Location**: `/root/OpenPCDet/output/autobaans_models/voxel_rcnn/` - This is where all training runs are stored
- **Data Location**: `/mnt/bfd/datasets/autobaans/3dod/` - Contains dataset folders (point clouds, annotations, splits) for different projects
- Training executes from `/root/OpenPCDet/` (likely in Docker), with data mounted from `/mnt/bfd/`

## Available Datasets and Their Projects

### 1. cosmos_proj_178
- **Location**: `/mnt/bfd/datasets/autobaans/3dod/cosmos_proj_178/`
- **Projects**: `[178]`
- **Classes**: Vehicle, Pedestrian, VulnerableVehicle, vehicle
- **Config**: `dataset_creation/config.yaml`
- **Train/Val Split**: 0.98 (98% train)
- **Filter PC**: True (filters to camera-visible regions)
- **Data Files**: 
  - `train.pickle` (20,113 bytes)
  - `val.pickle` (3,778,792 bytes)
  - `pcs/` directory with point cloud files
  - `annos/` directory with annotation files

### 2. conti3d
- **Location**: `/mnt/bfd/datasets/autobaans/3dod/conti3d/`
- **Projects**: `[873, 874, 915]`
- **Classes**: Car, Van, Truck, Bus, Motorbike, Person, Trailer
- **Config**: `dataset_creation/conti.yaml`
- **Train/Val Split**: 0.98
- **Filter PC**: False
- **Transform2LCS**: False

### 3. conti874
- **Location**: `/mnt/bfd/datasets/autobaans/3dod/conti874/`
- **Projects**: `[874]` (subset of conti3d)
- **Classes**: Car, Van, Truck, Bus, Motorbike, Person, Trailer
- **Config**: `dataset_creation/conti_test.yaml`
- **Train/Val Split**: 0.98
- **Filter PC**: False
- **Transform2LCS**: False

### 4. bosch_vru
- **Location**: `/mnt/bfd/datasets/autobaans/3dod/bosch_vru/`
- **Requests**: `[11221, 11176, 11144, 11143, 11127, 10934, 10833, 10677, 10617, 10601]`
- **Projects**: None (uses requests instead)
- **Classes**: Extensive list including Pedestrian, Rider, RidableVehicle, Motorbike, MotorScooter, Bicycle, RecumbentBicycle, CargoBicycle, AutoRickshaw, CycleRickshaw, Quad, Trike, BikeTrailer, Animal (Boar, Cat, Cow, Deer, Dog, Horse, Moose), MobilityDevice (Stroller, Scooter, SkateableMobilityDevice, WheelChair, WalkingFrame), Toy, BabyCarrier, ShoppingCart, IgnoreArea
- **Config**: `dataset_creation/orion.yaml`
- **Train/Val Split**: 0.98
- **Filter PC**: False
- **Transform2LCS**: False

### 5. kodiak3d
- **Location**: `/mnt/bfd/datasets/autobaans/3dod/kodiak3d/`
- **Projects**: `[921, 922, 1049]`
- **Classes**: vehicle, emergency_vehicle, trailer, sign_trailer, ego_trailer, motorbike, bicyclist, pedestrian
- **Config**: `dataset_creation/kodiak.yaml`
- **Train/Val Split**: 0.98
- **Filter PC**: False
- **Transform2LCS**: False

## Training Attempts Found

All training runs are located in `/root/OpenPCDet/output/autobaans_models/voxel_rcnn/`

### 1. VoxelRCNN (default) - **CURRENT ACTIVE TRAINING**
- **Location**: `/root/OpenPCDet/output/autobaans_models/voxel_rcnn/default/`
- **Model**: VoxelRCNN with CenterHead
- **Dataset**: bosch_vru (config saved May 23, 2025)
- **Checkpoints**: Up to epoch 39 (as of June 2025)
- **Latest Model**: `latest_model.pth` (June 4, 2025)
- **Bosch VRU Training (May 23, 2025)**:
  - ✅ **Training completed** (continued through June 2025, up to epoch 39)
  - Config file: `bosch_vru_20250523-095208.yaml` (saved May 23, 2025 at 09:52:08)
  - Training log: `train_20250523-095201.log` shows training started at 09:52:09
  - **Dataset**: bosch_vru - Completely different dataset focused on Vulnerable Road Users (VRU)
  - **Class configuration issue**: ⚠️ **CLASS_ADJUSTMENTS is a 1-to-1 (identity) mapping - useless for training**
    - The saved config shows CLASS_ADJUSTMENTS with all bosch_vru classes mapped to themselves (e.g., 'Rider': 'Rider', 'Bicycle': 'Bicycle', 'Pedestrian': 'Pedestrian')
    - This identity mapping provides no actual class adjustment - all classes remain as individual classes instead of being merged into the standard 3-class system
    - **Should be fixed**: bosch_vru classes should be mapped to **VulnerableVehicle** and **Pedestrian** (Vehicle class omitted since bosch_vru doesn't contain vehicles)
    - Example proper mapping should be:
      - All vulnerable vehicle types (RidableVehicle, Motorbike, MotorScooter, Bicycle, etc.) → 'VulnerableVehicle'
      - All pedestrian types (Rider, Pedestrian, etc.) → 'Pedestrian'
      - Animal classes could be mapped to 'VulnerableVehicle' or excluded
    - **This should conflict with the base model** which expects Vehicle/VulnerableVehicle/Pedestrian, but training completed - needs investigation
  - **Status**: ✅ **Training completed but never tested/evaluated** - Was meant to be tested but testing was never performed
  - **Note**: This is a different dataset with different purpose (VRU-focused), so not directly comparable to other models like small_translation
- **Status**: ✅ Training completed (up to epoch 39), but never tested/evaluated

### 2. VoxelRCNN (z+conti) - **PRODUCTION MODEL LOCATION**
- **Location**: `/root/OpenPCDet/output/autobaans_models/voxel_rcnn/z+conti/`
- **Model**: VoxelRCNN with CenterHead
- **Dataset**: Merged Zenseact + Conti dataset
  - **Note**: Config shows `DATA_PATH: /mnt/bfd/datasets/autobaans/3dod/conti3d` but this is a merged dataset
  - Zenseact data likely refers to cosmos_proj_178 (Project 178) or another Zenseact project
  - Conti data from conti3d (Projects 873, 874, 915) - 3 Conti projects with surround point cloud annotations
- **Key Improvements in this training**:
  - ✅ **Removed cuboid projection in dataset creation** - Fixed the "hovering boxes" issue in training data
  - ✅ **Started from initial Cosmos model from January** - Re-ran training from start (likely based on default_old or similar January/February model)
  - ✅ **Merged Zenseact + Conti datasets** - Better coverage
- **Checkpoints**: 50 epochs (checkpoint_epoch_25.pth through checkpoint_epoch_50.pth)
- **Key Models**:
  - ✅ **`small_translation.pth`** - **CURRENT PRODUCTION MODEL** (March 15, 2025, epoch 25)
    - **Why it's good**: 
      - No hovering boxes (fixed cuboid projection issue)
      - Better detections, especially behind ego (confidence >0.8)
      - Translation augmentation: [0.5, 0.5, 0.5] (small, appropriate level)
      - Trained on merged Zenseact + Conti with fixed dataset creation
    - **Known limitations**: Works sub-par for truck-mounted lidar and some multi-lidar configurations
  - ❌ `large_translation.pth` - Larger translation augmentation (April 2, 2025)
    - **Why it's worse**: Increased translation augmentation to [2.5, 2.5, 2.5] (5x larger) to try to cover more mounting geometries
    - **Result**: "Weren't better, actually reduced the overall object count detected"
    - **Conclusion**: Larger translation augmentation did not help and made performance worse
  - ❌ `z_conti_kodiak.pth` - Zenseact + Conti + Kodiak merged (April 2, 2025)
    - **Why it's worse**: Added Kodiak data to improve generalization for multi-lidar and truck-mounted sensors
    - **Result**: "Even worse" - Made things worse than small_translation
    - **Issue**: Fine-tuned on smaller, more specific dataset instead of starting fresh
    - **Conclusion**: Fine-tuning approach didn't work; may need entirely new training merging datasets
- **Training Period**: March - April 2025
- **Status**: ✅ Complete (50 epochs trained)

### 3. VoxelRCNN (c+c_train)
- **Location**: `/root/OpenPCDet/output/autobaans_models/voxel_rcnn/c+c_train/`
- **Model**: VoxelRCNN with CenterHead
- **Dataset**: conti3d (Projects 873, 874, 915)
- **Checkpoints**: Up to epoch 38 (February 2025)
- **Latest Model**: `latest_model.pth` (February 24, 2025)
- **Why it's not as good as small_translation**:
  - ❌ **Likely had cuboid projection issue** - Would have caused hovering boxes in training data
  - ❌ **Only Conti data** - Missing Zenseact data, less diverse dataset
  - ❌ **Trained before the fix** - This predates the March 2025 fix that removed cuboid projection
- **Status**: ✅ Complete (but superseded by z+conti with fixes)

### 4. VoxelRCNN (just_conti)
- **Location**: `/root/OpenPCDet/output/autobaans_models/voxel_rcnn/just_conti/`
- **Model**: VoxelRCNN with CenterHead
- **Dataset**: conti3d (Projects 873, 874, 915)
- **Checkpoints**: Only 2 epochs
- **Why it's not as good as small_translation**:
  - ❌ **Incomplete training** - Only 2 epochs, insufficient training
  - ❌ **Only Conti data** - Missing Zenseact data
  - ❌ **Early experiment** - Likely predates fixes and improvements
- **Status**: ⚠️ Incomplete (early experiment)

### 5. VoxelRCNN (default_old)
- **Location**: `/root/OpenPCDet/output/autobaans_models/voxel_rcnn/default_old/`
- **Model**: VoxelRCNN with CenterHead
- **Dataset**: conti3d (config from February 14, 2025)
- **Checkpoints**: 25 epochs
- **Why it's not as good as small_translation**:
  - ❌ **Had cuboid projection issue** - This predates the March 2025 fix, so would have had hovering boxes
  - ❌ **Only Conti data** - Missing Zenseact data
  - ❌ **Trained before fixes** - This is the "initial Cosmos model from January" that was used as a starting point for the successful z+conti training
- **Status**: ✅ Complete (archived/old version - used as starting point for z+conti)

### 6. VoxelRCNN (default_old2)
- **Location**: `/root/OpenPCDet/output/autobaans_models/voxel_rcnn/default_old2/`
- **Model**: VoxelRCNN with CenterHead
- **Dataset**: conti3d (configs from February 18-20, 2025)
- **Checkpoints**: 29 epochs
- **Why it's not as good as small_translation**:
  - ❌ **Had cuboid projection issue** - This predates the March 2025 fix, so would have had hovering boxes
  - ❌ **Only Conti data** - Missing Zenseact data
  - ❌ **Trained before fixes** - Multiple training attempts before the successful fix
- **Status**: ✅ Complete (archived/old version)


## Model File Summary

### Key Model Files

**Production Models**:
- ✅ **`small_translation.pth`** - Current production model (Zenseact + Conti, epoch 25)
  - Location: `/root/OpenPCDet/output/autobaans_models/voxel_rcnn/z+conti/ckpt/`
  - Date: March 15, 2025
- `large_translation.pth` - Larger translation augmentation variant
- `z_conti_kodiak.pth` - Extended dataset (Zenseact + Conti + Kodiak)

**Checkpoint Types**:
- `checkpoint_epoch_*.pth` - Epoch-specific checkpoints
- `latest_model.pth` - Latest checkpoint (auto-saved)
- Named models (e.g., `small_translation.pth`) - Specific model variants saved for production use

## Current Model Status

**✅ Current Production Model**: `small_translation.pth`
- **Location**: `/root/OpenPCDet/output/autobaans_models/voxel_rcnn/z+conti/ckpt/small_translation.pth`
- **Dataset**: Merged Zenseact + Conti dataset
- **Config**: `small_translation.yaml` (saved March 19, 2025)
- **Training Date**: March 2025
- **Epoch**: Trained as part of z+conti training run (epoch 25 checkpoint)
- **Size**: ~247 MB
- **Status**: ✅ **This is the current model in use**

**Other Notable Models** (and why they're not as good):
- **large_translation.pth**: Same location, larger translation augmentation (saved April 2, 2025)
  - **Issue**: Increased translation augmentation to [2.5, 2.5, 2.5] (5x larger than small_translation's [0.5, 0.5, 0.5])
  - **Result**: Reduced overall object count detected, didn't improve generalization for truck-mounted lidar
  - **Conclusion**: Larger translation augmentation was insufficient and made performance worse
  
- **z_conti_kodiak.pth**: Zenseact + Conti + Kodiak merged dataset (saved April 2, 2025)
  - **Issue**: Fine-tuned on smaller, more specific dataset (added Kodiak for multi-lidar/truck-mounted sensors)
  - **Result**: "Even worse" - Made things worse than small_translation
  - **Conclusion**: Fine-tuning approach didn't work; may need entirely new training merging datasets instead of fine-tuning

## Dataset to Project Mapping Summary

| Dataset Name | Projects/Requests | Primary Use Case | Used In Training |
|-------------|-------------------|------------------|----------------|
| cosmos_proj_178 | Project 178 | General 3D object detection (likely Zenseact) | z+conti (merged) |
| conti3d | Projects 873, 874, 915 | Continental dataset (multi-project) | z+conti, c+c_train, just_conti, default_old, default_old2 |
| conti874 | Project 874 | Continental dataset (single project subset) | - |
| bosch_vru | Requests: 11221, 11176, 11144, 11143, 11127, 10934, 10833, 10677, 10617, 10601 | Bosch VRU (Vulnerable Road Users) detection | default (current) |
| kodiak3d | Projects 921, 922, 1049 | Kodiak autonomous vehicle dataset | z_conti_kodiak (merged) |

**Merged Datasets**:
- **z+conti**: Zenseact (cosmos_proj_178) + Conti (conti3d) - **Used for production model `small_translation.pth`**
- **z_conti_kodiak**: Zenseact + Conti + Kodiak - Extended merged dataset

## Recommendations

1. **Current Production Model**:
   - ✅ **`small_translation.pth`** at `/root/OpenPCDet/output/autobaans_models/voxel_rcnn/z+conti/ckpt/small_translation.pth`
   - This is the model currently in use
   - Trained on merged Zenseact + Conti dataset
   - Epoch 25 checkpoint from 50-epoch training run

2. **For Model Deployment**:
   - Use `small_translation.pth` for production
   - Config file available: `small_translation.yaml` in same directory
   - Model size: ~247 MB

3. **For Further Training**:
   - Current active training: `/root/OpenPCDet/output/autobaans_models/voxel_rcnn/default/` (bosch_vru dataset)
   - Check tensorboard logs for training metrics
   - Review training logs in each directory for detailed training history

4. **Lessons Learned from Training History**:
   - ✅ **Removing cuboid projection** fixed hovering boxes issue
   - ✅ **Merging datasets** (Zenseact + Conti) improved performance
   - ❌ **Larger translation augmentation** (5x) did not help and reduced detection count
   - ❌ **Fine-tuning on smaller datasets** (adding Kodiak) made things worse
   - ⚠️ **Open question**: Should start entirely new training merging datasets vs fine-tuning? Fine-tuning approach has not worked well.

5. **Known Issues with small_translation.pth**:
   - Works sub-par for truck-mounted lidar
   - Works sub-par for some multi-lidar configurations
   - Multi-lidar tasks (e.g., Stellantis, Kodiak) work "okay-ish" but not well on multi-lidar truck data like Kodiak
   - **Next steps to consider**: Start entirely new training merging datasets (not fine-tuning) to address imbalance issues

## Clarifications Needed

**Questions for clarification**:
1. **Which exact model was the "initial Cosmos model from January"** that was used as the starting point for the successful z+conti training? 
   - Assumed to be `default_old` (February 14, 2025) or a January model, but confirmation would be helpful
2. **Was the z+conti training a complete re-training from scratch** or did it continue/fine-tune from the January model?
   - The text says "re-run the training from start" but also mentions "continue training on the existing model" - clarification on the exact approach would help
3. **What was the exact dataset merging process** for z+conti?
   - How were Zenseact (cosmos_proj_178) and Conti (conti3d) datasets merged?
   - Was it at the data level (combined train.pickle/val.pickle) or training level?

