# Dementia Classification

3-class MRI classification: **CN** (0), **MCI** (1), **AD** (2).

## 1. Data

Place NIfTI files anywhere accessible. Paths in the CSVs are relative to the **repo root**.

### CSV format — `train.csv` / `val.csv`

| Column | Type | Description |
|---|---|---|
| `filepath` | str | Path to full-brain MRI (`.nii.gz`), relative to repo root |
| `label` | int | 0 = CN, 1 = MCI, 2 = AD |

Example:
```
filepath,label
sample_dataset/train/train_sample_0001/norm_to_mni305_1mm.nii.gz,0
sample_dataset/train/train_sample_0002/norm_to_mni305_1mm.nii.gz,1
```

## 2. Configuration — `config.py`

Key settings to adjust before training:

| Setting | Default | Description |
|---|---|---|
| `model_name` | `'vit'` | Model architecture: `'vit'`, `'resnet'`, or `'densenet'` |
| `epochs` | `100` | Number of training epochs |
| `batch_size` | `8` | Samples per GPU per step |
| `learning_rate` | `1e-5` | Initial learning rate (AdamW) |
| `warmup_epochs` | `10` | Linear LR warmup length |
| `milestones` | `[50, 80]` | Epochs at which LR is multiplied by 0.1 |
| `patient` | `20` | Early stopping patience (epochs) |
| `version_name` | `'v1'` | Run label, used in weight and log paths |
| `weight_path` | `None` | Path to pretrained weights to resume from |
| `train_csv` | `dementia/train.csv` | Auto-resolved; edit the CSV file directly |
| `val_csv` | `dementia/val.csv` | Auto-resolved; edit the CSV file directly |

## 3. Training

Run from inside the `dementia/` directory:

```bash
cd dementia
python train.py
```

MONAI will cache preprocessed MRI volumes on first run under `dementia/cache_dir/`. Subsequent runs skip preprocessing and load from cache.

## 4. Outputs

**Weights** are saved to:
```
dementia/weights/{model_name}/{version_name}/Ep.best_model.pth   ← best val loss
dementia/weights/{model_name}/{version_name}/Ep.{N}_model.pth    ← every 10 epochs
```

**TensorBoard logs** are written to a `runs/` directory in the working directory. View with:
```bash
tensorboard --logdir runs
```
