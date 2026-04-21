# Visual Scores Prediction

Multi-task prediction of radiological scores from brain MRI. Two independent training scripts:

| Script | Task | Input MRI | Scores |
|---|---|---|---|
| `train_gca.py` | GCA atrophy | Full-brain | GCA (0–3) |
| `train_mta_erica.py` | Hippocampal atrophy | Cropped (left/right) | MTA (0–4), ERICA (0–3) |

## 1. Data

Paths in CSVs are relative to the **repo root**.

### GCA — `gca_train.csv` / `gca_val.csv`

One row per subject, full-brain MRI.

| Column | Type | Description |
|---|---|---|
| `filepath` | str | Path to full-brain MRI (`.nii.gz`), relative to repo root |
| `GCA` | int | Global Cortical Atrophy score: 0–3 |

```
filepath,GCA
sample_dataset/train/train_sample_0001/norm_to_mni305_1mm.nii.gz,0
```

### MTA + ERICA — `mta_erica_train.csv` / `mta_erica_val.csv`

Two rows per subject — one for each side (left, right) of the cropped hippocampal MRI.

| Column | Type | Description |
|---|---|---|
| `filepath` | str | Path to cropped MRI (`.nii.gz`), relative to repo root |
| `MTA` | int | Medial Temporal Atrophy score: 0–4 |
| `ERICA` | int | Entorhinal Cortex score: 0–3 |

```
filepath,MTA,ERICA
sample_dataset/train/train_sample_0001/cropped/norm_to_mni305_1mm_left.nii.gz,1,2
sample_dataset/train/train_sample_0001/cropped/norm_to_mni305_1mm_right.nii.gz,0,1
```

## 2. Configuration — `config.py`

| Setting | Default | Description |
|---|---|---|
| `epochs` | `30` | Training epochs |
| `batch_size` | `8` | Samples per GPU per step |
| `learning_rate` | `1e-6` | Initial learning rate (AdamW) |
| `warmup_epochs` | `0` | Linear LR warmup length |
| `milestones` | `[30, 100]` | Epochs at which LR is multiplied by 0.1 |
| `patient` | `10` | Early stopping patience |
| `gca_version` | `'v1'` | Run label for GCA weights/logs |
| `mta_erica_version` | `'v1'` | Run label for MTA+ERICA weights/logs |
| `weight_path` | `None` | Path to pretrained weights to resume from |

CSV paths (`gca_train_csv`, `mta_erica_train_csv`, etc.) are resolved automatically relative to `visual_scores/`.

## 3. Training

Run from inside the `visual_scores/` directory:

```bash
cd visual_scores

# Train GCA
python train_gca.py

# Train MTA + ERICA
python train_mta_erica.py
```

Preprocessed MRI volumes are cached on first run under `visual_scores/cache/gca/` and `visual_scores/cache/mta_erica/`.

## 4. Outputs

**Weights** are saved to:
```
visual_scores/weights/gca/{gca_version}/Ep.best_model.pth
visual_scores/weights/gca/{gca_version}/Ep.{N}_model.pth

visual_scores/weights/mta_erica/{mta_erica_version}/Ep.best_model.pth
visual_scores/weights/mta_erica/{mta_erica_version}/Ep.{N}_model.pth
```

**TensorBoard logs** are written to a `runs/` directory in the working directory:
```bash
tensorboard --logdir runs
```
