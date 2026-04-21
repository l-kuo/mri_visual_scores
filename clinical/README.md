# Clinical Fusion

3-class classification (CN / MCI / AD) combining MRI and tabular clinical data. Supports four input modes controlled by `config.py`.

## 1. Modes

| `mode` | Inputs |
|---|---|
| `clinical` | Clinical features only |
| `scores_clinical` | visual scores + clinical features |
| `mri_clinical` | MRI + clinical features |
| `mri_scores_clinical` | MRI + visual scores + clinical features |

## 2. Data

Paths in the CSV are relative to the **repo root**. A single `train.csv` and `val.csv` are used for all modes — include all columns; the script selects only what the active mode requires.

### CSV format — `train.csv` / `val.csv`

| Column | Required for | Type | Notes |
|---|---|---|---|
| `filepath` | `mri_*` modes | str | Full-brain MRI path, relative to repo root |
| `label` | all | int | 0 = CN, 1 = MCI, 2 = AD |
| `EXAMAGE` | all | float | Age at examination |
| `GENDER` | all | int | 0 / 1 |
| `PTEDUCAT` | all | float | Years of education |
| `CDR` | all | float | Clinical Dementia Rating (0, 0.5, 1, 2, 3) |
| `FAQ` | all | float | Functional Activities Questionnaire (0–30) |
| `TMSE` | all | float | Thai Mini-Mental State Exam (0–30) |
| `MOCA` | all | float | Montreal Cognitive Assessment (0–30) |
| `GCA` | `scores_*` modes | int | Global Cortical Atrophy (0–3) |
| `MTA_RIGHT` | `scores_*` modes | int | Medial Temporal Atrophy right (0–4) |
| `MTA_LEFT` | `scores_*` modes | int | Medial Temporal Atrophy left (0–4) |
| `ERICA_RIGHT` | `scores_*` modes | int | Entorhinal Cortex right (0–3) |
| `ERICA_LEFT` | `scores_*` modes | int | Entorhinal Cortex left (0–3) |

Clinical features are z-score normalised automatically using training-set statistics.

## 3. Configuration — `config.py`

| Setting | Default | Description |
|---|---|---|
| `mode` | `'mri_scores_clinical'` | Input combination (see modes table above) |
| `mri_backbone` | `'resnet'` | MRI encoder: `'resnet'` (512-dim) or `'densenet'` (1024-dim); ignored in non-`mri_*` modes |
| `epochs` | `30` | Training epochs |
| `batch_size` | `8` | Samples per GPU per step |
| `learning_rate` | `1e-6` | Initial learning rate (AdamW) |
| `warmup_epochs` | `0` | Linear LR warmup length |
| `milestones` | `[30, 100]` | Epochs at which LR is multiplied by 0.1 |
| `patient` | `10` | Early stopping patience |
| `version_name` | `'v1'` | Run label used in weight and log paths |
| `weight_path` | `None` | Path to pretrained weights to resume from |

## 4. Training

Run from inside the `clinical/` directory:

```bash
cd clinical
python train.py
```

For `mri_*` modes, MONAI caches preprocessed MRI volumes on first run under `clinical/cache/`. Subsequent runs skip preprocessing.

## 5. Outputs

**Weights** are saved to:
```
clinical/weights/{mode}_{backbone}/   ← backbone is 'mlp' for non-mri modes
    {version_name}/Ep.best_model.pth  ← best val loss
    {version_name}/Ep.{N}_model.pth   ← every 10 epochs
```

For example, with `mode='mri_scores_clinical'`, `mri_backbone='resnet'`, `version_name='v1'`:
```
clinical/weights/mri_scores_clinical_resnet/v1/Ep.best_model.pth
```

**TensorBoard logs** are written to a `runs/` directory in the working directory:
```bash
tensorboard --logdir runs
```
