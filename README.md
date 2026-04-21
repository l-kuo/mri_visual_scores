# AI-Powered Evaluation of Dementia Severity Based on Clinical Data and Visual Scoring Systems (MTA, ERICA, GCA) from MRI

Deep learning pipeline for dementia classification and visual scores prediction from 3D brain MRI. Three independent training modules share a common dataset structure.

## Repository Structure

```
mri_repo/
├── sample_dataset/          # 3D MRI data (NIfTI .nii.gz files)
│   ├── train/
│   │   └── {subject_id}/
│   │       ├── norm_to_mni305_1mm.nii.gz          # full-brain MRI
│   │       └── cropped/
│   │           ├── norm_to_mni305_1mm_left.nii.gz  # left medial temporal region
│   │           └── norm_to_mni305_1mm_right.nii.gz # right medial temporal region
│   └── test/
│       └── {subject_id}/   (same structure as train)
│
├── dementia/                # 3-class MRI classification (CN / MCI / AD)
├── visual_scores/           # Visual scores prediction (GCA, MTA, ERICA)
├── clinical/                # Multi-modal fusion (MRI + clinical tabular data)
└── requirements.txt
```

## Installation

### 1. Create and activate conda environment

```bash
conda create -n myEnv python=3.12
conda activate myEnv
```

### 2. Install PyTorch with CUDA

Install the CUDA-compatible build matching your GPU driver from [pytorch.org](https://pytorch.org/get-started/locally/). Example for CUDA 12.8:

```bash
pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

## Data

### Sources

| Dataset | Description | Access |
|---|---|---|
|[ ADNI ](https://adni.loni.usc.edu/)|The Alzheimer's Disease Neuroimaging Initiative (ADNI) seeks to develop biomarkers of the disease and advance the understanding of AD pathophysiology, improve diagnostic methods for early detection of AD and improve clinical trial design. Additional goals are examining the rate of progress for both mild cognitive impairment and Alzheimer's disease, as well as building a large repository of clinical and imaging data.|On Request|
| [ NACC ](https://www.naccdata.org/data-request-process/) | There are more than 54,000 participants with data at NACC. Data has been collected from more than 40 Alzheimer's Disease Research Centers (ADRCs) over the last 25+ years. | On Request |
| [ JADNI ](https://humandbs.dbcls.jp/en/hum0043-v1) | Japanese ADNI aims to establish imaging and biofluid markers that can predict and monitor the progression of changes in the brains of elderly individuals with AD, mild cognitive impairment (MCI) or normal cognition, and eventually be used as a surrogate markers for the clinical trials of DMDs for AD. | Restricted Access on Request |
| SIRIRAJ-MRI | The dataset obtained from Siriraj Hospital Thailand under ethical review board approval (COA no. SI 533/2023). This dataset was collected for this research project but not publicly available. | Private |

### Layout

Place MRI data under `sample_dataset/` at the repo root following the structure shown above. Each subject folder must contain the full-brain MRI at `norm_to_mni305_1mm.nii.gz`. The `cropped/` subdirectory is only required for `visual_scores` MTA+ERICA training.

CSV files in each module directory hold the file paths (relative to repo root) and labels/scores. See each module's README for the exact column format.

## Training Modules

### [`dementia/`](dementia/README.md)

Classifies MRI into CN / MCI / AD using ResNet18, DenseNet121, or a 3D Vision Transformer.

```bash
cd dementia
python train.py
```

### [`visual_scores/`](visual_scores/README.md)

Predicts Visual scores from MRI using a shared ResNet18 backbone with task-specific heads.

```bash
cd visual_scores
python train_gca.py          # GCA score (full-brain MRI)
python train_mta_erica.py    # MTA + ERICA scores (cropped hippocampal MRI)
```

### [`clinical/`](clinical/README.md)

Fuses MRI features with tabular clinical data (age, CDR, FAQ, TMSE, MOCA, etc.) for classification. Supports four input modes from pure tabular to fully fused.

```bash
cd clinical
python train.py
```

## Related Repositories

> **Note:** The `sample_dataset/` included in this repository contains mock data for structural reference only and is not suitable for training or evaluation. To replicate the results, you must obtain the data directly from the official sources listed in the [Data Sources](#sources) table above. We do not hold redistribution rights for any of the datasets used in this project.

---


| Repository | Description |
|---|---|
| [mri_visual_scores_web](https://github.com/l-kuo/mri_visual_scores_web) | Web application for MRI preprocessing and model inference. Handles skull stripping, MNI305 registration, hippocampal cropping, and serves trained models via a REST API. |