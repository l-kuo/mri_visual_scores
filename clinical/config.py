import os

_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_dir)


class Config:
    # Mode controls what inputs are combined:
    #   'clinical'            – tabular clinical features only
    #   'scores_clinical'     – visual scores + clinical features (no MRI)
    #   'mri_clinical'        – MRI + clinical features
    #   'mri_scores_clinical' – MRI + visual scores + clinical features
    mode = 'mri_scores_clinical'

    # MRI backbone (only used in mri_* modes): 'resnet' or 'densenet'
    mri_backbone = 'resnet'

    num_classes = 3
    target_names = ['CN', 'MCI', 'AD']

    # Clinical feature columns present in the CSV (applied to all modes)
    clinical_features = ['EXAMAGE', 'GENDER', 'PTEDUCAT', 'CDR', 'FAQ', 'TMSE', 'MOCA']

    # Visual score columns (only used in scores_* modes)
    score_features = ['GCA', 'MTA_RIGHT', 'MTA_LEFT', 'ERICA_RIGHT', 'ERICA_LEFT']

    # Training
    world_size = 1
    batch_size = 8
    num_workers = 4
    pin_memory = True
    epochs = 30
    learning_rate = 1e-6
    weight_decay = 0.0005
    warmup_epochs = 0
    milestones = [30, 100]
    patient = 10

    # Paths
    weight_path = None
    version_name = 'v1'
    repo_root = _repo_root
    train_csv = os.path.join(_dir, 'train.csv')
    val_csv = os.path.join(_dir, 'val.csv')
    cache_dir = os.path.join(_dir, 'cache')   # used in mri_* modes only
    root_dir = _dir
