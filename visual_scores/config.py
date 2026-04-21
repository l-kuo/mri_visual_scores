import os

_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_dir)


class Config:
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
    repo_root = _repo_root
    root_dir = _dir

    # MTA + ERICA settings
    # CSV columns required: filepath, MTA, ERICA
    # Each row = one side (left or right cropped hippocampal MRI)
    mta_erica_version = 'v1'
    mta_erica_train_csv = os.path.join(_dir, 'mta_erica_train.csv')
    mta_erica_val_csv = os.path.join(_dir, 'mta_erica_val.csv')
    mta_erica_cache_dir = os.path.join(_dir, 'cache', 'mta_erica')
    mta_classes = 5   # MTA scores: 0-4
    erica_classes = 4  # ERICA scores: 0-3

    # GCA settings
    # CSV columns required: filepath, GCA
    # Each row = one full-brain MRI
    gca_version = 'v1'
    gca_train_csv = os.path.join(_dir, 'gca_train.csv')
    gca_val_csv = os.path.join(_dir, 'gca_val.csv')
    gca_cache_dir = os.path.join(_dir, 'cache', 'gca')
    gca_classes = 4   # GCA scores: 0-3
