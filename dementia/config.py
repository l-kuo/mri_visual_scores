import os

_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_dir)


class Config:
    # Model selection: 'resnet', 'densenet', 'vit'
    model_name = 'vit'
    num_classes = 3
    target_names = ['CN', 'MCI', 'AD']

    # Training
    world_size = 1
    batch_size = 8
    num_workers = 4
    pin_memory = True
    epochs = 100
    learning_rate = 1e-5
    weight_decay = 0.0005
    warmup_epochs = 10
    milestones = [50, 80]
    patient = 20

    # Paths
    weight_path = None
    version_name = 'v1'
    repo_root = _repo_root
    train_csv = os.path.join(_dir, 'train.csv')
    val_csv = os.path.join(_dir, 'val.csv')
    cache_dir = os.path.join(_dir, 'cache_dir')
    root_dir = _dir

    # ViT-specific (only used when model_name='vit')
    image_size = (96, 96)
    depth = 96
    image_patch_size = 16
    depth_patch_size = 16
    channels = 1
    dim = 768
    vit_layers = 12
    heads = 12
    pool = 'cls'
    dim_head = 64
    mlp_dim = 3072
    dropout = 0.5
    emb_dropout = 0.1
