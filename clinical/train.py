import os
import time
import random
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LinearLR, MultiStepLR

from model import build_model
from config import Config
from logger import Logger
from sklearn.metrics import classification_report

from monai.data import DistributedSampler as MonaiDistributedSampler, PersistentDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureTyped, ToTensord,
    EnsureChannelFirstd, DivisiblePadd, Orientationd, NormalizeIntensityd,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_weights(model, weight_path, device):
    weights = torch.load(weight_path, map_location=device, weights_only=True)
    weights = {k.replace('module.', ''): v for k, v in weights.items()}
    model.load_state_dict(weights)
    return model


def print_gpu_usage():
    try:
        mem = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv,nounits,noheader']
        )
        util = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader']
        )
        for i, (m, u) in enumerate(zip(mem.decode().strip().split('\n'), util.decode().strip().split('\n'))):
            used, free = map(int, m.split(','))
            print(f"GPU {i}: {u.strip()}% util | {used} MiB used | {free} MiB free")
    except subprocess.CalledProcessError:
        pass


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class TabularDataset(Dataset):
    """Simple dataset for non-MRI modes. Returns pre-normalized feature tensors."""
    def __init__(self, df, feature_cols, label_col='label'):
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.labels = torch.tensor(df[label_col].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'features': self.features[idx], 'label': self.labels[idx]}


def get_feature_cols(config):
    cols = list(config.clinical_features)
    if 'scores' in config.mode:
        cols = list(config.score_features) + cols
    return cols


def normalize_features(train_df, val_df, cols):
    """Z-score normalize using training set statistics. NaN filled with column mean."""
    train_df = train_df.copy()
    val_df = val_df.copy()
    means = train_df[cols].mean()
    stds = train_df[cols].std().replace(0, 1)
    train_df[cols] = (train_df[cols] - means) / stds
    val_df[cols] = (val_df[cols] - means) / stds
    train_df[cols] = train_df[cols].fillna(0)
    val_df[cols] = val_df[cols].fillna(0)
    return train_df, val_df


def get_mri_transforms():
    keys = ['filepath']
    pipeline = Compose([
        LoadImaged(keys, reader='NibabelReader'),
        EnsureChannelFirstd(keys),
        EnsureTyped(keys),
        Orientationd(keys, as_closest_canonical=True),
        DivisiblePadd(keys, k=32),
        NormalizeIntensityd(keys, subtrahend=0, divisor=255),
        ToTensord(keys, dtype=torch.float32),
    ])
    return pipeline


def build_tabular_tensor(data, config):
    """Assemble tabular feature tensor [B, n_features] from a MONAI batch dict."""
    cols = get_feature_cols(config)
    return torch.stack([data[col].float() for col in cols], dim=1)


def build_dataloaders(rank, config):
    train_df = pd.read_csv(config.train_csv)
    val_df = pd.read_csv(config.val_csv)
    resolve = lambda p: os.path.abspath(os.path.join(config.repo_root, p))
    train_df['filepath'] = train_df['filepath'].apply(resolve)
    val_df['filepath'] = val_df['filepath'].apply(resolve)

    feature_cols = get_feature_cols(config)
    train_df, val_df = normalize_features(train_df, val_df, feature_cols)

    uses_mri = 'mri' in config.mode

    if uses_mri:
        keep_cols = ['filepath', 'label'] + feature_cols
        train_records = train_df[keep_cols].to_dict('records')
        val_records = val_df[keep_cols].to_dict('records')

        transforms = get_mri_transforms()
        train_ds = PersistentDataset(train_records, transforms,
                                     cache_dir=os.path.join(config.cache_dir, 'train'))
        val_ds = PersistentDataset(val_records, transforms,
                                   cache_dir=os.path.join(config.cache_dir, 'val'))

        train_sampler = MonaiDistributedSampler(train_ds, num_replicas=config.world_size,
                                                rank=rank, shuffle=True)
        val_sampler = MonaiDistributedSampler(val_ds, num_replicas=config.world_size,
                                              rank=rank, shuffle=False)
    else:
        train_ds = TabularDataset(train_df, feature_cols)
        val_ds = TabularDataset(val_df, feature_cols)
        train_sampler = DistributedSampler(train_ds, num_replicas=config.world_size,
                                           rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=config.world_size,
                                         rank=rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=False,
                              num_workers=config.num_workers, pin_memory=config.pin_memory,
                              sampler=train_sampler, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=config.pin_memory,
                            sampler=val_sampler, worker_init_fn=worker_init_fn)
    return train_loader, val_loader, train_sampler


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def forward_pass(model, data, config, rank):
    """Returns logits and labels regardless of mode."""
    labels = data['label'].long().to(rank)
    uses_mri = 'mri' in config.mode

    if uses_mri:
        mri = data['filepath'].to(rank)
        features = build_tabular_tensor(data, config).to(rank)
        logits = model(mri, features)
    else:
        features = data['features'].to(rank)
        logits = model(features)

    return logits, labels


def train(rank, config):
    setup(rank, config.world_size)
    set_seed(27 + rank)

    train_loader, val_loader, train_sampler = build_dataloaders(rank, config)

    if rank == 0:
        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset)
        print(f"Mode: {config.mode} | Backbone: {config.mri_backbone if 'mri' in config.mode else 'MLP'}")
        print(f"Train: {n_train} | Val: {n_val}")

    model = build_model(config).to(rank)
    if config.weight_path:
        model = load_weights(model, config.weight_path, rank)
        if rank == 0:
            print("Loaded pretrained weights.")

    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  betas=(0.9, 0.999), eps=1e-8, weight_decay=config.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                                 total_iters=config.warmup_epochs)
    main_scheduler = MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1)

    if rank == 0:
        logger = Logger(f'{config.mode}_{config.mri_backbone if "mri" in config.mode else "mlp"}',
                        config.version_name)

    since = time.time()
    best_loss = float('inf')
    best_epoch = 0
    patient = 0

    for epoch in range(config.epochs):
        train_sampler.set_epoch(epoch)
        epoch_start = time.time()

        # --- Training ---
        model.train()
        running_train_loss = 0.0
        corrects = 0.0
        train_loop = tqdm(train_loader, desc=f"Train Epoch {epoch}", disable=(rank != 0))

        for data in train_loop:
            optimizer.zero_grad()
            logits, labels = forward_pass(model, data, config, rank)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            corrects += (torch.argmax(logits, dim=1) == labels).float().mean().item()
            running_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        if rank == 0:
            train_loss = running_train_loss / len(train_loader)
            train_acc = corrects / len(train_loader) * 100
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            logger.log(epoch, 'Train Loss', train_loss)

        # --- Validation ---
        model.eval()
        running_val_loss = 0.0
        preds, targets = [], []
        val_loop = tqdm(val_loader, desc=f"Val   Epoch {epoch}", disable=(rank != 0))

        with torch.no_grad():
            for data in val_loop:
                logits, labels = forward_pass(model, data, config, rank)
                loss = loss_fn(logits, labels)
                running_val_loss += loss.item()

                if rank == 0:
                    preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    targets.extend(labels.cpu().numpy())

                val_loop.set_postfix(loss=loss.item())

        if rank == 0:
            val_loss = running_val_loss / len(val_loader)
            current_lr = optimizer.param_groups[0]['lr']
            print(classification_report(targets, preds, target_names=config.target_names, labels=list(range(config.num_classes)), digits=4))
            print(f"Epoch {epoch} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")
            logger.log(epoch, 'Val Loss', val_loss)
            logger.log_lr(epoch, current_lr)

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                patient = 0
                logger.save_models(config.root_dir, model, 'best')
                print(f"Saved best model at epoch {epoch}.")
            else:
                patient += 1

            if (epoch + 1) % 10 == 0:
                logger.save_models(config.root_dir, model, epoch)

            elapsed = time.time() - epoch_start
            print(f"Epoch time: {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
            print_gpu_usage()

            if patient == config.patient:
                print(f"Early stopping at epoch {epoch}.")
                cleanup()
                break

        if epoch < config.warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()

    if rank == 0:
        elapsed = time.time() - since
        print(f"Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
        print(f"Best val loss at epoch {best_epoch}: {best_loss:.6f}")
        logger.close()

    cleanup()


def main():
    config = Config()
    mp.spawn(train, args=(config,), nprocs=config.world_size, join=True)


if __name__ == '__main__':
    main()
