import os
import time
import random
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, MultiStepLR

from model import VisualScores
from config import Config
from logger import Logger

from monai.data import DistributedSampler, PersistentDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureTyped, ToTensord,
    EnsureChannelFirstd, DivisiblePadd, Orientationd, NormalizeIntensityd,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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


def get_transforms():
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
    return pipeline, pipeline


def train(rank, config):
    setup(rank, config.world_size)
    set_seed(27 + rank)

    scores = [('gca', config.gca_classes)]

    train_df = pd.read_csv(config.gca_train_csv, usecols=['filepath', 'GCA'])
    val_df = pd.read_csv(config.gca_val_csv, usecols=['filepath', 'GCA'])
    resolve = lambda p: os.path.abspath(os.path.join(config.repo_root, p))
    train_df['filepath'] = train_df['filepath'].apply(resolve)
    val_df['filepath'] = val_df['filepath'].apply(resolve)

    if rank == 0:
        print(f"GCA | Train: {len(train_df)} | Val: {len(val_df)}")

    train_transforms, val_transforms = get_transforms()

    train_ds = PersistentDataset(
        data=train_df.to_dict('records'), transform=train_transforms,
        cache_dir=os.path.join(config.gca_cache_dir, 'train')
    )
    val_ds = PersistentDataset(
        data=val_df.to_dict('records'), transform=val_transforms,
        cache_dir=os.path.join(config.gca_cache_dir, 'val')
    )

    train_sampler = DistributedSampler(train_ds, num_replicas=config.world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=config.world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=False,
                              num_workers=config.num_workers, pin_memory=config.pin_memory,
                              sampler=train_sampler, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=config.pin_memory,
                            sampler=val_sampler, worker_init_fn=worker_init_fn)

    model = VisualScores(scores=scores).to(rank)
    if config.weight_path:
        model = load_weights(model, config.weight_path, rank)
        if rank == 0:
            print("Loaded pretrained weights.")

    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  betas=(0.9, 0.999), eps=1e-8, weight_decay=config.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=config.warmup_epochs)
    main_scheduler = MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1)

    if rank == 0:
        logger = Logger('gca', config.gca_version)

    since = time.time()
    best_acc = 0.0
    best_epoch = 0
    patient = 0

    for epoch in range(config.epochs):
        train_sampler.set_epoch(epoch)
        epoch_start = time.time()

        model.train()
        running_train_loss = 0.0
        corrects = 0.0
        train_loop = tqdm(train_loader, desc=f"Train Epoch {epoch}", disable=(rank != 0))

        for data in train_loop:
            inputs = data['filepath'].to(rank)
            gca_labels = data['GCA'].long().to(rank)

            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output['gca'], gca_labels)
            loss.backward()
            optimizer.step()

            corrects += (torch.argmax(output['gca'], dim=1) == gca_labels).float().mean().item()
            running_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        if rank == 0:
            train_loss = running_train_loss / len(train_loader)
            train_acc = corrects / len(train_loader) * 100
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train GCA Acc: {train_acc:.2f}%")
            logger.log(epoch, 'Train Loss', train_loss)

        model.eval()
        running_val_loss = 0.0
        gca_gts, gca_preds = [], []
        val_loop = tqdm(val_loader, desc=f"Val   Epoch {epoch}", disable=(rank != 0))

        with torch.no_grad():
            for data in val_loop:
                inputs = data['filepath'].to(rank)
                gca_labels = data['GCA'].long().to(rank)
                output = model(inputs)
                loss = loss_fn(output['gca'], gca_labels)
                running_val_loss += loss.item()

                if rank == 0:
                    gca_gts.extend(gca_labels.cpu().numpy())
                    gca_preds.extend(torch.argmax(output['gca'], dim=1).cpu().numpy())

                val_loop.set_postfix(loss=loss.item())

        if rank == 0:
            val_loss = running_val_loss / len(val_loader)
            current_lr = optimizer.param_groups[0]['lr']
            logger.log(epoch, 'Val Loss', val_loss)
            logger.log_lr(epoch, current_lr)
            logger.display_status(epoch, 'Val Loss', val_loss)
            print(f"LR: {current_lr:.2e}")

            print("\nGCA Classification Report:")
            print(classification_report(gca_gts, gca_preds, digits=4))
            val_acc = accuracy_score(gca_gts, gca_preds)
            val_f1 = f1_score(gca_gts, gca_preds, average='macro')
            logger.log(epoch, 'Val GCA Acc', val_acc)
            logger.log(epoch, 'Val GCA F1', val_f1)

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                patient = 0
                logger.save_models(config.root_dir, model, 'best')
                print(f"Saved best model at epoch {epoch} (acc={best_acc:.4f}).")
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
        print(f"Best val GCA acc at epoch {best_epoch}: {best_acc:.4f}")
        logger.close()

    cleanup()


def main():
    config = Config()
    mp.spawn(train, args=(config,), nprocs=config.world_size, join=True)


if __name__ == '__main__':
    main()
