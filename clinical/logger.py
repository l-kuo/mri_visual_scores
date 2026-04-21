import os
import torch
from tensorboardX import SummaryWriter


class Logger:
    def __init__(self, model_name, version_name):
        self.model_name = model_name
        self.version_name = version_name
        self.data_subdir = f'{model_name}/{version_name}'
        self.writer = SummaryWriter(comment=f'_{model_name}_{version_name}')

    def log(self, cur_epoch, tag, value):
        self.writer.add_scalar(tag, value, cur_epoch)

    def log_lr(self, cur_epoch, current_lr):
        self.writer.add_scalar('Epoch LR', current_lr, cur_epoch)

    def display_status(self, cur_epoch, tag, value):
        print(f'{tag} Epoch {cur_epoch}: {value:.6f}')

    def save_models(self, root_dir, model, cur_epoch):
        out_dir = os.path.join(root_dir, 'weights', self.data_subdir)
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(out_dir, f'Ep.{cur_epoch}_model.pth'))

    def close(self):
        self.writer.close()
