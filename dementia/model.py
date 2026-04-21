import torch
from torch import nn
from monai.networks.nets import ResNetFeatures, DenseNet121
from vision_transformer3d import ViT


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = ResNetFeatures('resnet50', pretrained=False, spatial_dims=3, in_channels=1)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x[-1])
        x = x.view(x.size(0), -1)
        return self.fc(x)


def build_model(config):
    if config.model_name == 'resnet':
        return ResNetClassifier(num_classes=config.num_classes)

    elif config.model_name == 'densenet':
        return DenseNet121(
            spatial_dims=3, in_channels=1, out_channels=config.num_classes,
            init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
            pretrained=False,
        )

    elif config.model_name == 'vit':
        return ViT(
            image_size=config.image_size,
            depth=config.depth,
            image_patch_size=config.image_patch_size,
            depth_patch_size=config.depth_patch_size,
            num_classes=config.num_classes,
            channels=config.channels,
            dim=config.dim,
            layers=config.vit_layers,
            heads=config.heads,
            pool=config.pool,
            dim_head=config.dim_head,
            mlp_dim=config.mlp_dim,
            dropout=config.dropout,
            emb_dropout=config.emb_dropout,
        )

    else:
        raise ValueError(f"Unknown model_name '{config.model_name}'. Choose from: 'resnet', 'densenet', 'vit'")
