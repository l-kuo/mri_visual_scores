import torch
from torch import nn
from monai.networks.nets import ResNet, DenseNet121


class MRIEncoder(nn.Module):
    """
    Shared MRI feature extractor.
      'resnet'  → ResNet18, outputs 512-dim
      'densenet' → DenseNet121, outputs 1024-dim
    """
    def __init__(self, backbone):
        super().__init__()
        if backbone == 'resnet':
            self.net = ResNet(
                block='basic', layers=[2, 2, 2, 2], block_inplanes=[64, 128, 256, 512],
                spatial_dims=3, n_input_channels=1, num_classes=512,
                feed_forward=False, shortcut_type='A', bias_downsample=True,
            )
            self._out_dim = 512

        elif backbone == 'densenet':
            dense = DenseNet121(spatial_dims=3, in_channels=1, out_channels=3, pretrained=False)
            self.net = nn.Sequential(
                dense.features,
                dense.class_layers.norm5,
                dense.class_layers.relu5,
            )
            self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self._out_dim = 1024

        else:
            raise ValueError(f"Unknown backbone '{backbone}'. Choose 'resnet' or 'densenet'.")

        self.backbone = backbone

    @property
    def out_dim(self):
        return self._out_dim

    def forward(self, x):
        if self.backbone == 'resnet':
            return self.net(x)
        else:
            x = self.net(x)
            x = self.pool(x)
            return x.view(x.size(0), -1)


class ClinicalMLP(nn.Module):
    """
    Tabular-only model (clinical, scores_clinical modes).
    input_dim = len(clinical_features) or len(score_features + clinical_features)
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, features):
        return self.net(features)


class MRIFusion(nn.Module):
    """
    MRI + tabular fusion model (mri_clinical, mri_scores_clinical modes).
    tabular_dim = len(clinical_features) or len(score_features + clinical_features)
    """
    def __init__(self, backbone, tabular_dim, num_classes):
        super().__init__()
        self.mri_encoder = MRIEncoder(backbone)
        self.tabular_proj = nn.Sequential(
            nn.Linear(tabular_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(self.mri_encoder.out_dim + 64, num_classes)

    def forward(self, mri, features):
        mri_feat = self.mri_encoder(mri)
        tab_feat = self.tabular_proj(features)
        return self.classifier(torch.cat([mri_feat, tab_feat], dim=1))


def build_model(config):
    uses_mri = 'mri' in config.mode
    uses_scores = 'scores' in config.mode

    tabular_dim = len(config.clinical_features)
    if uses_scores:
        tabular_dim += len(config.score_features)

    if uses_mri:
        return MRIFusion(config.mri_backbone, tabular_dim, config.num_classes)
    else:
        return ClinicalMLP(tabular_dim, config.num_classes)
