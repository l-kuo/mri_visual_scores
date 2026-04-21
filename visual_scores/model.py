import torch
from torch import nn
from monai.networks.nets import ResNet


class VisualScores(nn.Module):
    """
    Shared ResNet18 backbone with one classification head per score.
    scores: list of (name, num_classes), e.g. [('mta', 5), ('erica', 4)]
    forward() returns dict: {name: logits_tensor}
    """
    def __init__(self, scores):
        super().__init__()
        self.backbone = ResNet(
            block='basic',
            layers=[2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=1,
            conv1_t_size=7,
            conv1_t_stride=1,
            no_max_pool=False,
            shortcut_type='A',
            widen_factor=1,
            num_classes=512,
            feed_forward=False,
            bias_downsample=True,
        )
        self.heads = nn.ModuleDict({
            name: nn.Linear(512, num_classes) for name, num_classes in scores
        })

    def forward(self, x):
        features = self.backbone(x)
        return {name: head(features) for name, head in self.heads.items()}
