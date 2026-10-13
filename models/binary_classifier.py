import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.models import resnet18, ResNet18_Weights


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # hil-serl uses a resnet-10 with learned pooling. However, this seems to be performing
        # well enough while using built-in frozen weights
        weights = ResNet18_Weights.DEFAULT
        self.backbone = resnet18(weights=weights)  # Pretrained weights
        self.backbone.requires_grad_(False)        # Freeze weights
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
    
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.Dropout(0.1),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

if __name__ == "__main__":
    model = BinaryClassifier()
    images = torch.randn(2, 3, 224, 224)  # 2 128x128 RGB images
    outputs = model(images)
    print(outputs.shape)