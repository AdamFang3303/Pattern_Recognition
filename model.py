import torch.nn as nn


class EnhancedCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.features = nn.Sequential(
            # 输入尺寸：1x28x28
            nn.Conv2d(config.in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 3))  # 调整池化尺寸
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),  # 扩大全连接层
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate // 2),
            nn.Linear(512, config.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)