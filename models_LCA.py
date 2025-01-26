import torch
import torch.nn as nn

class LCA(nn.Module):
    def __init__(self, channel, k_size=3):
        super(LCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 使用更小的卷积核
        self.conv = nn.Conv1d(1, 1, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        # 引入稀疏连接
        self.sparse_conv = nn.Conv2d(channel, channel, kernel_size=1, groups=channel//4, bias=False)

    def forward(self, x):
        # Global average pooling
        y = self.avg_pool(x)

        # Squeeze and Excitation
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        # 稀疏连接
        x = self.sparse_conv(x)

        return x * y.expand_as(x)

class LCANet(nn.Module):
    def __init__(self, class_num):
        super(LCANet, self).__init__()

        self.class_num = class_num

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            LCA(channel=128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, 1, 1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Hardswish(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.Hardswish(),

            LCA(channel=128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            LCA(channel=128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=448, out_channels=self.class_num, kernel_size=1, stride=1),
        )

    def forward(self, x):
        keep_features = list()

        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)

            if i in [0, 12, 17, 21]:
                keep_features.append(x)

        global_context = list()

        for i, f in enumerate(keep_features):
            f = nn.AdaptiveMaxPool2d(4)(f)
            global_context.append(f)

        x = torch.cat(global_context, 1)

        x = self.feature(x)

        logits = torch.mean(x, dim=[2, 3])

        return logits

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num