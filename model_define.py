import torch as t
from torch import nn
from torchvision import models


class MyNet(nn.Module):

    def __init__(self, num_classes, cluster_count):
        """

        :param num_classes: 物体类别数目
        :param cluster_count: bounding box聚类中心数目
        """
        super(MyNet, self).__init__()
        self.cluster_count = cluster_count
        self.num_classes = num_classes
        self.base_model = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        self.last_conv = nn.Sequential(
            nn.Conv2d(in_channels=6144, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=cluster_count * (5 + num_classes), kernel_size=1, stride=1, padding=0)
        )
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=3)

    def forward(self, x):
        input_size = int(x.size()[-1])
        conv_16_ratio_downsample = None
        for n, m in self.base_model._modules.items():
            x = m(x)
            if input_size / x.size()[-1] == 16:
                conv_16_ratio_downsample = x
        conv_16_ratio_downsample = conv_16_ratio_downsample.view(conv_16_ratio_downsample.size()[0], -1, int(input_size / 32), int(input_size / 32))
        x = t.cat((x, conv_16_ratio_downsample), dim=1)
        x = self.last_conv(x)
        x = x.permute(0, 2, 3, 1)
        _x = x
        x = self.sigmoid(x)
        x_ = x.clone()
        ####################
        for i in range(self.cluster_count):
            x_[:, :, :, i * (5 + self.num_classes) + 2:i * (5 + self.num_classes) + 4] = _x[:, :, :, i * (5 + self.num_classes) + 2:i * (5 + self.num_classes) + 4]
            x_[:, :, :, i * (5 + self.num_classes) + 5:(i + 1) * (5 + self.num_classes)] = self.softmax(_x[:, :, :, i * (5 + self.num_classes) + 5:(i + 1) * (5 + self.num_classes)])
        ####################
        # 输出形状为(N, input_size / 32, input_size / 32, cluster_count * (5 + num_classes))
        return x_


if __name__ == "__main__":
    d = t.randn(1, 3, 448, 448)
    model = MyNet(20, 5)
    output = model(d)
    print(output.size())