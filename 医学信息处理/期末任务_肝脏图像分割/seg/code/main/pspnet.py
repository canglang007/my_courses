import torch
from torch import nn
from torch.nn import functional as F
import main.backbone as extractor2


class PSPModule(nn.Module):
    def __init__(self, features, out_features=256, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        # print(feats.shape) #
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.interpolate(input=x, size=(h, w), mode='bilinear', align_corners=False)
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, n_classes=19, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34',
                 pretrained=True):
        super().__init__()
        self.feats = getattr(extractor2, backend)(pretrained)
        self.psp = PSPModule(psp_size, 256, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(256, 64)
        self.up_2 = PSPUpsample(64, 16)
        self.up_3 = PSPUpsample(16, 16)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(16, n_classes, kernel_size=1),
            nn.LogSoftmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # print(x.shape) # torch.Size([1, 3, 304, 1000])
        #print(x.max(),x.min())
        f, class_f = self.feats(x) 
        # print(f.shape) # torch.Size([1, 512, 38, 125])
        # print(f.max(),f.min()) #47.8075-0
        
        p = self.psp(f)
        # print(p.shape) # torch.Size([1, 1024, 38, 125])
        #print(p.max(),p.min())
        #p = self.drop_1(p)

        p = self.up_1(p)
        # print(p.shape) # torch.Size([1, 256, 76, 250])
        #print(p.max(),p.min())
        #p = self.drop_2(p)

        p = self.up_2(p)
        # print(p.shape) # torch.Size([1, 64, 152, 500])
        #print(p.max(),p.min())
        #p = self.drop_2(p)

        p = self.up_3(p)
        # print(p.shape) # torch.Size([1, 64, 304, 1000])
        #print(p.max(),p.min())
        #p = self.drop_2(p)

        auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))
        #print(self.final(p).shape) # torch.Size([1, 9, 304, 1000])
        #print(self.final(p).max(),self.final(p).min())
        return self.final(p), self.classifier(auxiliary)
