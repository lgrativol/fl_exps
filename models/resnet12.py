from utils import *
from args import *
import torch.nn as nn
import torch.nn.functional as F

class BasicBlockRN12(nn.Module):
    def __init__(self, in_planes, planes,batchn):
        super(BasicBlockRN12, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        if(batchn):
            self.bn1 = nn.BatchNorm2d(planes)
        else:
            self.bn1 = nn.GroupNorm(2,planes)        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        if(batchn):
            self.bn2 = nn.BatchNorm2d(planes)
        else:
            self.bn2 = nn.GroupNorm(2,planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        if(batchn):
            self.bn3 = nn.BatchNorm2d(planes)
        else:
            self.bn3 = nn.GroupNorm(2,planes)

        if(batchn):
            norm = nn.BatchNorm2d(planes)
        else:
            norm = nn.GroupNorm(2,planes)


        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),norm)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope = 0.1)
        out = F.leaky_relu(self.bn2(self.conv2(out)), negative_slope = 0.1)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        # if args.dropout > 0:
        #     out = F.dropout(out, p=args.dropout, training=self.training, inplace=True)
        return out
    
class ResNet12(nn.Module):
    def __init__(self, feature_maps, input_shape, num_classes, batchn=False):
        super(ResNet12, self).__init__()        
        layers = []
        layers.append(BasicBlockRN12(input_shape[0], feature_maps,batchn))
        layers.append(BasicBlockRN12(feature_maps, int(2.5 * feature_maps),batchn))
        layers.append(BasicBlockRN12(int(2.5 * feature_maps), 5 * feature_maps,batchn))
        layers.append(BasicBlockRN12(5 * feature_maps, 10 * feature_maps,batchn))        
        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear(10 * feature_maps, num_classes)
        #self.rotations = rotations
        #self.linear_rot = nn.Linear(10 * feature_maps, 4)
        self.mp = nn.MaxPool2d((2,2))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = x
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            out = self.mp(F.leaky_relu(out, negative_slope = 0.1))
        out = F.avg_pool2d(out, out.shape[2])
        features = out.view(out.size(0), -1)
        out = self.linear(features)
        return out