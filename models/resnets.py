
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1,batchn=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if(batchn):
            self.bn1 = nn.BatchNorm2d(planes)
        else:
            self.bn1 = nn.GroupNorm(2,planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if(batchn):
            self.bn2 = nn.BatchNorm2d(planes)
        else:
            self.bn2 = nn.GroupNorm(2,planes)

        self.shortcut = nn.Sequential()
        if(batchn):
            norm = nn.BatchNorm2d(planes)
        else:
            norm = nn.GroupNorm(2,planes)

        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),norm)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, feature_maps, input_shape, num_classes, batchn):
        super(ResNet, self).__init__()
        self.feature_maps = feature_maps
        self.in_planes = feature_maps
        self.length = len(num_blocks)
        self.conv1 = nn.Conv2d(input_shape[0], feature_maps, kernel_size=3, stride=1, padding=1, bias=False)
        if(batchn):
            self.bn1 = nn.BatchNorm2d(feature_maps)
        else:
            self.bn1 = nn.GroupNorm(2,feature_maps)

        layers = []
        for i, nb in enumerate(num_blocks):
            layers.append(self._make_layer(block, (2 ** i) * feature_maps, nb,stride = 1 if i == 0 else 2))
        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear((2 ** (len(num_blocks) - 1)) * feature_maps, num_classes)
        self.depth = len(num_blocks)

    def _make_layer(self, block, planes, num_blocks, stride,batchn=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i in range(len(strides)):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride,batchn))
            if i < len(strides) - 1:
                layers.append(nn.ReLU())
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
        out = F.relu(self.bn1(self.conv1(out)))
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            out = F.relu(out)
        out = F.avg_pool2d(out, out.shape[2])
        features = out.view(out.size(0), -1)
        out = self.linear(features)
        # return out, features
        return out

def ResNet8(feature_maps, input_shape, num_classes,batchn=False):
    return ResNet(BasicBlock, [1, 1, 1], feature_maps, input_shape, num_classes,batchn)

def ResNet18(feature_maps, input_shape, num_classes,batchn=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], feature_maps, input_shape, num_classes,batchn)

def ResNet20(feature_maps, input_shape, num_classes, batchn=False):
    return ResNet(BasicBlock, [3, 3, 3], feature_maps, input_shape, num_classes, batchn)