
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn

def _make_qconv2d_fx(in_planes, planes, kernel_size=3,stride = 1, padding=0, bias=False,weight_bit_width=None):
    if(weight_bit_width == None):
        return qnn.QuantConv2d(in_channels =in_planes,
                               out_channels = planes,
                               kernel_size = kernel_size,
                               stride = stride,
                               padding = padding,
                               bias = bias,
                               weight_quant=None,
                               input_quant=None,
                               bias_quant=None,
                               output_quant=None,
                               update_wqi=None,
                               update_bqi=None,
                               update_iqi=None,
                               update_oqi=None,
                               )
    else:
        return qnn.QuantConv2d(in_channels = in_planes,
                        out_channels = planes,
                        kernel_size = kernel_size,
                        stride = stride,
                        padding = padding,
                        bias = bias,
                        weight_bit_width =weight_bit_width,return_quant_tensor=True)

def _make_qlinear_fx(in_features, out_features, bias=True, weight_bit_width=None):
    if (weight_bit_width == None):
        return qnn.QuantLinear(in_features=in_features,
                           out_features=out_features,
                           bias=bias,
                           weight_quant=None,
                           input_quant=None,
                           bias_quant=None,
                           output_quant=None,
                           update_wqi=None,
                           update_bqi=None,
                           update_iqi=None,
                           update_oqi=None)
    else:
        return qnn.QuantLinear(in_features=in_features,
                           out_features=out_features,
                           bias=bias,
                           weight_bit_width=weight_bit_width,
                           return_quant_tensor=True)

_make_qconv2d = _make_qconv2d_fx
_make_qlinear = _make_qlinear_fx 

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1,batchn=False,wbits=None):
        super(BasicBlock, self).__init__()
        self.conv1 = _make_qconv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,weight_bit_width=wbits)
        if(batchn):
            self.bn1 = nn.BatchNorm2d(planes)
        else:
            self.bn1 = nn.GroupNorm(2,planes)
        self.conv2 = _make_qconv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False,weight_bit_width=wbits)
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
                _make_qconv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False,weight_bit_width=wbits),norm)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return out

class QResNet(nn.Module):
    def __init__(self, block, num_blocks, feature_maps, input_shape, num_classes, batchn,wbits=8):
        super(QResNet, self).__init__()
        self.feature_maps = feature_maps
        self.in_planes = feature_maps
        self.length = len(num_blocks)
        self.conv1 = _make_qconv2d(input_shape[0], feature_maps, kernel_size=3, stride=1, padding=1, bias=False,weight_bit_width=wbits)
        if(batchn):
            self.bn1 = nn.BatchNorm2d(feature_maps)
        else:
            self.bn1 = nn.GroupNorm(2,feature_maps)

        layers = []
        for i, nb in enumerate(num_blocks):
            layers.append(self._make_layer(block, (2 ** i) * feature_maps, nb,stride = 1 if i == 0 else 2,wbits=wbits))
        self.layers = nn.Sequential(*layers)
        self.linear = _make_qlinear((2 ** (len(num_blocks) - 1)) * feature_maps, num_classes,wbits)
        self.depth = len(num_blocks)

    def _make_layer(self, block, planes, num_blocks, stride,batchn=False,wbits=None):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i in range(len(strides)):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride,batchn,wbits))
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

def QResNet8(feature_maps, input_shape, num_classes,batchn=False,wbits=None):
    return QResNet(BasicBlock, [1, 1, 1], feature_maps, input_shape, num_classes,batchn,wbits)

def QResNet18(feature_maps, input_shape, num_classes,batchn=False,wbits=None):
    return QResNet(BasicBlock, [2, 2, 2, 2], feature_maps, input_shape, num_classes,batchn,wbits)

def QResNet20(feature_maps, input_shape, num_classes, batchn=False,wbits=None):
    return QResNet(BasicBlock, [3, 3, 3], feature_maps, input_shape, num_classes, batchn,wbits)