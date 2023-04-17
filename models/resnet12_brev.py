import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
from args import args

def _make_qconv2d_fx(in_planes, planes, kernel_size=3, padding=0, bias=False,weight_bit_width=None):
    if(weight_bit_width == None):
        return qnn.QuantConv2d(in_channels =in_planes,
                               out_channels = planes,
                               kernel_size = kernel_size,
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

# def _make_qconv2d_bnn(in_planes, planes, kernel_size=3, padding=0, bias=False,weight_bit_width=None):
#         return qnn.QuantConv2d(in_channels = in_planes,
#                         out_channels = planes,
#                         kernel_size = kernel_size,
#                         padding = padding,
#                         bias = bias,
#                         weight_quant = SignedBinaryWeightPerTensorConst,
#                         weight_scaling_const=1.0,)

# def _make_qlinear_bnn(in_features, out_features, bias=True, weight_bit_width=None):
#         return qnn.QuantLinear(in_features=in_features,
#                            out_features=out_features,
#                            bias=bias,
#                            weight_quant = SignedBinaryWeightPerTensorConst,
#                            weight_scaling_const=1.0,
#                            )

# def _make_qconv2d_ter(in_planes, planes, kernel_size=3, padding=0, bias=False,weight_bit_width=None):
#         return qnn.QuantConv2d(in_channels = in_planes,
#                         out_channels = planes,
#                         kernel_size = kernel_size,
#                         padding = padding,
#                         bias = bias,
#                         weight_quant = SignedTernaryWeightPerTensorConst,
#                         weight_scaling_const=1.0,)

# def _make_qlinear_ter(in_features, out_features, bias=True, weight_bit_width=None):
#         return qnn.QuantLinear(in_features=in_features,
#                            out_features=out_features,
#                            bias=bias,
#                            weight_quant = SignedTernaryWeightPerTensorConst,
#                            weight_scaling_const=1.0,
#                            )

_make_qconv2d = _make_qconv2d_fx
_make_qlinear = _make_qlinear_fx 

class BasicBlockRN12(nn.Module):
    def __init__(self, in_planes, planes,batchn,wbits):
        super(BasicBlockRN12, self).__init__()
        self.conv1 = _make_qconv2d(in_planes, planes, kernel_size=3, padding=1, bias=False,weight_bit_width=wbits)
        if(batchn):
            self.bn1 = nn.BatchNorm2d(planes)
        else:
            self.bn1 = nn.GroupNorm(2,planes)        
        self.conv2 = _make_qconv2d(planes, planes, kernel_size=3, padding=1, bias=False,weight_bit_width=wbits)
        if(batchn):
            self.bn2 = nn.BatchNorm2d(planes)
        else:
            self.bn2 = nn.GroupNorm(2,planes)
        self.conv3 = _make_qconv2d(planes, planes, kernel_size=3, padding=1, bias=False,weight_bit_width=wbits)
        if(batchn):
            self.bn3 = nn.BatchNorm2d(planes)
        else:
            self.bn3 = nn.GroupNorm(2,planes)

        if(batchn):
            norm = nn.BatchNorm2d(planes)
        else:
            norm = nn.GroupNorm(2,planes)

        self.shortcut = nn.Sequential(
            _make_qconv2d(in_planes, planes, kernel_size=1, bias=False,weight_bit_width=wbits),norm)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope = 0.1)
        out = F.leaky_relu(self.bn2(self.conv2(out)), negative_slope = 0.1)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)

        return out
    
class QResNet12(nn.Module):
    def __init__(self, feature_maps, input_shape, num_classes, batchn=False,wbits=args.wbits):
        super(QResNet12, self).__init__()        

        if(wbits == 0):
            wbits = None
            
        layers = []

        layers.append(BasicBlockRN12(input_shape[0], feature_maps,batchn,wbits))
        layers.append(BasicBlockRN12(feature_maps, int(2.5 * feature_maps),batchn,wbits))
        layers.append(BasicBlockRN12(int(2.5 * feature_maps), 5 * feature_maps,batchn,wbits))
        layers.append(BasicBlockRN12(5 * feature_maps, 10 * feature_maps,batchn,wbits))        
      
        self.layers = nn.Sequential(*layers)
        self.linear = _make_qlinear(10 * feature_maps, num_classes,bias=True,weight_bit_width=wbits)
        #self.linear_rot = qnn.QuantLinear(10 * feature_maps, 4,bias=True,weight_bit_width=wbits)
        self.mp = nn.MaxPool2d((2,2))
       
        for m in self.modules():
            if isinstance(m, qnn.QuantConv2d):
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