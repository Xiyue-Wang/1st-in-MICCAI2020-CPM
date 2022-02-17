import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet264']


def densenet121(**kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    return model


def densenet169(**kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    return model


def densenet201(**kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    return model


def densenet264(**kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 64, 48),
                     **kwargs)
    return model


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('denseblock{}'.format(ft_begin_index))
        ft_module_names.append('transition{}'.format(ft_begin_index))
    ft_module_names.append('norm5')
    ft_module_names.append('classifier')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm+1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu+1', nn.ReLU(inplace=True))
        self.add_module('conv+1', nn.Conv3d(num_input_features, bn_size * growth_rate,
                                            kernel_size=1, stride=1, bias=False))
        self.add_module('norm+2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu+2', nn.ReLU(inplace=True))
        self.add_module('conv+2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                            kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, sample_size, sample_duration, first,growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, last_fc=True):

        super(DenseNet, self).__init__()

        self.last_fc = last_fc

        self.sample_size = sample_size
        self.sample_duration = sample_duration

        # First convolution

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(first, num_init_features, kernel_size=7,
                                stride=(2, 2, 2), padding=(3, 3, 3), bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        # Linear layer
        # self.fc_1 = nn.Linear(num_features, 512)
        # self.fc_ce_1 = nn.Linear(num_features, 512)
        # self.relu = nn.ReLU()
        # self.relu_ce = nn.ReLU()
        # self.dropout = nn.Dropout(0.5)
        # self.dropout_ce = nn.Dropout(0.5)
        self.fc = nn.Linear(num_features, 1)
        self.fc_ce = nn.Linear(num_features, num_classes)
        #self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):

        features = self.features(x)

        out = F.relu(features, inplace=True)
        last_duration = math.ceil(self.sample_duration / 32)
        last_size1 = math.floor(self.sample_size / 32)
        last_size2 = math.floor(self.sample_size / 32)
        out = F.avg_pool3d(out, kernel_size=(last_duration,last_size2, last_size1)).view(features.size(0), -1)
        #if self.last_fc:
       # print("1",out.shape)

        #out = self.classifier(out)
        # x1 = self.fc_1(out)
        # x1 = self.relu(x1)
        # x1 = self.dropout(x1)
        x1 = self.fc(out)
        # x2=self.fc_ce(x)

        # x2 = self.fc_ce_1(out)
        # x2 = self.relu_ce(x2)
        # x2 = self.dropout_ce(x2)
        x2 = self.fc_ce(out)

        return x1,x2