import itertools
import math

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


class AttributeClassifierAblation(nn.Module):
    def __init__(self, dataset='Adult', accuracy_layers=(14, 64, 32, 1), fairness_layers=(32, 32, 32, 32),
                 fairness_layers_position=3, fairness_layer_mode="linear"):
        super(AttributeClassifierAblation, self).__init__()
        self.mode = fairness_layer_mode
        if fairness_layer_mode != "reg":
            assert(fairness_layers_position < len(accuracy_layers))
        assert(fairness_layers[0] == accuracy_layers[fairness_layers_position - 1])
        assert(fairness_layers[-1] == accuracy_layers[fairness_layers_position - 1])
        if dataset == 'Adult':
            assert(accuracy_layers[0] == 14 or accuracy_layers[0] == 101)

        elif dataset == 'Health':
            assert(accuracy_layers[0] == 125)

        elif dataset == 'Compass':
            assert(accuracy_layers[0] == 11)

        else:
            raise NotImplementedError()

        self.accuracy_layers_part_1 = nn.ModuleList()
        for i in range(fairness_layers_position - 1):
            self.accuracy_layers_part_1.append(nn.Linear(accuracy_layers[i], accuracy_layers[i + 1]))
        if fairness_layer_mode == 'reg':
            for i in range(len(fairness_layers) - 1):
                self.accuracy_layers_part_1.append(nn.Linear(fairness_layers[i], fairness_layers[i + 1]))
        self.accuracy_layers_part_2 = nn.ModuleList()
        for i in range(fairness_layers_position - 1, len(accuracy_layers) - 1):
            self.accuracy_layers_part_2.append(nn.Linear(accuracy_layers[i], accuracy_layers[i + 1]))

        self.fairness_layers = nn.ModuleList()
        if fairness_layer_mode == 'linear':
            for i in range(len(fairness_layers) - 1):
                self.fairness_layers.append(nn.Linear(fairness_layers[i], fairness_layers[i + 1]))
        elif fairness_layer_mode == 'reg':
            for i in range(len(fairness_layers) - 1):
                self.fairness_layers.append(nn.Linear(fairness_layers[i], fairness_layers[i + 1]))
            self.accuracy_layers_part_1 += self.fairness_layers + self.accuracy_layers_part_2
        else:
            raise NotImplementedError()

    def get_accuracy_parameters(self):
        return itertools.chain(self.accuracy_layers_part_1.parameters(), self.accuracy_layers_part_2.parameters())

    def get_fairness_parameters(self):
        return self.fairness_layers.parameters()

    def forward(self, x):
        if self.mode == 'reg':
            for idx, layer in enumerate(self.accuracy_layers_part_1):
                if idx != len(self.accuracy_layers_part_1) - 1:
                    x = F.relu(layer(x))
                else:
                    x = torch.sigmoid(layer(x))
            return x
        else:
            for layer in self.accuracy_layers_part_1:
                x = F.relu(layer(x))
                # x = F.dropout(x, 0.3)
            for layer in self.fairness_layers:
                if self.mode == 'linear':
                    # x = F.dropout(x, 0.1)
                    x = F.relu(layer(x))
                else:
                    x = layer(x)

            for idx, layer in enumerate(self.accuracy_layers_part_2):
                if idx != len(self.accuracy_layers_part_2) - 1:
                    # x = F.dropout(x, 0.3)
                    x = F.relu(layer(x))
                else:
                    # x = F.dropout(x, 0.3)
                    x = torch.sigmoid(layer(x))

            return x