from collections import OrderedDict
from dataclasses import dataclass
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import get_activation

from adapter.adapters import linear_layer, SamplerOutput


class TaskHyperNet(nn.Module):
    """This module generates the task-embeddings from the initial feeded task embeddings."""

    def __init__(self, config):
        super(TaskHyperNet, self).__init__()
        self.task_hidden_dim = config.task_hidden_dim
        self.projected_task_embedding_dim = config.projected_task_embedding_dim
        self.task_embeding_generator = nn.Sequential(
            linear_layer(config.task_embedding_dim, self.task_hidden_dim),
            nn.ReLU(),
            linear_layer(self.task_hidden_dim, self.projected_task_embedding_dim),
        )

    def forward(self, task_embedding):
        task_embedding = task_embedding.view(-1)
        generated = self.task_embeding_generator(task_embedding).view(-1)
        return generated

class LayerNormHyperNet(nn.Module):
    """This module generates the weight and bias for the task conditioned layer norm."""

    def __init__(self, config):
        super(LayerNormHyperNet, self).__init__()
        self.task_embedding_dim = (
            config.projected_task_embedding_dim
            if config.train_task_embeddings
            else config.task_embedding_dim
        )
        self.weight_generator = linear_layer(self.task_embedding_dim, config.adapter_input_dim)
        self.bias_generator = linear_layer(self.task_embedding_dim, config.adapter_input_dim)

    def forward(self, input):
        return self.weight_generator(input), self.bias_generator(input)


class AdapterLayersHyperNet(nn.Module):
    """This module generates the weights for all the meta adapter layers
    given the task embeddings and layer id."""

    def __init__(self, config, input_dim, output_dim):
        super(AdapterLayersHyperNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_generator = nn.Sequential(
            linear_layer(
                config.projected_task_embedding_dim, self.input_dim * self.output_dim
            )
        )
        self.bias_generator = nn.Sequential(
            linear_layer(config.projected_task_embedding_dim, self.input_dim)
        )

    def forward(self, embeddings):
        weight = self.weight_generator(embeddings).view(self.input_dim, self.output_dim)
        bias = self.bias_generator(embeddings).view(-1)
        return SamplerOutput(weight=weight, bias=bias)
