"""
Contains all relevant parts of https://github.com/rabeehk/hyperformer - originally spread out across multiple modules.
As we only make use of some parts, we collect them here. The parts are unchanged. 
"""
from collections import OrderedDict
from dataclasses import dataclass
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import get_activation


"""Adapter Configs"""
@dataclass
class AdapterConfig(object):
    """Implements the adapter configuration proposed by Houlsby et. al, 2019
    in https://arxiv.org/abs/1902.00751."""

    add_layer_norm_before_adapter: bool = False
    add_layer_norm_after_adapter: bool = True
    non_linearity: str = "swish"
    reduction_factor: int = 16
    weight_init_range = 1e-2
    # Whether to use conditional layer norms for adapters.
    conditional_layer_norm = False
    hidden_dim = 128
    # Whether to add adapter blocks, this is used in case we need
    # to tune only layer norms.
    train_adapters_blocks = True


class MetaAdapterConfig(AdapterConfig):
    """Implements Meta adapter in which a hyper-network generates the parameters of
    adapter layers. In this case we have a task embeddings which is feed to the
    hyper-network to allow it generate the weights for the adapter layers."""

    task_embedding_dim = 512
    metadata_dim = None
    hidden_dim = 128
    train_task_embeddings = False
    projected_task_embedding_dim = 64
    task_hidden_dim = 128
    parametric_task_embedding = False
    # If Specified, uses one hypernet to generates the adapters weights.
    unique_hyper_net = False
    unique_hyper_net_layer_norm = True
    # We consider only one hyper-net for all the blocks of transformer.
    efficient_unique_hyper_net = False
    task_mapping = None

ADAPTER_CONFIG_MAPPING = OrderedDict(
    [("adapter", AdapterConfig), ("meta-adapter", MetaAdapterConfig)]
)

"""end Adapter Configs"""


"""Dataclasses"""

@dataclass
class SamplerOutput:
    """Base class for the base and weights of each adapter."""

    weight: torch.FloatTensor = None
    bias: torch.FloatTensor = None


@dataclass
class LayerNormOutput:
    """Base class for the base and weights of the conditional
    layer norms."""

    weight: torch.FloatTensor = None
    bias: torch.FloatTensor = None


@dataclass
class AdapterOutput:
    """Base class for each adapter weights"""

    up: SamplerOutput = None
    down: SamplerOutput = None
    pre_norm: LayerNormOutput = None
    post_norm: LayerNormOutput = None


@dataclass
class AdapterBlockOutput:
    """
    Base class for adapter layer's outputs.
    """

    feed_forward: AdapterOutput = None
    self_attention: AdapterOutput = None


def init_linear_layer(linear_layer, std=1e-2):
    """Initializes the given linear module as explained in adapter paper."""
    nn.init.normal_(linear_layer.weight, std=std)
    nn.init.zeros_(linear_layer.bias)


def linear_layer(input_dim, output_dim, std=1e-2):
    """Generates a linear module and initializes it."""
    linear = nn.Linear(input_dim, output_dim)
    init_linear_layer(linear, std=std)
    return linear


"""end dataclasses"""

"""adapter"""

class MetaLayersAdapterController(nn.Module):
    """Implements Meta Adapter controller module, in which
    the adapter layers' weights are generated from a unique hyper-network."""

    def __init__(self, config):
        super().__init__()
        self.activation_type = config.non_linearity.lower()
        self.input_dim = config.adapter_input_dim
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter

    def apply_layer_norm(self, inputs, layer_norm_weights):
        """Applies layer norm to the inputs."""
        normalized =  torch.nn.functional.layer_norm(
            inputs,
            (self.input_dim,),
            weight=layer_norm_weights.weight,
            bias=layer_norm_weights.bias,
        )
        return normalized

    def call_adapter(self, inputs, adapter_weights):

        down = F.linear(
            inputs, weight=adapter_weights.down.weight, bias=adapter_weights.down.bias
        )
        
        middle = get_activation(self.activation_type)(down)

        output = F.linear(
            middle, weight=adapter_weights.up.weight, bias=adapter_weights.up.bias
        )
        return output


    def forward(self, inputs, adapter_weights):
        z = (
            self.apply_layer_norm(inputs, adapter_weights.pre_norm)
            if self.add_layer_norm_before_adapter
            else inputs
        )
        outputs = self.call_adapter(z, adapter_weights)
        if self.add_layer_norm_after_adapter:
            outputs = self.apply_layer_norm(outputs, adapter_weights.post_norm)
        outputs = outputs + inputs
        return outputs

"""end adapter"""