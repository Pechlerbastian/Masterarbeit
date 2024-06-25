import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from module.speechformer_v2_layer import SpeechFormer_v2_Encoder
from module.utils import create_PositionalEncoding, statistical_information
from module.utils import _no_grad_trunc_normal_
from model.speechformer import make_layers
from adapter.adapters import AdapterBlockOutput
from adapter.hypernets import TaskHyperNet, LayerNormHyperNet, AdapterLayersHyperNet
import math
import inspect

from adapter.adapters import LayerNormOutput, AdapterOutput


class MergeBlock(nn.Module):
    ''' Merge features between two phases.

        The number of tokens is decreased while the dimension of token is increased.
    '''

    def __init__(self, in_channels, merge_scale, num_wtok, expand=2):
        super().__init__()

        out_channels = in_channels * expand
        self.MS = merge_scale
        self.num_wtok = num_wtok
        self.pool = nn.AdaptiveAvgPool2d((1, in_channels))
        self.fc = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor, task=None, task_embedding=None):
        x_wtok, x_fea = x[:, :self.num_wtok], x[:, self.num_wtok:]

        B, T, C = x_fea.shape
        ms = T if self.MS == -1 else self.MS

        need_pad = T % ms
        if need_pad:
            pad = ms - need_pad
            x_fea = F.pad(x_fea, (0, 0, 0, pad), mode='constant', value=0)
            T += pad

        x_fea = x_fea.view(B, T // ms, ms, C)
        x_fea = self.pool(x_fea).squeeze(dim=-2)

        x = torch.cat((x_wtok, x_fea), dim=1)
        x = self.norm(self.fc(x))

        return x

class CustomSequential(nn.Sequential):
    def forward(self, input, task=None, task_embedding=None):
        for module in self:
            if isinstance(module, SpeechFormer_v2_Blocks):
                input = module(input, task=task, task_embedding=task_embedding)
            else:
                input = module(input)
        return input

class SpeechFormer_v2_Blocks(nn.Module):
    def __init__(self, num_layers, embed_dim, ffn_embed_dim=2304, local_size=0,
                 num_heads=8, dropout=0.1, attention_dropout=0.1, activation='relu',
                 use_position=False, num_wtok=0, train_adapters=False, adapter_config=None):
        super().__init__()
        self.position = create_PositionalEncoding(embed_dim) if use_position else None
        self.input_norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList(
            [SpeechFormer_v2_Encoder(embed_dim, ffn_embed_dim, local_size, num_heads, dropout,
                                     attention_dropout, activation, num_wtok=num_wtok,
                                     train_adapters=train_adapters, adapert_config=adapter_config
                                     ) for _ in range(num_layers)])
        self._reset_parameters()
        self.num_layers = num_layers
        self.train_adapters = train_adapters
        self.adapter_config = adapter_config
        if self.train_adapters:
            self.init_for_adapter_training()

    def init_for_adapter_training(self):
        self.layer_norm_epsilon = 1e-6
        self.task_embedding_dim = self.adapter_config.task_embedding_dim
        self.layer_id_embeddings = nn.Embedding(self.num_layers, self.task_embedding_dim)
        self.adapters_block_type = nn.Embedding(2, self.task_embedding_dim)
        self.adapter_config.task_embedding_dim = (self.task_embedding_dim * 2) + self.adapter_config.metadata_dim

        self.task_hypernet = TaskHyperNet(self.adapter_config)
        self.adapter_config.task_embedding_dim = self.task_embedding_dim
        self.unique_hyper_net_layer_norm = self.adapter_config.unique_hyper_net_layer_norm
        if self.unique_hyper_net_layer_norm:
            self.LayerNorm = nn.LayerNorm(
                self.adapter_config.projected_task_embedding_dim, eps=self.layer_norm_epsilon
            )
        self.input_dim = self.adapter_config.adapter_input_dim
        self.down_sample_size = self.input_dim // self.adapter_config.reduction_factor
        # Defines the adapters hyper-nets.
        self.up_sampler_hyper_net = AdapterLayersHyperNet(
            self.adapter_config, self.input_dim, self.down_sample_size
        )
        self.down_sampler_hyper_net = AdapterLayersHyperNet(
            self.adapter_config, self.down_sample_size, self.input_dim
        )
        # Defines the layer norms' hyper net.
        self.add_layer_norm_before_adapter = self.adapter_config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = self.adapter_config.add_layer_norm_after_adapter
        self.train_task_embeddings = self.adapter_config.train_task_embeddings
        self.adapter_config.train_task_embeddings = True
        if self.add_layer_norm_before_adapter:
            self.pre_layernorm_hypernet = LayerNormHyperNet(self.adapter_config)
        if self.add_layer_norm_after_adapter:
            self.post_layernorm_hypernet = LayerNormHyperNet(self.adapter_config)
        self.adapter_config.train_task_embeddings = self.train_task_embeddings

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def configure_efficient_hypernets(self):
        if self.train_adapters:

            for param_name, param in self.layer_id_embeddings.named_parameters():
                param.requires_grad = True
                print("1")

            for param_name, param in self.adapters_block_type.named_parameters():
                param.requires_grad = True
                print("2")

            for param_name, param in self.task_hypernet.named_parameters():
                param.requires_grad = True
                print("3")

            for param_name, param in self.LayerNorm.named_parameters():
                param.requires_grad = True
                print("4")

            for param_name, param in self.up_sampler_hyper_net.named_parameters():
                param.requires_grad = True
                print("5")

            for param_name, param in self.down_sampler_hyper_net.named_parameters():
                param.requires_grad = True
                print("6")

            if self.add_layer_norm_before_adapter:
                for param_name, param in self.pre_layernorm_hypernet.named_parameters():
                    print("7")
                    param.requires_grad = True

            if self.add_layer_norm_after_adapter:
                for param_name, param in self.post_layernorm_hypernet.named_parameters():
                    print("8")
                    param.requires_grad = True

    def forward(self, x, kmeans_mask=None, task=None, task_embedding=None, ):
        output = self.input_norm(x)
        
        outputs = []
        
        for i, layer in enumerate(self.layers):
            adapter_block = None
            if self.train_adapters:
                # adapter_block is containing the weights for feedforward and attention adapters
                adapter_block = self.adapter_forward(task_embedding, i)
            output = layer(output, self.position, kmeans_mask, block_adapters=adapter_block)
        return output
     

    def get_embedding(self, task_embedding, layer_id, block_type):
        """Concatenates the task embedding with the embedding for the layer id and
        returns the final joint embedding."""
        # TODO replace hardcoded device by a variable
        layer_id_tensor = torch.tensor([layer_id], dtype=torch.long, device="cuda:0")
        layer_embedding = self.layer_id_embeddings(layer_id_tensor)
        type_id_tensor = torch.tensor(
            [block_type], dtype=torch.long, device="cuda:0"
        )
        type_embedding = self.adapters_block_type(type_id_tensor)
        layer_embedding = layer_embedding.view(-1)
        type_embedding = type_embedding.view(-1)
        embeddings = torch.cat(
            [
                task_embedding.view(1, -1),
                layer_embedding.view(1, -1),
                type_embedding.view(1, -1),
            ],
            axis=1,
        )
        embeddings = self.task_hypernet(embeddings.view(-1))
        if self.unique_hyper_net_layer_norm:
            embeddings = self.LayerNorm(embeddings)
        return embeddings


    def adapter_forward(self, task_embedding, layer_id):
       
        feed_forward_embeddings = self.get_embedding(task_embedding, layer_id, 0)
        self_attention_embeddings = self.get_embedding(task_embedding, layer_id, 1)
        
        # Generates the adapters weights in feed-forward.
        feed_forward_down = self.down_sampler_hyper_net(feed_forward_embeddings)
        feed_forward_up = self.up_sampler_hyper_net(feed_forward_embeddings)
      
        # Generates the adapter weights in self-attention.
        self_attention_down = self.down_sampler_hyper_net(self_attention_embeddings)
        self_attention_up = self.up_sampler_hyper_net(self_attention_embeddings)

        feed_forward_output = AdapterOutput(up=feed_forward_up, down=feed_forward_down)
        self_attention_output = AdapterOutput(
            up=self_attention_up, down=self_attention_down
        )

        # Generates the weights and baises for pre and post layer norms.
        if self.add_layer_norm_before_adapter:
            weight, bias = self.pre_layernorm_hypernet(feed_forward_embeddings)
            feed_forward_output.pre_norm = LayerNormOutput(weight=weight, bias=bias)
            weight, bias = self.pre_layernorm_hypernet(self_attention_embeddings)
            self_attention_output.pre_norm = LayerNormOutput(weight=weight, bias=bias)

        if self.add_layer_norm_after_adapter:
            weight, bias = self.post_layernorm_hypernet(feed_forward_embeddings)
            feed_forward_output.post_norm = LayerNormOutput(weight=weight, bias=bias)
            weight, bias = self.post_layernorm_hypernet(self_attention_embeddings)
            self_attention_output.post_norm = LayerNormOutput(weight=weight, bias=bias)

        return AdapterBlockOutput(
            feed_forward=feed_forward_output, self_attention=self_attention_output
        )


def make_layers(Locals: list, Merge: list, expand: list, num_layers: list, Former_blocks, Merge_blocks,
                Former_args: dict, Merge_args: dict):
    layers = []
    last_merge = 1
    while len(expand) < len(Merge):
        expand = expand + [-1]

    for l, ms, exp, num in zip(Locals, Merge, expand, num_layers):
        _l = l // last_merge if l != -1 else -1
        _ms = ms // last_merge if ms != -1 else -1

        Former_args['num_layers'] = num
        Former_args['local_size'] = _l
        module1 = Former_blocks(**Former_args)
        layers += [module1]

        if Merge_blocks is not None:
            if _ms != -1:
                Merge_args['merge_scale'] = _ms
                Merge_args['expand'] = exp
                module2 = Merge_blocks(**Merge_args)
                layers += [module2]

                Merge_args['in_channels'] *= exp
                Former_args['embed_dim'] *= exp
                Former_args['ffn_embed_dim'] *= exp

            last_merge = ms

        if Former_args['use_position']:
            Former_args['use_position'] = False  # only the first layer use positional embedding.

    return CustomSequential(*layers)


class SpeechFormer_v2(nn.Module):
    def __init__(self, input_dim, ffn_embed_dim, num_layers, num_heads, hop, num_classes,
                 expand, dropout=0.1, attention_dropout=0.1, train_adapters=False, adapter_config=None,
                 stim_try_embedding=False, **kwargs):
        super().__init__()
        self.input_dim = input_dim // num_heads * num_heads
        Locals, Merge = statistical_information(hop)
        assert isinstance(num_layers, list)

        self.num_wtok = math.ceil(kwargs['length'] / Merge[-2])

        self.wtok = nn.Parameter(torch.empty(1, self.num_wtok, input_dim), requires_grad=True)
        _no_grad_trunc_normal_(self.wtok, std=0.02)

        Former_args = {'num_layers': None, 'embed_dim': self.input_dim, 'ffn_embed_dim': ffn_embed_dim,
                       'local_size': None,
                       'num_heads': num_heads, 'dropout': dropout, 'attention_dropout': attention_dropout,
                       'activation': 'relu',
                       'use_position': True, 'num_wtok': self.num_wtok, 'train_adapters': train_adapters,
                       'adapter_config': adapter_config}
        Merge_args = {'in_channels': self.input_dim, 'merge_scale': None, 'expand': None, 'num_wtok': self.num_wtok}

        self.layers = make_layers(Locals, Merge, expand, num_layers, SpeechFormer_v2_Blocks, MergeBlock, Former_args,
                                  Merge_args)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Define the embedding layer for 'stim' and 'try'
        self.stim_try_embedding = stim_try_embedding

        dim_expand = abs(reduce(lambda x, y: x * y, expand))
        classifier_dim = self.input_dim * dim_expand

        if stim_try_embedding:
            self.stim_embedding_dim = 4 
            self.try_embedding_dim = 2
            self.stim_embedding = nn.Embedding(num_embeddings=11, embedding_dim=self.stim_embedding_dim)  # 'stim' values range from 0 to 10
            self.try_embedding = nn.Embedding(num_embeddings=4, embedding_dim=self.try_embedding_dim)     # 'try' values range from 0 to 2
            classifier_dim = classifier_dim + self.stim_embedding_dim + self.try_embedding_dim
        
        print(stim_try_embedding)
        print("c_dim", classifier_dim)

        self.classifier = nn.Sequential(
            nn.Linear(classifier_dim, classifier_dim // 2),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(classifier_dim // 2, classifier_dim // 4),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(classifier_dim // 4, 1)

        )


    def forward(self, x, task=None, task_embedding=None, stim=None, try_=None):
        if self.input_dim != x.shape[-1]:
            x = x[:, :, :self.input_dim]
        wtok = self.wtok.expand(x.shape[0], -1, -1)
        x = torch.cat((wtok, x), dim=1)

        x = self.layers(x, task=task, task_embedding=task_embedding)
        x = self.avgpool(x.transpose(-1, -2)).squeeze(dim=-1)
        if stim is not None and try_ is not None:
            stim_embedded = self.stim_embedding(stim)
            try_embedded = self.try_embedding(try_)

            stim_embedded = stim_embedded.squeeze(dim=2).squeeze(dim=1)
            try_embedded = try_embedded.squeeze(dim=2).squeeze(dim=1)
            x = torch.cat((x, stim_embedded, try_embedded), dim=1)
        pred = self.classifier(x)
        return pred.view(-1)


    def freeze_encoder_blocks(self):
        # print("entered freeze_encoder_blocks")
        for encoder_block in self.layers:
            # # mergeblock parameters will be frozen here
            if isinstance(encoder_block, MergeBlock):
                print("freeze merge block")
                for param in encoder_block.pool.parameters():
                    param.requires_grad = False
                for param in encoder_block.fc.parameters():
                    param.requires_grad = False
                for param in encoder_block.norm.parameters():
                    param.requires_grad = False
            
            if isinstance(encoder_block, SpeechFormer_v2_Blocks):
            
                print("speechformer v2 blocks frozen")
                for encoder in encoder_block.layers:
                    if isinstance(encoder, SpeechFormer_v2_Encoder):
                        # Freeze all parameters in the encoder
                        self.freeze_module(encoder)
                    # Unfreeze the parameters in adapter_after_attention and adapter_after_feed_forward
                    if hasattr(encoder, 'adapter_after_attention') and encoder.adapter_after_attention is not None:
                        self.unfreeze_module(encoder.adapter_after_attention)
                        print('adapter_after_attention unfrozen')

                    if hasattr(encoder, 'adapter_after_feed_forward') and encoder.adapter_after_feed_forward is not None:
                        self.unfreeze_module(encoder.adapter_after_feed_forward)
                        print('adapter_after_feed_forward unfrozen')
    
    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True



    def load_pretrained_weights(self, pretrained_path):
        state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
        state_dict["model"] = {k.replace("module.", ""): v for k, v in state_dict["model"].items()}
        self.load_state_dict(state_dict["model"])
