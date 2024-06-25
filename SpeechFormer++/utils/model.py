"""
Created on Tue Sep 28 16:53:37 CST 2021
@author: lab-chen.weidong
"""

import torch
from torch import distributed as dist

def load_model(model_type, device='cpu', train_adapters=False, adapter_config=None, checkpoint_path=None, stim_try_embedding=False, **kwargs):
    if model_type == 'Transformer':
        from model.transformer import build_vanilla_transformer
        model = build_vanilla_transformer(**kwargs)
    elif model_type == 'SpeechFormer':
        from model.speechformer import SpeechFormer
        model = SpeechFormer(**kwargs)
    elif model_type == 'SpeechFormer++' or model_type == 'SpeechFormer_v2':
        from model.speechformer_v2 import SpeechFormer_v2
        model = SpeechFormer_v2(adapter_config=adapter_config, train_adapters=train_adapters, stim_try_embedding=stim_try_embedding, **kwargs)
        if checkpoint_path is not None:
            
            pretrained_state_dict = torch.load(checkpoint_path) 
            model_dict = pretrained_state_dict['model']

            # Remove 'module.' from the keys, this is necessary because of wrapping of the model in distributed parallel
            model_dict = {k.replace('module.', ''): v for k, v in model_dict.items()}

            # Load the weights into the new model
            model.load_state_dict(model_dict, strict=False)
          
            for name, param in model.named_parameters():
                if name in model_dict and torch.all(torch.eq(param, model_dict[name].to(param.device))):
                    print(f'Loading succeeded for layer: {name}')
                else:
                    print(f'Loading did not succeed for layer: {name}')
            # model.load_pretrained_weights(checkpoint_path)

    else:
        raise KeyError(f'Unknown model type: {model_type}')

    if device == 'cuda':
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()], find_unused_parameters=True)
    return model


