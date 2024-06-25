
from .config import _C as Cfg
import os
import torch
import json
import shutil
from functools import reduce

def create_workshop(cfg, local_rank, config_name="", qualifier="", stim_try=False):
    modeltype = cfg.model.type
    database = cfg.dataset.database
    batch = cfg.train.batch_size
    feature = cfg.dataset.feature
    lr = cfg.train.lr
    epoch = cfg.train.EPOCH
    es = cfg.train.early_stopping_epochs
    
    world_size = torch.cuda.device_count()
    batch = batch * world_size
    config_name = config_name 
    # potential name for stim_try True
    #config_name = config_name + f'all_recordings_{modeltype}_{database}_try_stim/'
    config_name = config_name + f'all_recordings_{modeltype}_{database}_try_stim/'
    if cfg.train.find_init_lr:
        if cfg.mark is None:
            cfg.mark = 'find_init_lr'
        else:
            cfg.mark = 'find_init_lr_' + cfg.mark
        config_name = config_name + f'es{es}_e{epoch}_b{batch}_find_lr_{feature}'
    else:
        config_name = config_name + f'es{es}_e{epoch}_b{batch}_lr{lr}_{feature}'

    if cfg.mark is not None:
        config_name = config_name + '_{}'.format(cfg.mark)
    
    
    if cfg.train.train_adapters:
        load_path = config_name
        config_name = config_name + qualifier


    cfg.workshop = os.path.join(config_name, f'fold_{cfg.train.current_fold}')
    
    if cfg.train.train_adapters:
        load_path = os.path.join(load_path, f'fold_{cfg.train.current_fold}')
        cfg.train.load_path = os.path.join(load_path, 'checkpoint/best_32.pt')
        print(str(cfg.train.load_path))
    cfg.ckpt_save_path = os.path.join(cfg.workshop, 'checkpoint')
    
    if local_rank == 0:
        if os.path.exists(cfg.workshop):
            print(f'Workshop {cfg.workshop} already exists...')
            # Deleting the contents of the directory
            # shutil.rmtree(cfg.workshop)
        else:
            os.makedirs(cfg.workshop)
            os.makedirs(cfg.ckpt_save_path)

def modify_config(cfg, args):
    # merge from Namespace
    args = vars(args)  # Namespace -> Dict
    args = dict_2_list(args)
    print('Use arguments from command line:', args)
    cfg.merge_from_list(args)
    
    if cfg.model.type == 'SpeechFormer++':
        modeltype = 'SpeechFormer_v2'
    elif cfg.model.type == 'SpeechFormer_v2':
        cfg.model.type = 'SpeechFormer++'
        modeltype = 'SpeechFormer_v2'
    elif cfg.model.type == 'SpeechFormer_v2_base':
        cfg.model.type = 'SpeechFormer++'
        modeltype = 'SpeechFormer_v2'
    else:
        modeltype = cfg.model.type

    print(cfg)
    database = cfg.dataset.database
    feature = cfg.dataset.feature
    
    train_config = json.load(open(f'/data/eihw-gpu1/pechleba/SpeechFormer/SpeechFormer2/config/train_{modeltype}.json', 'r'))[database][feature]
    model_config = json.load(open('/data/eihw-gpu1/pechleba/SpeechFormer/SpeechFormer2/config/model_config.json', 'r'))[modeltype]

    # merge from train_config
    train_config = dict_2_list(train_config)
    print('Use arguments from train json file:', train_config)
    cfg.merge_from_list(train_config)
    
    train_adapters = False
    
    # modify cfg.mark
    if modeltype == 'Transformer':
        num_layers = model_config['num_layers']
        _mark = f'L{num_layers}'
    elif modeltype == 'SpeechFormer': 
        L = reduce(lambda x, y: str(x) + str(y), model_config['num_layers'])
        expa = reduce(lambda x, y: str(x) + str(y), model_config['expand'])
        _mark = f'L{L}_expa{expa}'
    elif modeltype == 'SpeechFormer++' or modeltype == 'SpeechFormer_v2': 
        L = reduce(lambda x, y: str(x) + str(y), model_config['num_layers'])
        expa = reduce(lambda x, y: str(x) + str(y), model_config['expand'])
        cfg.train.train_adapters = model_config['train_adapters']
        cfg.train.early_stopping_epochs = model_config['early_stopping_epochs']
        cfg.train.early_stopping_improvement = model_config['early_stopping_improvement']
        cfg.train.stim_try_embedding = model_config['stim_try_embedding']
        _mark = f'L{L}_expa{expa}'
    else:
        _mark = ''

    cfg.mark = _mark + '_' + cfg.mark if cfg.mark is not None else _mark
    print('Modified mark:', cfg.mark)

    # modify cfg.train.batch_size in the case of multi-GPUs training
    num_gpus = len(cfg.train.device_id.split(','))
    if num_gpus > 1:
        ddp_batch_size = round(cfg.train.batch_size / num_gpus)
        print(f'Modified batch size: {cfg.train.batch_size} -> {ddp_batch_size}.')
        cfg.train.batch_size = ddp_batch_size

    cfg.train.evaluate = json.load(open(f'/data/eihw-gpu1/pechleba/SpeechFormer/SpeechFormer2/config/{database}_feature_config.json', 'r'))['evaluate']

def dict_2_list(dict):
    lst = []
    for key, value in dict.items():
        if value is not None:
            lst.extend([key, value])
    return lst
