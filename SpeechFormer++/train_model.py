#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 18:16:12 CST 2021
@author: lab-chen.weidong
"""

import os
import sys
print(sys.executable)
from torch import nn
from tqdm import tqdm
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
import csv
import json
import argparse
import pandas as pd
from numpy import mean
import utils
from model.speechformer_v2 import SpeechFormer_v2_Blocks
from adapter.adapters import MetaAdapterConfig, AdapterConfig
from config import Cfg, create_workshop, modify_config

class Engine():
    def __init__(self, cfg, local_rank: int, world_size: int, qualifier=""):
        self.cfg = cfg
        self.qualifier = qualifier
        self.local_rank = local_rank
        self.world_size = world_size
        self.ckpt_save_path = self.cfg.ckpt_save_path
        self.device = self.cfg.train.device
        self.EPOCH = self.cfg.train.EPOCH
        self.current_epoch = 0
        self.iteration = 0
        self.output_csv_path = "/data/eihw-gpu2/pechleba/results/"
        self.task = 'regression'
        self.missing_metadata = set()

        if self.cfg.train.find_init_lr:
            self.cfg.train.lr = 0.000001
            self.cfg.train.step_size = 1
            self.cfg.train.gamma = 1.05
            if self.local_rank == 0:
                self.writer = SummaryWriter(self.cfg.workshop)

        ### prepare model and train tools
        model_type = self.cfg.model.type
        model_type = 'SpeechFormer_v2' if model_type == 'SpeechFormer++' or model_type == 'SpeechFormer_v2_base' else model_type
        with open('/data/eihw-gpu1/pechleba/SpeechFormer/SpeechFormer2/config/model_config.json', 'r') as f1, open(
                f'/data/eihw-gpu1/pechleba/SpeechFormer/SpeechFormer2/config/{self.cfg.dataset.database}_feature_config.json',
                'r') as f2:
            model_json = json.load(f1)[model_type]
            feas_json = json.load(f2)
            data_json = feas_json[self.cfg.dataset.feature]

            model_json['num_classes'] = feas_json['num_classes']
            model_json['input_dim'] = (data_json['feature_dim'] // model_json['num_heads']) * model_json['num_heads']
            model_json['length'] = data_json['length']
            model_json['ffn_embed_dim'] = model_json['input_dim'] // 2
            model_json['hop'] = data_json['hop']
        self.model_json = model_json

        self.data_loader_feactory = utils.dataset_lmdb.DataloaderFactory(self.cfg)

        self.train_dataloader = self.data_loader_feactory.build(state='train', **data_json)
        self.dev_dataloader = self.data_loader_feactory.build(state='dev', **data_json)
        self.test_dataloader = self.data_loader_feactory.build(state='test', **data_json)

        # start init adapter config
        if cfg.train.train_adapters:
            
            with open(data_json['adapter_config']) as adapter_file:
                adapter_data = json.load(adapter_file)

            selected_columns = ["Code"] + adapter_data['metadata_columns']
            df_metadata = pd.read_csv(data_json["adapter_meta_csv_file"], usecols=selected_columns)
            embedding_data = {}
            if adapter_data["speaker_embedding_file"] != "":
                                
                df_embeddings = pd.read_csv(adapter_data["speaker_embedding_file"])
                df_embeddings.set_index('subject', inplace=True)
                df_embeddings['embedding'] = df_embeddings['embedding'].apply(lambda x: list(map(float, x.split(', '))))
                embedding_data = df_embeddings['embedding'].to_dict()
               
            df_metadata.set_index("Code", inplace=True)
            selected_metadata = df_metadata.to_dict(orient="index")
            if adapter_data["speaker_embedding_file"] != "":
                combined_data = {}
                new_list = []
                for key in selected_metadata:
                    new_list = list(selected_metadata[key].values())
                    new_list.extend(embedding_data[key])
                   
                    combined_data[key] = new_list
                metadata_len = len(new_list)
            else:
                combined_data = {key: list(selected_metadata[key].values()) for key in selected_metadata}
                metadata_len = len(selected_columns) - 1

            self.task_embeddings = combined_data
            # TODO add metadata_transposer and columns_from_datasamples if reapprasial

            adapter_config = MetaAdapterConfig

            adapter_config.adapter_input_dim = 1024
            # code is included in selected columns, but is no metadata for adapter
            adapter_config.metadata_dim = metadata_len
            print("metadata", selected_columns)

            adapter_config.tasks = (
                    self.train_dataloader.dataset.get_all_keys() + self.dev_dataloader.dataset.get_all_keys()
                    + self.test_dataloader.dataset.get_all_keys()
            )
            
            extra_adapter_params = (
                "task_embedding_dim",
                "add_layer_norm_before_adapter",
                "add_layer_norm_after_adapter",
                "reduction_factor",
                "hidden_dim",
                "non_linearity",
                "projected_task_embedding_dim",
                "task_hidden_dim",
                "conditional_layer_norm",
                "train_adapters_blocks",
                "unique_hyper_net",
                "unique_hyper_net_layer_norm",
                "efficient_unique_hyper_net",
            )
            for p in extra_adapter_params:
                if p in adapter_data and hasattr(adapter_config, p):
                    setattr(adapter_config, p, adapter_data[p])

            adapter_config.device = cfg.train.device_id
        else:
            adapter_config = None
        self.adapter_config = adapter_config
        #end adapter config init
        print("Model: "+str(model_json))
        print("adapter: "+str(self.adapter_config))
        self.stim_try_embedding = cfg.train.stim_try_embedding

        self.model = utils.model.load_model(model_type, device=self.device,
                                            adapter_config=self.adapter_config, 
                                            checkpoint_path=cfg.train.load_path, additional_classifier="_adapter",
                                            **model_json)

        lr = self.cfg.train.lr
        self.optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, self.model.parameters()),
                                         lr=lr, momentum=0.9)
        if self.task == 'regression':
            self.loss_func = torch.nn.MSELoss()
            self.calculate_score = utils.toolbox.calculate_score_regression
        else: 
            raise ValueError("Invalid task. Only 'regression' is currently supported.")
            self.calculate_score = utils.toolbox.calculate_score_classification
            self.loss_func = torch.nn.CrossEntropyLoss()

        if self.cfg.train.find_init_lr:
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.cfg.train.step_size,
                                                 gamma=self.cfg.train.gamma)
        else:
            self.scheduler = lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.EPOCH,
                                                            eta_min=lr / 100)
        ### freeze modules
        if cfg.train.train_adapters:
            print("!!!!!!!!!!!!!!!!!!!! freezing")
            self.freezing_params(model_json)

        ### prepare logger
        self.logger_train = utils.logger.create_logger(self.cfg.workshop,
                                                       name='train') if self.local_rank == 0 else None
        self.logger_test = utils.logger.create_logger(self.cfg.workshop, name='dev') if self.local_rank == 0 else None
        if self.logger_train is not None:
            self.logger_train.info(f'workshop: {self.cfg.workshop}')
            self.logger_train.info(f'seed: {self.cfg.train.seed}')
            self.logger_train.info(f'pid: {os.getpid()}')

        ### prepare meters
        data_type = torch.int64
        self.loss_meter = utils.avgmeter.AverageMeter(device='cpu')
        self.score_meter = utils.avgmeter.AverageMeter(device='cpu')
        self.predict_recoder = utils.recoder.TensorRecorder(device='cuda', dtype=data_type)
        self.label_recoder = utils.recoder.TensorRecorder(device='cuda', dtype=data_type)
        self.tag_recoder = utils.recoder.StrRecorder(device='cpu', dtype=str)
        self.filename_recoder = utils.recoder.StrRecorder(device='cpu', dtype=str)
        self.train_score_1, self.train_score_2, self.train_score_3, self.train_loss = [], [], [], []
        self.test_score_1, self.test_score_2, self.test_score_3, self.test_score_spearman, self.test_loss =[], [], [], [], []

    def freeze_params(self, model: nn.Module):
        """Set requires_grad=False for each of model.parameters()"""
        for par in model.parameters():
            par.requires_grad = False

    def freeze_embeds(self, model):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        model_type = model.config.model_type

        if model_type == "wav2vec2":
            model.freeze_feature_encoder()
        else:
            self.freeze_params(model.model.shared)
            for d in [model.model.encoder, model.model.decoder]:
                self.freeze_params(d.embed_positions)
                self.freeze_params(d.embed_tokens)

    def freezing_params(self, model_params):
        """
        Freezes the model parameters based on the given setting in the arguments.
        Args:
          model: the given model.
          training_args: defines the training arguments.
          model_args: defines the model arguments.
          adapter_args: defines the adapters arguments.
        """
        if self.cfg.train.train_adapters:
            self.freeze_params(self.model.module)

            if self.adapter_config.efficient_unique_hyper_net:
                for name, sub_module in self.model.module.named_modules():
                    if isinstance(sub_module, SpeechFormer_v2_Blocks):
                        sub_module.configure_efficient_hypernets()

        if model_params["freeze_model"]:
            self.freeze_params(self.model.module)

        if model_params["unfreeze_classifier_head"]:
            for param in self.model.module.classifier.parameters():
                print("param in classifier head unfrozen")
                param.requires_grad = True
        else:
            for param in self.model.module.classifier.parameters():
                param.requires_grad = False
        # freeze training of the additional layers needed for feedback experiments (needed for rising values caused by reappraisals)
        if self.stim_try_embedding:
            for param in self.model.module.stim_embedding.parameters():
                param.requires_grad = False
                print("stim embedding frozen")
            for param in self.model.module.try_embedding.parameters():
                param.requires_grad = False
                print("try embedding frozen")

        # Extractor and Projector are not there in speechformer2
        # This will freeze the 
        if model_params["freeze_embeds"]:
            self.model.module.freeze_encoder_blocks()

        # Unfreezes layer norms.
        if model_params["unfreeze_layer_norms"]:
            for name, sub_module in self.model.module.named_modules():
                if isinstance(sub_module, nn.LayerNorm):
                    print("layer norm unfrozen")
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True


        if model_params["unfreeze_model"]:
            for param in self.model.module.parameters():
                param.requires_grad = True

    def reset_meters(self):
        self.loss_meter.reset()
        self.score_meter.reset()

    def reset_recoders(self):
        self.predict_recoder.reset()
        self.label_recoder.reset()
        self.tag_recoder.reset()
        self.filename_recoder.reset()

    def gather_distributed_data(self, gather_data):
        if isinstance(gather_data, torch.Tensor):
            _output = [torch.zeros_like(gather_data) for _ in range(self.world_size)]
            dist.all_gather(_output, gather_data, async_op=False)
            output = torch.cat(_output)
        else:
            if gather_data[0] is not None:
                _output = [None for _ in range(self.world_size)]
                if hasattr(dist, 'all_gather_object'):
                    dist.all_gather_object(_output, gather_data)
                else:
                    utils.distributed.all_gather_object(_output, gather_data, self.world_size)
                output = []
                for lst in _output:
                    output.extend(lst)
            else:
                output = None
        return output

    def train_epoch(self):
        self.train_dataloader.set_epoch(self.current_epoch)
        if self.local_rank == 0:
            print(f'-------- {self.cfg.workshop} --------')
        discrip_str = f'Epoch-{self.current_epoch}/{self.EPOCH}'
        pbar_train = tqdm(self.train_dataloader, disable=self.local_rank != 0)
        pbar_train.set_description('Train' + discrip_str)
        self.reset_meters()
        self.reset_recoders()

        self.model.train()
        for data in pbar_train:
            self.iteration += 1
            if self.task == 'regression':
                vote_tag = data[2]
                # EMA
                if self.cfg.dataset.database == 'ema':
                    tasks = [s.split('_')[-7] for s in vote_tag]
                # Feedback Experiments
                if self.cfg.dataset.database == 'feedback_experiment':
                    tasks = [s.split('_')[-4] for s in vote_tag]
                                
                if not all(task == tasks[0] for task in tasks):
                    raise RuntimeError("Not all tasks in a batch are the same")

                x = torch.stack(data[0], dim=0).to(self.device)

                y = torch.tensor(data[1]).to(self.device)
                if self.cfg.train.train_adapters:
                    task_embeddings = torch.tensor(self.task_embeddings[tasks[0]], device="cuda")

                batch_size = y.shape[0]
                self.optimizer.zero_grad()
                stim=None
                try_=None
                if self.stim_try_embedding:
                    stripped_filenames = [s.split('.')[-2] for s in vote_tag]
                    split = [s.split('_') for s in stripped_filenames]
                  
                    stim = torch.tensor([[[int(s[-2].split('=')[1])]] for s in split], dtype=torch.long, device=self.device)
                    try_ = torch.tensor([[[int(s[-1].split('=')[1])]] for s in split], dtype=torch.long, device=self.device)
                        
                    
                if self.cfg.train.train_adapters:
                    out = self.model(x, task=tasks[0], task_embedding=task_embeddings, stim=stim, try_=try_)
                else:
                    out = self.model(x, stim=stim, try_=try_)

                loss = self.loss_func(out, y)
                loss.backward()
                self.optimizer.step()

                y_pred = out.detach()
                self.predict_recoder.record(y_pred)
                self.label_recoder.record(y)
                self.tag_recoder.record(tasks)
                
            else:
                x = torch.stack(data[0], dim=0).to(self.device)
                y = torch.tensor(data[1]).to(self.device)
                vote_tag = data[2]
                batch_size = y.shape[0]

                self.optimizer.zero_grad()
                if self.cfg.train.train_adapters:
                    out = self.model(x, task=vote_tag, task_embedding=self.task_embeddings[vote_tag])
                else:
                    out = self.model(x)

                loss = self.loss_func(out, y)
                loss.backward()
                self.optimizer.step()

                y_pred = torch.argmax(out, dim=1)

                self.predict_recoder.record(y_pred)
                self.label_recoder.record(y)
                self.tag_recoder.record(vote_tag)

            if self.task == 'regression':
            
                score = utils.toolbox.calculate_basic_score_regression(y_pred.cpu(), y.cpu())
            else:
                score = utils.toolbox.calculate_basic_score(y_pred.cpu(), y.cpu())
            self.loss_meter.update(loss.item())
            self.score_meter.update(score, batch_size)

            pbar_train_dic = OrderedDict()
            pbar_train_dic['iter'] = self.iteration
            pbar_train_dic['lr'] = self.optimizer.param_groups[0]['lr']
            pbar_train_dic['score'] = f'{self.score_meter.avg:.5f}'  # acc / MSE in local_rank: 0
            pbar_train_dic['loss'] = f'{self.loss_meter.avg:.5f}'  # loss in local_rank: 0
            pbar_train.set_postfix(pbar_train_dic)

            if self.cfg.train.find_init_lr:
                if loss.item() > 20:
                    raise ValueError(
                        f'Loss: {loss.item()} started to expand. Please use tensorboard to find the appropriate lr.')
                if self.local_rank == 0:
                    self.writer.add_scalar('Step Loss', loss.item(), self.iteration)
                    self.writer.add_scalar('Total Loss', self.loss_meter.avg, self.iteration)
                    self.writer.add_scalar('Step LR', self.optimizer.param_groups[0]['lr'], self.iteration)
                self.scheduler.step()

        epoch_preds = self.gather_distributed_data(self.predict_recoder.data).cpu()
        epoch_labels = self.gather_distributed_data(self.label_recoder.data).cpu()
        epoch_tag = self.gather_distributed_data(self.tag_recoder.data)

        if self.local_rank == 0:
            if self.task == 'regression':
                # rmse, mae (percentage), mse
                score_1, score_2, score_3, score_4, score_5 = self.calculate_score(epoch_preds, epoch_labels, epoch_tag)

                self.train_score_1.append(score_1)
                self.train_score_2.append(score_2)
                self.train_score_3.append(score_3)
                self.train_loss.append(self.loss_meter.avg)
            else:
                average_f1 = 'weighted' if self.cfg.dataset.database == 'meld' else 'macro'
                score_1, score_2, score_3, score_4 = self.calculate_score(epoch_preds, epoch_labels,
                                                                                   average_f1)

                self.train_score_1.append(score_1)
                self.train_score_2.append(score_2)
                self.train_score_3.append(score_3)
                self.train_loss.append(self.loss_meter.avg)

        if self.logger_train is not None:
            if self.task == 'regression':
                self.logger_train.info(
                    f'Training epoch: {self.current_epoch}, loss: {self.loss_meter.avg:.5f}, mae: {score_2:.5f}, rmse: {score_3:.5f}, overall_spearman: {score_5:.5f}'
                )
            else:
                self.logger_train.info(
                    f'Training epoch: {self.current_epoch}, accuracy: {score_1:.5f}, precision: {score_4:.5f}, recall: {score_2:.5f}, F1: {score_3:.5f}, loss: {self.loss_meter.avg:.5f}'
                )


    def validate(self):
        discrip_str = f'Epoch-{self.current_epoch}'
        pbar_test = tqdm(self.dev_dataloader, disable=self.local_rank != 0)
        pbar_test.set_description('Dev' + discrip_str)

        self.reset_meters()
        self.reset_recoders()

        self.model.eval()
        with torch.no_grad():
            for data in pbar_test:
                if self.task == 'regression':

                    vote_tag = data[2]
                    # EMA
                    if self.cfg.dataset.database == 'ema':
                        tasks = [s.split('_')[-7] for s in vote_tag]
                    # Feedback Experiments
                    if self.cfg.dataset.database == 'feedback_experiment':
                        tasks = [s.split('_')[-4] for s in vote_tag]
                        
                    if not all(task == tasks[0] for task in tasks):
                        raise RuntimeError("Not all tasks in a batch are the same")
 
                    x = torch.stack(data[0], dim=0).to(self.device)

                    y = torch.tensor(data[1]).to(self.device)
                    if self.cfg.train.train_adapters:
                        task_embeddings = torch.tensor(self.task_embeddings[tasks[0]], device="cuda")

                    batch_size = y.shape[0]
                    self.optimizer.zero_grad()
                    stim=None
                    try_=None
                    if self.stim_try_embedding:
                        stripped_filenames = [s.split('.')[-2] for s in vote_tag]
                        split = [s.split('_') for s in stripped_filenames]
                        stim = torch.tensor([[[int(s[-2].split('=')[1])]] for s in split], dtype=torch.long, device=self.device)
                        try_ = torch.tensor([[[int(s[-1].split('=')[1])]] for s in split], dtype=torch.long, device=self.device)
                       
                    if self.cfg.train.train_adapters:
                        out = self.model(x, task=tasks[0], task_embedding=task_embeddings, stim=stim, try_=try_)
                    else:
                        out = self.model(x, stim=stim, try_=try_)

                    loss = self.loss_func(out, y)
                    self.optimizer.step()

                    y_pred = out.detach()

                    self.predict_recoder.record(y_pred)
                    self.label_recoder.record(y)
                    self.tag_recoder.record(tasks)

                else:
                    x = torch.stack(data[0], dim=0).to(self.device)
                    y = torch.tensor(data[1]).to(self.device)
                    vote_tag = data[2]
                    batch_size = y.shape[0]

                    self.optimizer.zero_grad()
                    out = self.model(x)
                    loss = self.loss_func(out, y)
                    self.optimizer.step()

                    y_pred = torch.argmax(out, dim=1)

                    self.predict_recoder.record(y_pred)
                    self.label_recoder.record(y)
                    self.tag_recoder.record(tasks)

                if self.task == 'regression':
                    score = utils.toolbox.calculate_basic_score_regression(y_pred.cpu(), y.cpu())
                else:
                    score = utils.toolbox.calculate_basic_score(y_pred.cpu(), y.cpu())
                self.loss_meter.update(loss.item())
                self.score_meter.update(score, batch_size)

                pbar_test_dic = OrderedDict()
                pbar_test_dic['score'] = f'{self.score_meter.avg:.5f}'
                pbar_test_dic['loss'] = f'{self.loss_meter.avg:.7f}'
                pbar_test.set_postfix(pbar_test_dic)

            epoch_preds = self.gather_distributed_data(self.predict_recoder.data).cpu()
            epoch_labels = self.gather_distributed_data(self.label_recoder.data).cpu()
            tasks = self.gather_distributed_data(self.tag_recoder.data)

            if self.local_rank == 0:
                if self.task == 'regression':
                    score_1, score_2, score_3, score_4, score_5 = self.calculate_score(epoch_preds, epoch_labels, tasks)

                    self.test_score_1.append(score_1)
                    self.test_score_2.append(score_2)
                    self.test_score_3.append(score_3)
                    self.test_score_spearman.append(score_4)
                    print(score_4)
                    self.test_loss.append(self.loss_meter.avg)

                    if self.cfg.train.save_best:
                        if self.cfg.dataset.database in ['feedback_experiment', 'ema']:
                            is_best = max(self.test_score_spearman) == score_4  # average_spearman in r_4, if regression
                        else:
                            is_best = False
                        if is_best:
                             
                            if self.cfg.train.save_best:
                                self.model_save(is_best=True)
                else:
                    if hasattr(self.cfg.train, 'vote'):
                        if self.cfg.dataset.database == 'pitt':
                            modify_tag_func = utils.toolbox._majority_target_Pitt
                        elif self.cfg.dataset.database == 'daic_woz':
                            modify_tag_func = utils.toolbox._majority_target_DAIC_WOZ
                        else:
                            raise KeyError(f'Database: {self.cfg.dataset.database} do not need voting!')
                        _, epoch_preds, epoch_labels = utils.toolbox.majority_vote(epoch_tag, epoch_preds, epoch_labels,
                                                                                   modify_tag_func)

                    average_f1 = 'weighted' if self.cfg.dataset.database == 'meld' else 'macro'
                    # Calculate accuracy, recall, F1, precision, confuse_matrix
                    score_1, score_2, score_3, score_4, score_5 = self.calculate_score(epoch_preds, epoch_labels,
                                                                                       average_f1)
                    self.test_score_1.append(score_1)  # accuracy
                    self.test_score_2.append(score_2)  # recall
                    self.test_score_3.append(score_3)  # F1
                    self.test_loss.append(self.loss_meter.avg)

                    if self.cfg.train.save_best:
                        if self.cfg.dataset.database in ['iemocap', 'pitt']:
                            is_best = max(self.test_score_1) == score_1  # accuracy is more important
                        elif self.cfg.dataset.database in ['meld', 'daic_woz']:
                            is_best = max(self.test_score_3) == score_3  # F1 is more important
                        else:
                            is_best = False
                        if is_best:
                            self.model_save(is_best=True)

            if self.logger_test is not None:
                if self.task == 'regression':
                    self.logger_test.info(
                        f'Testing epoch: {self.current_epoch}, loss: {self.loss_meter.avg:.5f}, mae: {score_2:.5f}, rmse: {score_3:.5f}, average_spearman: {score_4:.5f}, overall_spearman: {score_5:.5f}'
                    )
                else:
                    self.logger_test.info(
                        f'Testing epoch: {self.current_epoch}, accuracy: {score_1:.5f}, precision: {score_4:.5f}, recall: {score_2:.5f}, F1: {score_3:.5f}, loss: {self.loss_meter.avg:.5f}, confuse_matrix: \n{score_5}'
                    )

    def check_early_stopping(self):
        # Check if early stopping should be applied
        if len(self.test_score_spearman) >= self.cfg.train.early_stopping_epochs:
            # Compare the current score with the best score in recent epochs (excluding the most recent one)
            current_score = self.test_score_spearman[-1]

            if all(current_score <= score + self.cfg.train.early_stopping_improvement 
                    for score in self.test_score_spearman[-self.cfg.train.early_stopping_epochs:-1]):                
                return True  # Early stopping condition met

        return False  # Continue training

    def test(self, fold=None, train_adapters=False):
        discrip_str = f'Epoch-{self.current_epoch}'
        pbar_test = tqdm(self.test_dataloader, disable=self.local_rank != 0)
        pbar_test.set_description('Test' + discrip_str)

        self.reset_meters()
        self.reset_recoders()

        self.model.eval()
        predictions = []
       
        preds = []
        labels = []
        ids = []

        with torch.no_grad():
            for data in pbar_test:
                if self.task == 'regression':
                    
                    vote_tag = data[2]
                    # EMA
                    if self.cfg.dataset.database == 'ema':
                        tasks = [s.split('_')[-7] for s in vote_tag]
                    # Feedback Experiments
                    if self.cfg.dataset.database == 'feedback_experiment':
                        tasks = [s.split('_')[-4] for s in vote_tag]
                    x = torch.stack(data[0], dim=0).to(self.device)

                    y = torch.tensor(data[1]).to(self.device)
                    if self.cfg.train.train_adapters:
                        task_embeddings = torch.tensor(self.task_embeddings[tasks[0]], device="cuda")

                    batch_size = y.shape[0]
                    self.optimizer.zero_grad()
                    stim=None
                    try_=None
                    if self.stim_try_embedding:
                        stripped_filenames = [s.split('.')[-2] for s in vote_tag]
                        split = [s.split('_') for s in stripped_filenames]
                        stim = torch.tensor([[[int(s[-2].split('=')[1])]] for s in split], dtype=torch.long, device=self.device)
                        try_ = torch.tensor([[[int(s[-1].split('=')[1])]] for s in split], dtype=torch.long, device=self.device)

                    if self.cfg.train.train_adapters:
                        out = self.model(x, task=tasks[0], task_embedding=task_embeddings, stim=stim, try_=try_)
                    else:
                        out = self.model(x, stim=stim, try_=try_)

                    loss = self.loss_func(out, y)
                    self.optimizer.step()

                    y_pred = out.detach()
                    label = y.cpu().numpy()
                    pred = list([y.item() for y in y_pred])
                    
                    self.predict_recoder.record(y_pred)
                    self.label_recoder.record(y)
                    self.tag_recoder.record(tasks)
                    self.filename_recoder.record(vote_tag)


                else:
                    x = torch.stack(data[0], dim=0).to(self.device)
                    y = torch.tensor(data[1]).to(self.device)
                    filename = data[2]

                    self.optimizer.zero_grad()
                    out = self.model(x)
                    loss = self.loss_func(out, y)
                    self.optimizer.step()

                    y_pred = torch.argmax(out, dim=1)

                    for filename, pred, true_value in zip(filename, y_pred, y.cpu().numpy()):
                        predictions.append({'filename': filename, 'prediction': pred, 'true': true_value})
            
            epoch_preds = self.gather_distributed_data(self.predict_recoder.data).cpu()
            epoch_labels = self.gather_distributed_data(self.label_recoder.data).cpu()
            epoch_tag = self.gather_distributed_data(self.tag_recoder.data)
            epoch_file_names = self.gather_distributed_data(self.filename_recoder.data)
            predictions = []
            for filename, p, l in zip(epoch_file_names, epoch_preds, epoch_labels):
                predictions.append({'filename': filename, 'prediction': p.item(), 'true': l.item()})
            



        # Save predictions to CSV
        if train_adapters:
            output_file = self.output_csv_path + "speechformer_adapter/" 
        else: 
            output_file = self.output_csv_path + "speechformer_base/" 

        os.makedirs(output_file, exist_ok=True)
        database = self.cfg.dataset.database
        batch = self.cfg.train.batch_size
        feature = self.cfg.dataset.feature
        lr = self.cfg.train.lr
        
        output_file = output_file + f'all_recordings_{database}'
        if self.cfg.train.early_stopping_epochs > 0:
            output_file = output_file + f'_e{self.EPOCH}_es{self.cfg.train.early_stopping_epochs}_'+self.qualifier+'/'
        else:
            output_file = output_file + f'_e{self.EPOCH}_full_train/'

        os.makedirs(output_file, exist_ok=True)
        output_file = output_file + str(fold) 
        os.makedirs(output_file, exist_ok=True)
        
        with open(output_file + "/predictions.test.csv", 'a', newline='') as csvfile:
            fieldnames = ['filename', 'prediction', 'true']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if csvfile.tell() == 0:
                writer.writeheader()

            writer.writerows(predictions)
        
        # save spearman coefficient per person for regression tasks
        print("prediction ids", ids)
        if self.task == 'regression': 
            spearman_per_ids, corresponding_ids = utils.toolbox.calculate_spearman_per_id(epoch_preds, epoch_labels, epoch_tag)
            
            with open(output_file + "/predictions.spearman.csv", 'a', newline='') as csvfile:
                fieldnames = ['id', 'spearman']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if csvfile.tell() == 0:
                    writer.writeheader()

                for spearman, corresponding_id in zip(spearman_per_ids, corresponding_ids):
                    writer.writerow({'id': corresponding_id, 'spearman': spearman})
            print(mean(spearman_per_ids))

    def model_save(self, is_best=False):
        if is_best:
            ckpt_save_file = os.path.join(self.ckpt_save_path, 'best_32.pt')
            print(ckpt_save_file)
        else:
            ckpt_save_file = os.path.join(self.ckpt_save_path, f'epoch{self.current_epoch}.pt')
            print(ckpt_save_file)

        save_dict = {
            'cfg': self.cfg,
            'epoch': self.current_epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        torch.save(save_dict, ckpt_save_file)

    def run(self, fold=None, train_adapters=False):
        if self.cfg.train.find_init_lr:
            while self.current_epoch < self.EPOCH:
                self.train_epoch()
                self.current_epoch += 1
        else:
            plot_sub_titles = ['WA-train', 'UA-train', 'F1-train', 'Loss-train', 'WA-test', 'UA-test', 'F1-test',
                               'Loss-test']
            plot_data_name = ['train_score_1', 'train_score_2', 'train_score_3', 'train_loss', 'test_score_1',
                              'test_score_2', 'test_score_3', 'test_loss']

            while self.current_epoch < self.EPOCH:
            # while self.current_epoch < 1:     # debug
                self.train_epoch()
                self.scheduler.step()
                self.validate()

                if self.local_rank == 0:
                    plot_data = [getattr(self, data_name) for data_name in plot_data_name]
                    utils.write_result.plot_process(plot_data, plot_sub_titles, self.cfg.workshop)
                	
                if self.check_early_stopping():
                    print(f"Early stopping at epoch {self.current_epoch}")
                    break

                self.current_epoch += 1

        model_type = self.cfg.model.type
        model_type = 'SpeechFormer_v2' if model_type == 'SpeechFormer++' or model_type == 'SpeechFormer_v2_base' else model_type
        # load best performing model to make final test
        load_path = os.path.join(self.ckpt_save_path, 'best_32.pt')
        print("db", self.cfg.dataset.database)
        print("lp", load_path)
        self.model = utils.model.load_model(model_type, device=self.device,
                                            adapter_config=self.adapter_config, 
                                            checkpoint_path=load_path, 
                                            **self.model_json)

        self.test(fold=fold, train_adapters=train_adapters)
        print("missing metadata: " + str(self.missing_metadata))
        utils.logger.close_logger(self.logger_train)
        utils.logger.close_logger(self.logger_test)


def main_worker(local_rank, cfg, world_size, dist_url):
    utils.environment.set_seed(cfg.train.seed + local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method=dist_url,
        world_size=world_size,
        rank=local_rank,
    )

    if cfg.dataset.database == 'iemocap':
        cfg.train.strategy = '5cv'
        folds = [1, 2, 3, 4, 5] if cfg.train.folds is None else cfg.train.folds
        folds = [folds] if not isinstance(folds, list) else folds
    elif cfg.dataset.database == 'meld':
        folds = [1]
    elif cfg.dataset.database == 'pitt':
        cfg.train.vote = True
        folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] if cfg.train.folds is None else cfg.train.folds
        folds = [folds] if not isinstance(folds, list) else folds
    elif cfg.dataset.database == 'daic_woz':
        cfg.train.vote = True
        folds = [1]
    elif cfg.dataset.database == 'feedback_experiment':
        folds = [0,1,2,3,4]
    elif cfg.dataset.database == 'ema':
        folds = [0,1,2,3,4]
    else:
        raise KeyError(f'Unknown database: {cfg.dataset.database}')

    # This qualifier enables adapter-tuning multiple models, 
    # otherwise naming conflict with baseline for the .
    # Since the qualifier value is needed for initializing the config (/config/__init__.py)
    # and for writing the results (utils/write_results.py) it is not possible to load it from the 
    # standard configuration files. 
    # If adapter-tunings should be generated for one baseline, the qualifier must be adjusted 
    # in the Code.  
    qualifier = "e"
    # Adjusting the config name here enables to create multiple baseline models. 
    config_name = '/data/eihw-gpu1/pechleba/SpeechFormer/SpeechFormer2/exp/'
    for f in folds:

        cfg_clone = cfg.clone()
        cfg_clone.train.current_fold = f
        create_workshop(cfg_clone, local_rank, config_name=config_name, qualifier=qualifier)
        engine = Engine(cfg_clone, local_rank, world_size, qualifier=qualifier)
        engine.run(fold=f, train_adapters=cfg.train.train_adapters)
        torch.cuda.empty_cache()

    if local_rank == 0:
        # if self.task == 'regression': TODO --> no self here, maybe pass in cfg
        if True:
            criterion = ["mse", "mae", "rmse"]
        else:
            criterion = ['accuracy', 'precision', 'recall', 'F1']
        evaluate = cfg.train.evaluate
        outfile = f'/data/eihw-gpu1/pechleba/SpeechFormer/SpeechFormer2/result/result_{cfg.model.type}.csv'
        utils.write_result.path_to_csv(os.path.dirname(cfg_clone.workshop), criterion, evaluate, csvfile=outfile,
                                       logname='dev.log')


def main(cfg):
    utils.environment.visible_gpus(cfg.train.device_id)
    utils.environment.set_seed(cfg.train.seed)

    free_port = utils.distributed.find_free_port()
    dist_url = f'tcp://127.0.0.1:{free_port}'
    world_size = torch.cuda.device_count()  # num_gpus
    print(f'world_size={world_size} Using dist_url={dist_url}')

    mp.spawn(fn=main_worker, args=(cfg, world_size, dist_url), nprocs=world_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mo", "--model.type", help="modify cfg.train.model.type", type=str)
    parser.add_argument("-d", "--dataset.database", help="modify cfg.dataset.database", type=str)
    parser.add_argument("-f", "--dataset.feature", help="modify cfg.dataset.feature", type=str)
    parser.add_argument("-g", "--train.device_id", help="modify cfg.train.device_id", type=str)
    parser.add_argument("-m", "--mark", help="modify cfg.mark", type=str)
    parser.add_argument("-s", "--train.seed", help="modify cfg.train.seed", type=int)
    # parser.add_argument("-save", "--train.save_best", help="modify cfg.train.save_best", action='store_true') # commented out, because is set to true in config.py
    parser.add_argument("-folds", "--train.folds", nargs='*', help="modify cfg.train.folds", type=int)
    args = parser.parse_args()

    modify_config(Cfg, args)
    main(Cfg)
