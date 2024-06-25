
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import utils
import pickle
import multiprocessing as mp
from torch.utils.data.dataloader import default_collate
import re
import math
from collections import defaultdict
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
import random
from typing import TypeVar, Optional, List

T_co = TypeVar('T_co', covariant=True)


def identity(x):
    return x

def subject_collater(batch):
    """
    Collate function to load one subject per batch based on the subject ID in the 'name' field.
    """
    # Extract individual components from the batch
    x, y, names = zip(*batch)

    # Get the subject IDs from the names
    if self.corpus == "feedback_experiment":
        subject_ids = [name.split("_")[-4] for name in names]
    else:
        subject_ids = [name.split("_")[-7] for name in names]


    # Select a unique subject ID for this batch
    selected_subject_id = set(subject_ids)
    selected_subject_id = list(selected_subject_id)[0]
    # Find the indices corresponding to the selected subject ID
    
    selected_index = subject_ids.index(selected_subject_id)

    # Extract the corresponding samples
    x_batch = [x[selected_index]]
    y_batch = [y[selected_index]]
    names_batch = [names[selected_index]]

    # Apply the default collate function to the selected samples
    x_collated = default_collate(x_batch)
    y_collated = default_collate(y_batch)


    return x_collated, y_collated, names_batch


class DistributedDalaloaderWrapper():
    def __init__(self, dataloader: DataLoader, collate_fn):
        self.dataloader = dataloader
        self.collate_fn = collate_fn
    
    def _epoch_iterator(self, it):
        for batch in it:
            yield self.collate_fn(batch)

    def __iter__(self):
        it = iter(self.dataloader)
        return self._epoch_iterator(it)

    def __len__(self):
        return len(self.dataloader)

    @property
    def dataset(self):
        return self.dataloader.dataset

    def set_epoch(self, epoch: int):
        self.dataloader.batch_sampler.set_epoch(epoch)

def universal_collater(batch):
    if isinstance(batch[0], torch.Tensor):
        # When batch_size is None, batch is a single tensor (the entire dataset)
        return batch
    else:
        all_data = [[] for _ in range(len(batch[0]))]
        for one_batch in batch:
            for i, (data) in enumerate(one_batch):
                # debug
                all_data[i].append(data)
        return all_data

class LMDB_Dataset(Dataset):
    def __init__(self, corpus, lmdb_root, map_size, label_conveter, state, mode, length, feature_dim, pad_value, fold=0):
  

        if corpus == 'ema' or corpus == 'feedback_experiment': 
            lmdb_root = os.path.join(lmdb_root, str(fold))

        lmdb_path = lmdb_root if os.path.exists(os.path.join(lmdb_root, 'meta_info.pkl')) else os.path.join(lmdb_root, state)
        #print("path: "+ str(lmdb_path))
        self.meta_info = pickle.load(open(os.path.join(lmdb_path, 'meta_info.pkl'), "rb"))
        self.LMDBReader = utils.lmdb_kit.LMDBReader(lmdb_path, map_size * len(self.meta_info['key']) * 10)
        self.LMDBReader_km = None
        self.corpus = corpus

        if self.corpus == 'iemocap':
            self.load_name = False
            dict_list = {'name': self.meta_info['key'], 'label': self.meta_info['label'], 'shape': self.meta_info['shape']}
            self.meta_info['key'], self.meta_info['label'], self.meta_info['shape'] = utils.dataset_kit.iemocap_session_split(fold, dict_list, state)
        elif self.corpus == 'meld':
            self.load_name = False
        elif self.corpus == 'feedback_experiment':
            self.load_name = True
        elif self.corpus == 'ema':
            self.load_name = True
        elif self.corpus == 'pitt':
            self.load_name = True
            dict_list = {'name': self.meta_info['key'], 'label': self.meta_info['label'], 'shape': self.meta_info['shape']}
            self.meta_info['key'], self.meta_info['label'], self.meta_info['shape'] = utils.dataset_kit.pitt_speaker_independent_split_10fold(fold, dict_list, state)
        elif self.corpus == 'daic_woz':
            self.load_name = True
        else:
            raise ValueError(f'Got unknown database: {self.corpus}')

        self.need_pad = True
        self.conveter = label_conveter
        self.kit = utils.speech_kit.Speech_Kit(mode, length, feature_dim, pad_value)

    def load_a_sample(self, idx=0):
        label = self.meta_info['label'][idx]
        x = self.LMDBReader.search(key=self.meta_info['key'][idx])
        T, C = [int(s) for s in self.meta_info['shape'][idx].split('_')]
        x = x.reshape(T, C)
        y = torch.tensor(self.label_2_index(label))
        return x, y

    #added
    def get_all_keys(self):
        return self.meta_info['key']

    def load_wav_path(self, idx):
        name = self.load_sample_name(idx)
        if self.corpus == 'iemocap':
            session = int(re.search('\d+', name).group())
            ses = '_'.join(name.split('_')[:-1])
            wav_path = f'/148Dataset/data-chen.weidong/iemocap/Session{session}/sentences/wav/{ses}/{name}.wav'
        else:
            raise ValueError(f'Got unknown database: {self.corpus}')

        return wav_path

    def load_sample_name(self, idx):
        return self.meta_info['key'][idx]
    
    def label_2_index(self, label):
        if self.corpus == 'feedback_experiment' or self.corpus == "ema":
            return float(label) 
            # return float(label) / 10 
        index = self.conveter[label]
        return index

    def get_need_pad(self):
        return self.need_pad

    def set_need_pad(self, need_pad):
        self.need_pad = need_pad

    def get_load_name(self):
        return self.load_name

    def set_load_name(self, load_name):
        self.load_name = load_name

    def __len__(self):
        return len(self.meta_info['key'])

    def __getitem__(self, idx):
        x, y = self.load_a_sample(idx)
        x = self.kit.pad_input(x) if self.need_pad else x    # ndarray -> Tensor
        name = self.load_sample_name(idx) if self.load_name else None
        return x, y, name


class MultiTaskBatchSampler(Sampler[T_co]):
    """Defines a sampler to sample multiple datasets with temperature sampling
    in a distributed fashion."""

    def __init__(self, dataset, dataset_sizes: List[int], batch_size: int, temperature: float,
                 num_replicas: Optional[int] = None, rank: Optional[int] = None,
                 seed: int = 0, shuffle: bool = True) -> None:
        """Constructor for MultiTaskBatchSampler.
        Args:
            dataset_sizes: a list of integers, specifies the number of samples in
                each dataset.
            batch_size: integer, specifies the batch size.
            temperature: float, temperature used for temperature sampling. The larger
                the value, the datasets are sampled equally, and for value of 0, the datasets
                will be sampled according to their number of samples.
            num_replicas: integer, specifies the number of processes.
            rank: integer, specifies the rank of the current process/
            seed: integer, random seed.
            shuffle: bool, if set to true, the datasets will be shuffled in each epoch.
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.dataset_sizes = dataset_sizes

        self.dataset = dataset
        base_size = [dataset_size // self.num_replicas for dataset_size in
                            self.dataset_sizes]
        remainder = [dataset_size % self.num_replicas for dataset_size in
                            self.dataset_sizes]

        # Distribute the data across the ranks
        self.rank_dataset_sizes = [base_size[i] + (1 if i < remainder[i] else 0) for i in range(self.num_replicas)]
       # By default we drop the last elements if dataset is not divisible by the number of ranks.
        # self.rank_dataset_sizes = [dataset_size // self.num_replicas for dataset_size in self.dataset_sizes]
        self.dataset_offsets = torch.cumsum(torch.LongTensor([0] + dataset_sizes), 0)
        self.total_sizes = [(dataset_size // self.num_replicas) * self.num_replicas for dataset_size in
                            self.dataset_sizes]
        self.temperature = temperature
        self.seed = seed
        self.epoch = 0
        # self.num_batches_per_epoch = (np.sum(
        #     dataset_sizes) + self.batch_size -1) // self.batch_size // self.num_replicas

        self.num_batches_per_epoch = math.ceil((np.sum(
            dataset_sizes) + self.batch_size -1) / self.batch_size / self.num_replicas) + 2
        self.shuffle = shuffle

    def _get_indices(self):
        if self.shuffle:
            shuffled_subjects = torch.randperm(len(list(self.subjects.keys()))).tolist()
            # print("Shuffled subjects:", shuffled_subjects)
            # print(shuffled_subjects) 
            current_subject_id = shuffled_subjects[0]
            current_subject = list(self.subjects.keys())[current_subject_id]
            subject_indices = self.subjects[current_subject]
            random.shuffle(subject_indices)
        else:
            current_subject = list(self.subjects.keys())[0]
            subject_indices = self.subjects[current_subject]
        # Determine the number of samples to take for this subject
        num_samples = min(self.batch_size, len(subject_indices))

        # Take multiple samples for this subject
        batch_indices = [subject_indices.pop(0) for _ in range(num_samples)]
        if len(subject_indices) == 0:
            self.subjects.pop(current_subject, None)
        else:
            self.subjects[current_subject] = subject_indices
        return batch_indices

    def _get_subjects(self):
        subjects = defaultdict(list)
        for idx in range(len(self.dataset)):
            
            #For Feedback Experiments
            subject_id = self.dataset[idx][2].split("_")[-4]
            #For EMA
            # subject_id = self.dataset[idx][2].split("_")[-7]
            subjects[subject_id].append(idx)
        return subjects

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        self.subjects = self._get_subjects()

        indices = []
        for dataset_size in self.dataset_sizes:
            indices.append(list(range(dataset_size)))

        self.rank_indices = []
        for i in range(len(self.dataset_sizes)):
            self.rank_indices.append(indices[i][self.rank:self.total_sizes[i]:self.num_replicas])



        # To make the model consistent across different processes, since the
        # model is based on tasks, we need to make sure the same task is selected
        # across different processes.
        tasks_distribution: torch.Tensor = self.generate_tasks_distribution()

        batch_task_assignments = torch.multinomial(tasks_distribution,
                                                   self.num_batches_per_epoch, replacement=True, generator=generator)

        for batch_task in batch_task_assignments:
            if len(self.subjects) == 0:
                return 
            num_task_samples = self.rank_dataset_sizes[batch_task]
            indices = self._get_indices()
            rank = self.rank_indices[batch_task]
            results = (self.dataset_offsets[batch_task] + torch.tensor(rank)[indices]).tolist()
            yield results


    def __len__(self):
        return self.num_batches_per_epoch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def generate_tasks_distribution(self):
        """Given the dataset sizes computes the weights to sample each dataset
        according to the temperature sampling."""
        total_size = sum(self.dataset_sizes)
        weights = np.array([(size / total_size) ** (1.0 / self.temperature) for size in self.dataset_sizes])
        weights = weights / np.sum(weights)
        return torch.as_tensor(weights, dtype=torch.double)

class DataloaderFactory():
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, state, **kwargs):
        corpus = self.cfg.dataset.database
        mode = self.cfg.dataset.padmode
        fold = self.cfg.train.current_fold
        length = kwargs['length']
        feature_dim = kwargs['feature_dim']
        pad_value = kwargs['pad_value']
        lmdb_root = kwargs['lmdb_root']
        
        if corpus == 'daic_woz':
            state = 'dev' if state != 'train' else 'train'   # test refers to dev in daic_woz corpus
            
        map_size = length * feature_dim * 4
        if corpus == 'feedback_experiment' or corpus == 'ema':
            label_conveter = None
        else:
            label_conveter = utils.dataset_kit.get_label_conveter(corpus)
        dataset = LMDB_Dataset(corpus, lmdb_root, map_size, label_conveter, state, mode, length, feature_dim, pad_value, fold)
        
        collate_fn = universal_collater
        sampler = MultiTaskBatchSampler(dataset=dataset, dataset_sizes=[len(dataset)], batch_size=self.cfg.train.batch_size, temperature=10.0, shuffle=True)
        sampler_test = MultiTaskBatchSampler(dataset=dataset, dataset_sizes=[len(dataset)], batch_size=self.cfg.train.batch_size, temperature=10.0, shuffle=False)
        # test_sampler = DistributedSampler(dataset, shuffle=state == 'train')
        if state == 'test':
            dataloader = DataLoader(
                dataset=dataset, 
                # batch_size=None,                               
                drop_last=False,                                    
                num_workers=self.cfg.train.num_workers, 
                collate_fn=identity,
                batch_sampler=sampler_test,
                pin_memory=True,
                multiprocessing_context=mp.get_context('fork'), # quicker! Used with multi-process loading (num_workers > 0)
            )
        elif state != 'train':
            dataloader = DataLoader(
                dataset=dataset, 
                drop_last=False,                                   
                num_workers=self.cfg.train.num_workers, 
                collate_fn=identity,
                batch_sampler=sampler,
                pin_memory=True,
                multiprocessing_context=mp.get_context('fork'), # quicker! Used with multi-process loading (num_workers > 0)
            )
        else:
            dataloader = DataLoader(
                dataset=dataset, 
                num_workers=self.cfg.train.num_workers, 
                collate_fn=identity,
                batch_sampler=sampler, 
                pin_memory=True,
                multiprocessing_context=mp.get_context('fork'), # quicker! Used with multi-process loading (num_workers > 0)
            )

        return DistributedDalaloaderWrapper(dataloader, collate_fn)
