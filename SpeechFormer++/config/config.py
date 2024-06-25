
from yacs.config import CfgNode as CN
_C = CN(new_allowed=True)
_C.train = CN(new_allowed=True)
_C.model = CN(new_allowed=True)
_C.dataset = CN(new_allowed=True)

_C.train.device = 'cuda'
_C.train.num_workers = 8

# Choose the type of the model
_C.model.type = 'SpeechFormer++'   # Transformer, SpeechFormer, SpeechFormer++/SpeechFormer_v2

# Total epochs for training
_C.train.EPOCH = 120
# The size of a mini-batch 
_C.train.batch_size = 32
# Initial learning rate
_C.train.lr = 0.0005
# Set a random seed for reproducition
_C.train.seed = 123
# Select the GPUs used
_C.train.device_id = '0'
# Whether to find a appropriate initial learning rate automatically
_C.train.find_init_lr = False
# Whether to save the best checkpoint
_C.train.save_best = True
# train adapters ADDED
_C.train.train_adapters = True
# load_path ADDED
_C.train.load_path = None
# early stopping ADDED
_C.train.early_stopping_epochs = 15
_C.train.early_stopping_improvement = 0.025
_C.train.stim_try_embedding = True

# Perform corss-validation, e.g. 5-fold CV in iemocap corpus
_C.train.folds = None    # modify in the main process
# Select a database to train a model
_C.dataset.database = 'feedback_experiment'   # iemocap, meld, pitt, daic_woz, feedback_experiment
# Select a kind of feature to train a model
_C.dataset.feature = 'hubert12_updated'    # spec, wav2vec, logmel, hubert12, hubert24, w2v12, w2v24
# Select a padding mode when preparing data a mini-batch 
_C.dataset.padmode = 'constant'   # constant, repeat


# Special mark for current process
_C.mark = None

