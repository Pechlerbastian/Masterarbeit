  

## Usage

This project was built on the basis of the SpeechFormer++ architecture.
For basic information about this basis, you can read [here](https://github.com/HappyColor/SpeechFormer) and [here](https://github.com/HappyColor/SpeechFormer2).
The basic steps to use the model will be explained by this README, as the original documentation did not provide a detailed guide for configuration and required steps before using the models.

### Changes in the output

As the goal of this thesis was predicting depression scores via regression, the error function, metrics and classifier of the original model were adjusted to fit the task at hand. 
The code for classification was left in but is not guaranteed to workl with the adapter tuning process. Other configuration than classification will produce a value error for now.

  

### Adjustments

Modifying the dataset and model can be done with the following commands:

    -mo, --model.type: Set the model type. (currently only SpeechFormer++ is working with adapter training)
    
    -d, --dataset.database: Set the dataset database to use. (e.g. feedback_experiments, ema)
    
    -f, --dataset.feature: Set the dataset feature.
    
    -g, --train.device_id: Set the device ID for training.
    
    -m, --mark: Set the mark.
    
    -s, --train.seed: Set the seed for training.
    
    -folds, --train.folds: Set the folds for training.

  The qualifier variable must be set in train_model.py, when adapter-tuning. Else naming conflicts will occure with the baseline model.
    Reading it from file was not possible, adding a command could be possible.  

The model and dataset here are determining, which further configuration will be loaded from ./SpeechFormer/SpeechFormer2/config/model_config.json and thus which ./SpeechFormer/SpeechFormer2/config/train_{option}.json will be used
 (e.g.  ./SpeechFormer/SpeechFormer2/config/train_SpeechFormer_v2.json)

  

Alternatively can these configurations be set in config/config.py.

  

### model.json

Example configuration:


    {
    
      
      
    
	    "SpeechFormer_v2": {
		    
		    "num_layers": [2, 2, 4, 4], -> number layers in different encoder blocks
		    
		    "expand": [1, 1, 1],
		    
		    "num_heads": 8, -> number attention heads
		    
		    "dropout": 0.1,
		    
		    "attention_dropout": 0.1,
		    
		    "train_adapters": true, -> whether to adapter-tune (true) or baseline train (false)
		    
		      
		    
		    "early_stopping_epochs" : 15, -> number of min epochs for early stopping criteria
		    
		    "early_stopping_improvement": 0.025, -> minimal improvement to be surpassed for last few epochs
		    
		    "stim_try_embedding": true, -> stim try embedding was created for potential pattern in
		    
		    feedback experiment dataset
		    
		      
		    
		    "freeze_embeds": true, -> only needed for train_adapter=true
		    
		    "unfreeze_classifier_head": false, -> only needed for train_adapter=true
		    
		    "unfreeze_layer_norms": true, -> only needed for train_adapter=true
		    
		    "freeze_model": false, -> only needed for train_adapter=true
		    
		    "unfreeze_model": false -> only needed for train_adapter=true
		    
		      
		    
	    },
	    
      
    
    }

  
  

### train_SpeechFormer_v2.json

Example configuration:

    {...
    
	    "feedback_experiment":{
	    
	      
			    
			    "hubert12_updated": {
			    
			    "train.EPOCH": 60,
			    
			    "train.batch_size": 32,
			    
			    "train.lr": 0.0001
			    
			    },
	    
	    },
    
	    "ema":{
	    
			    "hubert12_updated": {
			    
			    "train.EPOCH": 60,
			    
			    "train.batch_size": 32,
			    
			    "train.lr": 0.0001
	    
	    },
    
    }
    
    ...
    
    }

Defines maximum train_epochs, batchsize for training and initial lerning rate.

  
  

### [dataset]_feature_config.json
- [dataset] will be replaced by the dataset.database (e.g. via commandline)
- Later, dataset.feature will be relevant as this determines the feature to use for training (e.g. hubert12_updated).
- Different features can be extracted by HuBERT (or other models), this is done with lmdb_kit.py (for explanation look below at section feature extraction)
  

    {
	    
	    "num_classes": 1, -> regression
	    
	      
	    
	    "evaluate": ["mse", "mae", "rmse"],
	    
	    "meta_csv_file": "./SpeechFormer++/metadata/metadata_", 
	    											-> part of path leading to metadata_[fold_number].csv
	    
	    this is needed to prepare test, dev and train folds via lmdb_kit.py
	    
	    the different .pkl files for the splits and folds
    
      
    
	    "hubert12_updated": { -> name of the dataset
		    
		    feature
		    
		    "feature_dim": 1024,
		    
		    "lmdb_root": "./SpeechFormer++/metadata/",
		    
		    "matdir": "/data/eihw-gpu2/pechleba/ParaSpeChaD/FeedbackExperiment/FeedbackExperiment/wav/",
		    
		    "matkey": "hubert",
		    
		    "length": 224,
		    
		    "adapter_config": "/data/eihw-gpu2/pechleba/ParaSpeChaD/metadata/adapter_config.json", -> Adapter configuration file
		    
		    "pad_value": 0,
		    
		    "frame": 0.025, -> frame size (for Frame Stage and HuBERT feature extractor)
		    
		    "hop": 0.02,
		    
		    "adapter_meta_csv_file": "/data/eihw-gpu2/pechleba/ParaSpeChaD/metadata/merged_file_fixed.csv" 
		    -> this path has to point on the metadata file for adapter-tuning
		    },
    
    }



### Preparing the Dataset for SpeechFormer++ 
Command:
` python ./SpeechFormer++//utils/lmdb_kit.py`
This uses the configurations in the __main__() methods

    opt = {
    
    'database': 'feedback_experiment',
    
    'feature': 'hubert12_updated',
    
    'lmdb_name': 'metadata_'+str(fold)+'.csv',
    
    'lmdb_root': './SpeechFormer++/metadata/',
    
    'commit_interval': 100,
    
    'state': 'train', # Valid when database is meld or daic_woz.
    
    'fold': str(fold)
    
    }

To create .pkl files from .csv files.
the CSV-files need to look like this:

    filename,label,state
    ...
    Feedback_Kontrollgruppe_AJ0139_Negative_stim=10_try=0.mat,0.0,train
    Feedback_Kontrollgruppe_AJ0139_Negative_stim=10_try=1.mat,0.0,train
    Feedback_Kontrollgruppe_AJ0139_Negative_stim=10_try=2.mat,0.0,train
    Feedback_Kontrollgruppe_AJ0139_Negative_stim=1_try=0.mat,1.0,train
    Feedback_Kontrollgruppe_AJ0139_Negative_stim=1_try=1.mat,1.0,train
    Feedback_Kontrollgruppe_AJ0139_Negative_stim=1_try=2.mat,1.0,train
    Feedback_Kontrollgruppe_AJ0139_Negative_stim=2_try=0.mat,1.0,train
    Feedback_Kontrollgruppe_AJ0139_Negative_stim=2_try=1.mat,1.0,train
    ...

The .mat files are extracted features by e.g. HuBERT, the model will load the filenames from opt.feature combined to the [dataset]_feature_config.json: "matdir"

 ### Feature Extraction
   

    python  ./SpeechFormer/SpeechFormer2/extract_feature/extract_hubert.py

This program uses the HuBERT model which has to be added to a path (e.g. "/data/eihw-gpu2/pechleba/masterarbeit/results/hubert_large_ll60k.pt") to extract the features from wav files. 
There are different predefined methods for different datasets (e.g. EMA and Feedback Experiments). 
The paths for the audio folder where the wav files are located and the extraction folder need to be adjusted in this extract_[extractor_model].py file.

    audio_folder =  "/data/eihw-gpu2/pechleba/ParaSpeChaD/FeedbackExperiment/FeedbackExperiment/wav/"
    
    hubert_L12 = audio_folder +  "hubert12_updated/"

Please note that the SpeechFormer++ model requires extracted features since it cannot operate directly on wav files. 
This step is required to be completed for adapter-tuning and basline model training.

Please also use the right hubert large trained model. Fine-Tuned versions are not working for this step, at least not with the current configuration, since their architecture is different.

### creating the metadata_[fold_nr].csv
The util folder (../util/create_metadata_speechformer.py) contains the necessary logic to convert the wav paths to .mat-paths, remove certain subject ids and change the rating to 10-depression_score for subjects.
This enables the dataset preparation described in "Preparing the Dataset for SpeechFormer++".

This folder contains all the evaluation files for different dataset-model combinations.

Please note that the HyperPersonalisation environment ahs to be used for ../utils/ programs!


### creating the Speaker Embeddings

This can be done by using the ..utils/speaker_embedding.py file.
Please note that the HyperPersonalisation environment ahs to be used for ../utils/ programs!



### Adapter-Tuning configuration
The adapter_config.json is only needed if '"train_adapters": true' in model.json.
The adapter configuration file is similar than the one for the HyperPersonalisation project. 
The SpeechFormer++ implementation has not the ability to convert metadata columns to a one-hot encoded form or to use additional metadata columns from file names.
"speaker_embedding_file" is enabeling that the Speaker Embedding metadata (created by ..utils/speaker_embedding.py) can be used for adapter-tuning.
"metadata_columns" ("t0_alter" - "t1_sek_27") are the metadata columns from questionnaires and demographic information.
"metadata_columns" ("meanF0" - "ASD(speakingtime / nsyll)") are the metadata columns extracted from a neutral text recording via Parselmouth.

The reduction factor is set to 64 compared to 32 in the case of HyperPersonalisation.

These are the confifuration columns:
	{
		"adapter_config_name": "meta-adapter",
		"task_embedding_dim": 64,
		"add_layer_norm_before_adapter": false,
		"add_layer_norm_after_adapter": true,
		"hidden_dim": 128,
		"reduction_factor": 64,
		"non_linearity": "swish",
		"projected_task_embedding_dim": 64,
		"conditional_layer_norm": true,
		"train_adapters_blocks": true,
		"unique_hyper_net": false,
		"efficient_unique_hyper_net": true,
		"unique_hyper_net_layer_norm": true,
		"num_labels": 1,
		"problem_type": "regression",
		"print_num_parameters": true,
		"speaker_embedding_file": "/data/eihw-gpu2/pechleba/ParaSpeChaD/metadata/embeddings_fixed.csv",

		"metadata_columns": [
			"t0_alter",
			"group",
			"Geschlecht",
			"t0_Schulabschluss",
			"t0_Berufsabschluss",
			"t0_Vollzeit",
			"t1_state_01",
			"t1_state_03",
			"t1_state_05",
			"t1_state_07",
			"t1_state_08",
			"t1_state_09",
			"t1_state_10",
			"t1_state_11",
			"t1_state_12",
			"t1_HAMD_01",
			"t1_HAMD_02",
			"t1_HAMD_03",
			"t1_HAMD_04",
			"t1_HAMD_05",
			"t1_HAMD_06",
			"t1_HAMD_07",
			"t1_HAMD_08",
			"t1_HAMD_09",
			"t1_HAMD_10",
			"t1_HAMD_11",
			"t1_HAMD_12",
			"t1_HAMD_13",
			"t1_HAMD_14",
			"t1_HAMD_15",
			"t1_HAMD_16",
			"t1_HAMD_17",
			"t1_HAMD_18",
			"t1_HAMD_19",
			"t1_HAMD_20",
			"t1_HAMD_21",
			"t1_HAMD_22",
			"t1_HAMD_23",
			"t1_HAMD_24",
			"t1_phq_1",
			"t1_phq_2",
			"t1_phq_3",
			"t1_phq_4",
			"t1_phq_5",
			"t1_phq_6",
			"t1_phq_7",
			"t1_phq_8",
			"t1_phq_9",
			"t0_bdi_01",
			"t0_bdi_02",
			"t0_bdi_03",
			"t0_bdi_04",
			"t0_bdi_05",
			"t0_bdi_06",
			"t0_bdi_07",
			"t0_bdi_08",
			"t0_bdi_09",
			"t0_bdi_10",
			"t0_bdi_11",
			"t0_bdi_12",
			"t0_bdi_13",
			"t0_bdi_14",
			"t0_bdi_15",
			"t0_bdi_16",
			"t0_bdi_17",
			"t0_bdi_18",
			"t0_bdi_19",
			"t0_bdi_20",
			"t0_bdi_21",
			"t1_tipi_1",
			"t1_tipi_2",
			"t1_tipi_3",
			"t1_tipi_4",
			"t1_tipi_5",
			"t1_tipi_6",
			"t1_tipi_7",
			"t1_tipi_8",
			"t1_tipi_9",
			"t1_tipi_10",
			"t1_sek_1",
			"t1_sek_2",
			"t1_sek_3",
			"t1_sek_4",
			"t1_sek_5",
			"t1_sek_6",
			"t1_sek_7",
			"t1_sek_8",
			"t1_sek_9",
			"t1_sek_10",
			"t1_sek_11",
			"t1_sek_12",
			"t1_sek_13",
			"t1_sek_14",
			"t1_sek_15",
			"t1_sek_16",
			"t1_sek_17",
			"t1_sek_18",
			"t1_sek_19",
			"t1_sek_20",
			"t1_sek_21",
			"t1_sek_22",
			"t1_sek_23",
			"t1_sek_24",
			"t1_sek_25",
			"t1_sek_26",
			"t1_sek_27",
			"meanF0",
			"stdevF0",
			"hnr",
			"localJitter",
			"localAbsoluteJitter",
			"rapJitter",
			"ppq5Jitter",
			"ddpJitter",
			"localShimmer",
			"localdbShimmer",
			"apq3Shimmer",
			"aqpq5Shimmer",
			"apq11Shimmer",
			"ddaShimmer",
			"f1_mean",
			"f2_mean",
			"f3_mean",
			"f4_mean",
			"f1_median",
			"f2_median",
			"f3_median",
			"f4_median",
			"nsyll",
			"npause",
			"dur(s)",
			"phonationtime(s)",
			"speechrate(nsyll / dur)",
			"articulation rate(nsyll / phonationtime)",
			"ASD(speakingtime / nsyll)"
		]
	}