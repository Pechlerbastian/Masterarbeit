# Masterarbeit_Bastian_Pechler


There are seperate README.md for the subprojects SpeechFormer++ and HyperPersonalisation.
The SpeechFormer++ README contains a detailed explanation for using the SpeechFormer++ architecture, as well as configuring the adapter tuning.
Since the HyperPersonalisation Repository already contained a lot of information, the README was kept, and additional configuration added.

Speaker Embeddings can be created via ./utils/speaker_embedding.py,
Creating the metadata for SpeechFormer++ can be done with /util/create_metadata_speechformer.py.

The jupyter notbooks results [model and dataset name].ipynb were used for evaluation of the test data in the master's thesis.

The environments are contained in the subprojects. The HyperPersonalisation env can be used to run the programs in utils as well.

The SpeechFormer++ environment is present in ./SpeechFormer++/.env, the HyperPersonalisation environment is contained in the ./HyperPersonalisation/.nix folder.
