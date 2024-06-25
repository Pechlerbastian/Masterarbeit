
import pandas as pd
import os

# Function to process the CSVs and create the metadata CSV
def create_metadata(target_id, exclude_ids=None, category_of_recording=""):

    # Read the CSV files
    # base_path = "/data/eihw-gpu2/pechleba/ParaSpeChaD/EMA-folds/"
    # base_path = "/data/eihw-gpu2/pechleba/ParaSpeChaD/FeedbackExperiment/FeedbackExperiment/folds/"
    base_path = "/data/eihw-gpu2/pechleba/ParaSpeChaD/EMA-folds/reappraisal_folds_self_rated/"

    def modify_filename(filename):
        parts = filename.split('/')
        filename = '_'.join(parts[0:-1]) + '_' + parts[-1].replace('.wav', '.mat')
        return filename

    for fold in range(5):
        train = pd.read_csv(base_path + "/" + str(fold) + '/train.csv')
        dev = pd.read_csv(base_path + "/" + str(fold) + '/dev.csv')
        test = pd.read_csv(base_path + "/" + str(fold) + '/test.csv')

        train = train[train['filename'].str.contains(category_of_recording)]
        dev = dev[dev['filename'].str.contains(category_of_recording)]
        test = test[test['filename'].str.contains(category_of_recording)]


        # Exclude specified subject IDs
        for exclude_id in exclude_ids:
            exclude_filter = train['filename'].str.contains(exclude_id)
            train = train[~exclude_filter]

            exclude_filter = dev['filename'].str.contains(exclude_id)
            dev = dev[~exclude_filter]

            exclude_filter = test['filename'].str.contains(exclude_id)
            test = test[~exclude_filter]
        # Filter data for the target ID
        train_with_scores_to_invert = train[train['filename'].str.contains(target_id)]
        dev_with_scores_to_invert = dev[dev['filename'].str.contains(target_id)]
        test_with_scores_to_invert = test[test['filename'].str.contains(target_id)]

        # Change Depression value to 10 minus the current value
        # train_with_scores_to_invert['selfRatingDepression'] = 10 - train_with_scores_to_invert['selfRatingDepression']
        # dev_with_scores_to_invert['selfRatingDepression'] = 10 - dev_with_scores_to_invert['selfRatingDepression']
        # test_with_scores_to_invert['selfRatingDepression'] = 10 - test_with_scores_to_invert['selfRatingDepression']
        train_with_scores_to_invert['Depression'] = 10 - train_with_scores_to_invert['Depression']
        dev_with_scores_to_invert['Depression'] = 10 - dev_with_scores_to_invert['Depression']
        test_with_scores_to_invert['Depression'] = 10 - test_with_scores_to_invert['Depression']

        # Concatenate the modified data with the rest of the data
        train = pd.concat([train[train['filename'].str.contains(target_id) == False], train_with_scores_to_invert])
        dev = pd.concat([dev[dev['filename'].str.contains(target_id) == False], dev_with_scores_to_invert])
        test = pd.concat([test[test['filename'].str.contains(target_id) == False], test_with_scores_to_invert])

        # Modify the 'filename' column for each dataset after concatenation
        train['filename'] = train['filename'].apply(modify_filename)
        dev['filename'] = dev['filename'].apply(modify_filename)
        test['filename'] = test['filename'].apply(modify_filename)

        # Rename the 'selfRatingDepression' column to 'label'
        # train.rename(columns={'selfRatingDepression': 'label'}, inplace=True)
        # dev.rename(columns={'selfRatingDepression': 'label'}, inplace=True)
        # test.rename(columns={'selfRatingDepression': 'label'}, inplace=True)
        train.rename(columns={'Depression': 'label'}, inplace=True)
        dev.rename(columns={'Depression': 'label'}, inplace=True)
        test.rename(columns={'Depression': 'label'}, inplace=True)

        # Add 'state' column for each dataset
        train['state'] = 'train'
        dev['state'] = 'dev'
        test['state'] = 'test'

        # Concatenate all datasets
        combined = pd.concat([train[['filename', 'label', 'state']], dev[['filename', 'label', 'state']], test[['filename', 'label', 'state']]])

        # Write the new metadata to a CSV file
        # combined.to_csv('/data/eihw-gpu2/pechleba/masterarbeit/ema_metadata_answer/metadata_'+str(fold)+'.csv', index=False)
        combined.to_csv('/data/eihw-gpu1/pechleba/SpeechFormer/SpeechFormer2/metadata/metadata_'+str(fold)+'.csv', index=False)

# Call the function with the target ID and exclude_ids
create_metadata(target_id='CK1056', exclude_ids=['EC0563', 'UR0961', 'LK0568', 'MJ0167', 'HH0656'], category_of_recording="")
