import pandas as pd
import os

def adjust_and_remove_data():
    # Define the target id and the exclude ids
    target_id = 'CK1056'
    exclude_ids = ['EC0563', 'UR0961', 'LK0568', 'MJ0167', 'HH0656']

    # Define the original and new directories
    original_dir = "/data/eihw-gpu2/pechleba/ParaSpeChaD/EMA-folds/reappraisal_folds_self_rated/"
    new_dir = "/data/eihw-gpu2/pechleba/ParaSpeChaD/EMA-folds/reappraisal_folds_self_rated/adjusted/"

    # Iterate over the folds and the file types
    for fold in range(5):
        for file_type in ['dev', 'test', 'train']:
            # Define the original and new file paths
            original_file = os.path.join(original_dir, str(fold), f"{file_type}.csv")
            new_file = os.path.join(new_dir, str(fold), f"{file_type}.csv")

            # Read the csv file
            df = pd.read_csv(original_file)

            # Adjust the value of column selfRatingDepression if filename contains the target id
            if target_id in original_file:
                df['selfRatingDepression'] = 10 - df['selfRatingDepression']

            # Remove the rows if filename contains one of the exclude ids
            for exclude_id in exclude_ids:
            
                df = df[~df['filename'].str.contains(exclude_id)]

                
            # Create the new directory if it does not exist
            os.makedirs(os.path.dirname(new_file), exist_ok=True)

            # Save the new csv file
            df.to_csv(new_file, index=False)

if __name__ == "__main__":
    adjust_and_remove_data()