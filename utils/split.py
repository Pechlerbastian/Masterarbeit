import os
import pandas as pd
import argparse


def filter_and_save_csv(input_directory='/data/eihw-gpu2/pechleba/ParaSpeChaD/EMA-folds',
                        output_directory='/data/eihw-gpu2/pechleba/ParaSpeChaD/EMA-folds/fold_mapping/'):
    # Loop through folders '0' to '4'
    for folder_name in ['0', '1', '2', '3', '4']:
        input_folder_path = os.path.join(input_directory, folder_name)

        # Check if the input folder exists
        if not os.path.exists(input_folder_path):
            print(f"Input folder {folder_name} does not exist.")
            continue

        # Loop through CSV files in the folder
        for csv_file in ['test.csv', 'train.csv', 'dev.csv']:
            input_csv_path = os.path.join(input_folder_path, csv_file)

            # Check if the input CSV file exists
            if not os.path.exists(input_csv_path):
                print(f"CSV file {csv_file} in folder {folder_name} does not exist.")
                continue

            # Read the CSV file into a DataFrame
            df = pd.read_csv(input_csv_path)

            # Select only the 'subject' and 'group' columns and drop duplicates
            df_filtered = df[['subject', 'group']].drop_duplicates()

            # Define the output directory for the filtered CSV
            output_folder_path = os.path.join(output_directory, folder_name)

            # Create the output directory if it doesn't exist
            os.makedirs(output_folder_path, exist_ok=True)

            # Define the output file path
            output_csv_path = os.path.join(output_folder_path, f'filtered_{csv_file}')

            # Save the filtered DataFrame to the output CSV file
            df_filtered.to_csv(output_csv_path, index=False)

            print(f"Filtered CSV saved to {output_csv_path}")


def main():

    filter_and_save_csv()


if __name__ == '__main__':
    main()

