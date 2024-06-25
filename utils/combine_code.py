import pandas as pd
import os
import pyreadstat
import numpy as np

def merge_csv_files():
    csv1_path = '/data/eihw-gpu2/pechleba/ParaSpeChaD/metadata/Alle_Daten_bis_t2.sav'
    csv2_path = '/data/eihw-gpu2/pechleba/ParaSpeChaD/NT-features-fixed/praat/features.csv'
    output_path = '/data/eihw-gpu2/pechleba/ParaSpeChaD/metadata/merged_file_fixed.csv'
      # Read the metadata
    csv1, _ = pyreadstat.read_sav(csv1_path)

    # Read the neutral text csvs
    csv2 = pd.read_csv(csv2_path)

    # Filter rows where the filename contains "NT_SP.wav"
    csv2 = csv2[csv2['filename'].str.contains('NT_SP.wav', na=False)]

    # Extract the subject from the 'filename' column
    csv2['subject'] = csv2['filename'].apply(lambda x: x.split('/')[1] if '/' in x else x)

    # Drop the 'filename' column in voice analytice
    csv2 = csv2.drop(columns=['filename'])

    # Merge the two DataFrames using the 'subject' column
    merged_csv = pd.merge(csv1, csv2, left_on='Code', right_on='subject')

    # Convert all columns to numeric
    for col in merged_csv.columns:
      if col != 'Code':
        merged_csv[col] = pd.to_numeric(merged_csv[col], errors='coerce')
    # Replace all empty values with -1
    merged_csv = merged_csv.fillna(-1)
    merged_csv = merged_csv.replace('', -1)
    merged_csv = merged_csv.replace("", -1)
    merged_csv = merged_csv.replace(pd.NaT, -1)
    merged_csv = merged_csv.replace(np.nan, -1)

    # Replace all -1 values in numeric columns by average of the values which are not -1
    for col in merged_csv.select_dtypes(include='number').columns:
        if merged_csv[col].isnull().sum() == 0:  # if there are no NaN values after conversion
            avg = merged_csv.loc[merged_csv[col] != -1, col].mean()
            merged_csv[col] = merged_csv[col].replace(-1, avg)

    # merged_csv = merged_csv["t0_alter"]
    merged_csv.to_csv(output_path, index=False)

    print(f'Merged file saved as {output_path}')

if __name__ == "__main__":
    merge_csv_files()
