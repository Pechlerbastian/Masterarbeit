import os
import pandas as pd


def process_fold(fold_num):
    
    labels_path = "/data/eihw-gpu2/pechleba/ParaSpeChaD/FeedbackExperiment/FeedbackExperiment/labels_12.10.23.csv"
    input_base_folder = f"/data/eihw-gpu2/pechleba/ParaSpeChaD/EMA-folds/fold_mapping/{fold_num}"
    output_base_folder = f"/data/eihw-gpu2/pechleba/ParaSpeChaD/EMA-folds/reappraisal_folds_self_rated/{fold_num}"
    
    # Load the labels.csv file
    labels_df = pd.read_csv(labels_path)

    # Initialize lists to store missing subjects
    missing_subjects_in_labels = []
    missing_subjects_in_filtered = []

    # Initialize a dictionary to store subject counts per file
    subject_counts = {}

    # Get a set of all subjects from labels.csv
    all_labels_subjects = set(labels_df['SubjectID'])

    # Iterate over the filtered*.csv files in the input directory for fold 0
    for file_name in os.listdir(input_base_folder):
        if file_name.startswith("filtered") and file_name.endswith(".csv"):
            # Load the filtered*.csv file
            filtered_df = pd.read_csv(os.path.join(input_base_folder, file_name))

            # Count subjects in the current file
            subject_count = len(filtered_df['subject'])
            subject_counts[file_name] = subject_count

            # Check for missing subjects in labels.csv
            missing_subjects_fold_in_labels = set(filtered_df['subject']) - all_labels_subjects
            for missing_subject_in_labels in missing_subjects_fold_in_labels:
                missing_subject_info = filtered_df.loc[filtered_df['subject'] == missing_subject_in_labels, 'group'].iloc[0]
                missing_subjects_in_labels.append((missing_subject_in_labels, missing_subject_info))

            # Check for missing subjects in filtered*.csv
            missing_subjects_fold_in_filtered = all_labels_subjects - set(filtered_df['subject'])
            for missing_subject_in_filtered in missing_subjects_fold_in_filtered:
                if all("_" not in group for group in filtered_df['group']):
                    missing_subjects_in_filtered.append(missing_subject_in_filtered)

            # Merge the filtered data with labels based on 'subject' and 'SubjectID'
            merged_df = pd.merge(filtered_df, labels_df, left_on='subject', right_on='SubjectID')

            # Drop rows where 'VoiceRating' is None
            merged_df = merged_df.dropna(subset=['Depression'])
            merged_df = merged_df[merged_df['Depression'] != ""]

            # Drop the 'subject' and 'group' columns after the merge
            merged_df.drop(['SubjectID', 'group'], axis=1, inplace=True)

            # Save the reappraisal*.csv file in the output directory
            reappraisal_file_name = file_name.replace("filtered_", "")
            reappraisal_file_path = os.path.join(output_base_folder, reappraisal_file_name)
            merged_df.to_csv(reappraisal_file_path, index=False)

    # Output missing subjects in labels.csv but not in filtered*.csv
    if missing_subjects_in_labels and fold_num == 0:
        print(f"\nSubjects in \"filtered\"-csvs but not in labels.csv for fold {fold_num}:")
        for missing_subject_in_labels, missing_subject_info in missing_subjects_in_labels:
            print(f"Subject: {missing_subject_in_labels}, Group: {missing_subject_info}")

    # Output missing subjects in filtered*.csv but not in labels.csv
    if missing_subjects_in_filtered and fold_num == 0:
        print(f"\nSubjects in labels.csv but not in \"filtered\"-csvs for fold {fold_num}:")
        for missing_subject_in_filtered in missing_subjects_in_filtered:
            print(f"Subject: {missing_subject_in_filtered}")

    # Output subject counts per file
    print(f"Subject counts per file for fold {fold_num}:")
    for file_name, subject_count in subject_counts.items():
        print(f"File: {file_name}, Subject Count: {subject_count}")

    # Print total count of missing subjects in labels
    total_missing_subjects = len(missing_subjects_in_labels)
    print(f"\nTotal missing subjects for fold {fold_num} in labels: {total_missing_subjects}")

    # Print total count of missing subjects in "filtered"
    total_missing_subjects = len(missing_subjects_in_filtered)
    print(f"\nTotal missing subjects for fold {fold_num} in \"filtered\": {total_missing_subjects}\n\n")


if __name__ == "__main__":
    for fold_number in range(5):
        process_fold(fold_number)
