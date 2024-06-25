import os
import librosa
import csv
import numpy as np

def calculate_length_and_quantile(folder_path, keyword):
    lengths = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".wav") and keyword in file_name:
                file_path = os.path.join(root, file_name)
                audio, sr = librosa.load(file_path, sr=None)
                duration = librosa.get_duration(y=audio, sr=sr)
                lengths.append(duration)
                print(f"File: {file_path}, Duration: {duration:.2f} seconds")

    quantile_80 = np.percentile(lengths, 80)
    print(f"\n80% Quantile of File Durations: {quantile_80:.2f} seconds")
    return quantile_80

def write_to_csv(root_folders, quantiles, keywords):
    file_exists = os.path.isfile("/data/eihw-gpu2/pechleba/ParaSpeChaD/EMA/audio_info.csv")

    with open("/data/eihw-gpu2/pechleba/ParaSpeChaD/EMA/audio_info.csv", "a", newline="") as csvfile:
        fieldnames = ["Root Folder", "Quantile (80%)", "Keyword"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for root_folder, quantile, keyword in zip(root_folders, quantiles, keywords):
            writer.writerow({"Root Folder": root_folder, "Quantile (80%)": quantile, "Keyword": keyword})

def main():
    # Specify the root folders, keywords, and initialize quantiles list
    root_folders = [
        "/data/eihw-gpu2/pechleba/ParaSpeChaD/EMA/Kontrollgruppe",
        "/data/eihw-gpu2/pechleba/ParaSpeChaD/EMA/PatientInnen",
        "/data/eihw-gpu2/pechleba/ParaSpeChaD/EMA/Subklinische_Gruppe"
    ]
    keywords = ["Frage"] * len(root_folders)
    quantiles = []

    # Iterate through root folders and calculate lengths
    for folder_path, keyword in zip(root_folders, keywords):
        print(f"\nChecking files in folder: {folder_path}")
        quantile_80 = calculate_length_and_quantile(folder_path, keyword)
        quantiles.append(quantile_80)

    # Write information to CSV file
    write_to_csv(root_folders, quantiles, keywords)

if __name__ == "__main__":
    main()
