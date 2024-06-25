import os

base_folder = "/data/eihw-gpu2/pechleba/ParaSpeChaD/NT"
group_folders = {
    0: "Kontrollgruppe",
    1: "SubklinischeGruppe",
    2: "PatientInnen"
}

with open("/data/eihw-gpu2/pechleba/ParaSpeChaD/metadata/Alle_Daten_bis_t2.csv", "r", encoding="utf-8-sig") as file:
    lines = file.readlines()

header = lines[0].strip().split(";")

subject_col = header.index("subject")
group_col = header.index("group")

for line in lines[1:]:
    data = line.strip().split(";")

    subject_id = data[subject_col]
    group_id = int(data[group_col])

    subject_folder = os.path.join(base_folder, group_folders.get(group_id, ""), subject_id)
    group_name = group_folders.get(group_id, "Unknown Group")

    # Check if the subject folder exists
    if not os.path.exists(subject_folder):
        print(f"Folder for subject {subject_id} in group {group_name} does not exist.")
        continue

    # List files in the subject folder
    files_in_folder = os.listdir(subject_folder)

    # Check if there are less than 2 files in the folder
    if len(files_in_folder) < 2:
        print(f"Folder for subject {subject_id} in group {group_name} contains {len(files_in_folder)} file(s):")
        for file_name in files_in_folder:
            print(file_name)

print("Folder and file checking completed.")
