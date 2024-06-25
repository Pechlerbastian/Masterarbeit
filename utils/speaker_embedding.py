import torchaudio
import pandas as pd

import os
from speechbrain.pretrained import SpeakerRecognition

# Initialize the ECAPA-TDNN model
model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="/data/eihw-gpu2/pechleba/pretrained_models/spkrec-ecapa-voxceleb")

# Directories where the audio files are stored
audio_dirs = [
    "/data/eihw-gpu2/pechleba/ParaSpeChaD/NT/PatientInnen",
    "/data/eihw-gpu2/pechleba/ParaSpeChaD/NT/SubklinischeGruppe",
    "/data/eihw-gpu2/pechleba/ParaSpeChaD/NT/Kontrollgruppe"
]

# File to save the speaker embeddings
embeddings_file = "/data/eihw-gpu2/pechleba/ParaSpeChaD/metadata/embeddings_fixed.csv"

# DataFrame to store the speaker embeddings
df = pd.DataFrame()

# Iterate over each directory in the audio directories
for audio_dir in audio_dirs:
    for dirpath, dirnames, filenames in os.walk(audio_dir):
        sp_included = ("NT_SP.wav" in filenames)
        for filename in filenames:
            if filename.endswith("SP.wav") or not sp_included: 
                audio_file = os.path.join(dirpath, filename)
                
                waveform, sample_rate = torchaudio.load(audio_file)
                
                # Extract speaker embedding
                embedding = model.encode_batch(waveform) 
                embedding_str = ', '.join(map(str, embedding.detach().numpy().flatten()))
                
                id = os.path.basename(dirpath)
                
                df = df.append({"subject": id, "embedding": embedding_str}, ignore_index=True)
                break
            elif filename.endswith("HQ.wav"):
                hq_file_name = filename
            
df.to_csv(embeddings_file, index=False)