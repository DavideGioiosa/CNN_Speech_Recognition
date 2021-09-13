import librosa
import json
import os
from constant_variables import SAMPLES_TO_CONSIDER, DATASET_PATH, JSON_PATH


def prepare_dataset(dataset_path, json_path, n_mfcc = 13, hop_length=512, n_fft=2048):

    # data dictionary
    data = {
        "mappings": [], #["on", "off"...]
        "labels": [],   #[0,0,0,1,1...]
        "MFCCs": [],
        "files": [] # ["dataset/on/1.wav"]
    }

    # loop through all the sub-dirs

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # no root level
        if dirpath is not dataset_path:

            # save label (i.e., sub-folder name) in the mapping
            category = dirpath.split("\\")[-1]
            data["mappings"].append(category)

            # loop through files
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                #extract mfcc
                signal, sr = librosa.load(file_path)
                # audio is at least 1 sec
                if len(signal) >= SAMPLES_TO_CONSIDER:
                    #enforce 1 sec signal

                    signal = signal[:SAMPLES_TO_CONSIDER]

                    #mfcc
                    mfcc = librosa.feature.mfcc(signal, n_mfcc = n_mfcc,
                                                hop_length=hop_length, n_fft=n_fft)

                    #store in json
                    data["MFCCs"].append(mfcc.tolist())
                    data["labels"].append(i - 1)
                    data["files"].append(file_path)
                    print(f"Processing: {file_path}")

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    prepare_dataset(DATASET_PATH, JSON_PATH)
