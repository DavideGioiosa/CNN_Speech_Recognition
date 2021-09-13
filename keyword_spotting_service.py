import tensorflow.keras as keras
import numpy as np
import librosa
from constant_variables import SAMPLES_TO_CONSIDER, MODEL_PATH

# Singleton class
class _Keyword_Spotting_Service:
    model = None
    _mappings = [
        "down",
        "off",
        "on",
        "no",
        "yes",
        "stop",
        "up",
        "right",
        "left",
        "go"
    ]
    _instance = None

    def predict(self, file_path):
        # extract MFCCs
        MFCCs = self.preprocess(file_path) # (# segments, # coeff)
        # covert 2d to 4d -> (# samples, # segments, # coeff, 1)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        #make prediction
        prediction = self.model.predict(MFCCs)  # [ [0.1, 0.3 ... 0.2] ]
        predicted_index = np.argmax(prediction)
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword

    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):

        # load audio file
        signal, sr = librosa.load(file_path)

        # ensure consistency in the file length
        if len(signal) >= SAMPLES_TO_CONSIDER:
            signal = signal[:SAMPLES_TO_CONSIDER]

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        return MFCCs.T

def Keyword_Spotting_Service():
    # ensure that we have only 1 instance of KSS
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance


if __name__ == "__main__":
    kss = Keyword_Spotting_Service()
    keyword1 = kss.predict("test/down.wav")

    print(f"Predicted keyword 1 : {keyword1}")