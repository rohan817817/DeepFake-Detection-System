import librosa
import numpy as np

def extract_mfcc(audio_path, n_mfcc = 40):
    audio, sample_rate = librosa.load(audio_path, sr = None)

    mfcc = librosa.feature.mfcc(y = audio, sr = sample_rate, n_mfcc = n_mfcc)
    mfcc = np.mean(mfcc.T, axis = 0)
    return mfcc