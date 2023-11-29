import glob
import wave
from numpy import *
import pandas as pd
import librosa
from librosa import *
import matplotlib.pyplot as plt


def read_files(files):
    data = []
    for file in files:
        y, sr = librosa.load(file)
        # print(y)
        # print(sr)
        if file == '*K.wav':
            data.append(['K', y, sr])
        elif file == '*M.wav':
            data.append(['M', y, sr])

    data_df = pd.DataFrame(data, columns=['target', 'waveform', 'sampling_rate'])
    return data_df


def extract_features(data):
    for row in data:
        # Extracting pitch and magnitude from each dataframe record
        pitches, magnitudes = librosa.core.piptrack(y=row.y, sr=row.sr)
        mean_pitches = pitches.mean()
        # Extracting Mel-Frequency Cepstral Coefficients
        mfccs = librosa.feature.mfcc(y=row.y, sr=row.sr, n_mfcc=13)
        mfccs_flattened = mfccs.flatten()   # transform to 1-D vector

        # adding new features to dataframe
        row['pitches'] = pitches
        row['magnitues'] = magnitudes
        row['mean_pitches'] = mean_pitches
        row['mfccs'] = mfccs_flattened



if __name__ == '__main__':
    wav_files = glob.glob('../data/lab3_data/lab_3b/train/*.wav')
    print(wav_files)
    df = read_files(wav_files)


    ###################
    # TEST
    plt.figure(figsize=(20, 5))
    librosa.display.waveplot(df[0].waveform, sr=df[0].sampling_rate)
    plt.show()