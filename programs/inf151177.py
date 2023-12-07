import sys
import warnings
import glob
import numpy as np
import scipy.io.wavfile
from matplotlib import pylab as plt
import scipy.io.wavfile as wav


def train_all(wav_files):
    exp = []
    res = []
    for file in wav_files:
        exp.append(file[-5])
        res.append(find_gender(file))

    print(exp)
    print(res)
    correct = sum(r == e for r, e in zip(res, exp))
    n = len(wav_files)
    print("Accuracy: {0:.1f}%".format(100 * correct / float(n)))


def find_gender(file):
    # Parameters
    time_start = 1  # seconds
    time_end = 3  # seconds
    filter_stop_freq = 70  # Hz
    filter_pass_freq = 170  # Hz
    filter_order = 1001

    warnings.simplefilter("ignore", category=scipy.io.wavfile.WavFileWarning)
    fs, audio = wav.read(file)
    # If stereo, convert to mono
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    # High-pass filter
    nyquist_rate = fs / 2.
    desired = (0, 0, 1, 1)
    bands = (0, filter_stop_freq, filter_pass_freq, nyquist_rate)

    warnings.simplefilter("ignore", category=DeprecationWarning)
    filter_coefs = scipy.signal.firls(filter_order, bands, desired, nyq=nyquist_rate)

    # Apply high-pass filter
    filtered_audio = scipy.signal.filtfilt(filter_coefs, [1], audio)

    # Only analyze the audio between time_start and time_end
    time_seconds = np.arange(filtered_audio.size, dtype=float) / fs
    audio_to_analyze = filtered_audio[(time_seconds >= time_start) &
                                      (time_seconds <= time_end)]

    fundamental_frequency = freq_from_hps(audio_to_analyze, fs)
    print('Fundamental frequency is {} Hz'.format(fundamental_frequency))
    if 170 >= fundamental_frequency:
        gender = "M"
        print(gender)
        return gender
    elif 170 < fundamental_frequency:
        gender = "K"
        print(gender)
        return gender


def freq_from_hps(signal, fs):

    signal = np.asarray(signal) + 0.0

    N = len(signal)
    signal -= np.mean(signal)  # Remove DC offset

    # Compute Fourier transform of windowed signal
    windowed = signal * np.kaiser(N, 100)

    # Get spectrum
    X = np.log(abs(np.fft.rfft(windowed)))

    # Remove mean of spectrum
    X -= np.mean(X)

    # Downsample sum logs of spectra instead of multiplying
    hps = np.copy(X)
    for h in range(2, 9):
        dec = scipy.signal.decimate(X, h, zero_phase=True)
        hps[:len(dec)] += dec

    i_peak = np.argmax(hps[:len(dec)])          # Find the peak

    # Convert to equivalent frequency
    return fs * i_peak / N  # Hz


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Invalid number of arguments")
        print("Running on training data")

        files = glob.glob('../data/lab3_data/lab_3b/train/*.wav')
        train_all(files)
        print("To use on test data: python3 inf151177.py <path_to_wav_file>")
    else:
        find_gender(sys.argv[1])
