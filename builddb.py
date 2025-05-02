import argparse
import os
import sqlite3

import scipy.signal
from scipy import signal
from scipy.signal import ShortTimeFFT
import librosa
import matplotlib.pyplot as plt
import librosa.display
from numpy import ndarray
from scipy.signal.windows import gaussian
import numpy as np

SONGS = []
WINDOW = (200, 100)

def read_wav(path: str) -> tuple[ndarray, int | float]:
    data, sample_rate = librosa.load(path, sr=None, mono=True)
    return data, sample_rate
'''
def compute_spectrogram(audio_path):
    data, sample_rate = librosa.load(audio_path, sr=None, mono=True)
    S = np.abs(librosa.stft(data, n_fft=2048, hop_length=512))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db, sample_rate
'''
def compute_spectrogram2(audio_channel, sample_rate):
    win = gaussian(50, std=12, sym=True)  # Gaussian window

    STF = ShortTimeFFT(win, hop=2, fs=sample_rate, mfft=800, scale_to='psd')
    Sxx = STF.spectrogram(audio_channel)  # Apply it to your signal
    return Sxx, STF

def spectrogram(samples, sample_rate):
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    plt.pcolormesh(times, frequencies, np.log(spectrogram))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    return frequencies, times, spectrogram

from scipy.ndimage import maximum_filter
from skimage.feature import peak_local_max
from scipy.signal import find_peaks
from skimage.feature import peak_local_max
import numpy as np

def find_constellation_peaks(Sxx, min_distance=35, amp_min=-40, max_peaks_per_frame=50):
    """
    Sxx: 2D spectrogram (in dB scale; rows=freqs, cols=times)
    min_distance: Minimum number of pixels separating peaks (in freq-time space)
    amp_min: Minimum dB threshold to filter out low-energy peaks
    max_peaks_per_frame: Optional, cap peaks per time frame for uniform coverage
    """

    # Find local maxima
    #logaritme de l'amplitud
    coordinates = peak_local_max(Sxx, min_distance=min_distance, threshold_abs=amp_min, exclude_border=False)

    # Optionally limit number of peaks per time bin
    if max_peaks_per_frame is not None:
        from collections import defaultdict
        frame_peaks = defaultdict(list)
        for y, x in coordinates:  # y=freq, x=time
            frame_peaks[x].append((x, y, Sxx[y, x]))

        # Keep only top-N peaks (by amplitude) per time frame
        capped_peaks = []
        for peaks in frame_peaks.values():
            peaks.sort(key=lambda tup: tup[2], reverse=True)  # sort by amplitude
            capped_peaks.extend(peaks[:max_peaks_per_frame])

        return capped_peaks  # List of (time, freq, amplitude)

    # Return list of (time, freq, amplitude)
    return [(x, y, Sxx[y, x]) for y, x in coordinates]


def plot_constellation_map(Sxx, peaks):
    """
    Sxx: 2D spectrogram (freq x time)
    peaks: Array of [freq, time] coordinates
    """
    plt.figure(figsize=(12, 6))
    plt.imshow(Sxx, origin='lower', aspect='auto', cmap='inferno')
    plt.colorbar(label='Amplitude (dB)')

    # Unpack peak coordinates
    peaks = np.array(peaks)  # convert list to array
    freqs, times = peaks[:, 1], peaks[:, 0]  # (y=freq, x=time)

    # Plot peaks
    plt.scatter(times, freqs, color='cyan', s=10, marker='x', label='Peaks')
    plt.title("Constellation Map")
    plt.xlabel("Time Bins")
    plt.ylabel("Frequency Bins")
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_constellationpoints( f, t, Sxx):
        # Apply the maximum filter to identify local maxima in the spectrogram.
        local_maxima = maximum_filter(Sxx, size=50) == Sxx
        #print(local_maxima)

        # Extract the coordinates of the local maxima
        maxima_coordinates = np.where(local_maxima == True)

        t_constellationvalues = t[np.array(maxima_coordinates[1])]
        f_constellationvalues = f[np.array(maxima_coordinates[0])]

        constellation_points = list(zip(t_constellationvalues, f_constellationvalues))

        return constellation_points

def hash_generation(constellation):

    pass

def build_database(input_folder: str, output: str):
    # List all .wav files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.wav'):
            song_path = os.path.join(input_folder, filename)
            data, sr = read_wav(song_path)
            f, t, sp = spectrogram(data, sr)
            #S_db, sr = compute_spectrogram(song_path)
            #show_spectrogram(S_db, sr)
            #Sxx, STF = compute_spectrogram2(data, sr)
            #show_custom_spectrogram(Sxx, STF)
            peaks = find_constellation_peaks(sp)
            peaks2 = get_constellationpoints(f, t, sp)
            plot_constellation_map(sp, peaks)
            plot_constellation_map(sp, peaks2)
            print(f"Reading {filename} - Data shape: {data.shape}")
            SONGS.append(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Builds an audio fingerprint database from a folder of songs.",
        epilog="Example: python builddb.py -i songs-folder -o database.db"
    )
    parser.add_argument('-i', '--input', required=True, help="Path to the folder containing songs.")
    parser.add_argument('-o', '--output', required=True, help="Path to the output SQLite database file.")

    args = parser.parse_args()

    build_database(args.input, args.output)