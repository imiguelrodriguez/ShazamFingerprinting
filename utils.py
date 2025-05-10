import os
import hashlib
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import maximum_filter
from skimage.feature import peak_local_max


def read_wav(path):
    data, sample_rate = librosa.load(path, sr=None, mono=True)
    return data, sample_rate


def compute_spectrogram(samples, sample_rate):
    frequencies, times, Sxx = signal.spectrogram(samples, sample_rate)
    return frequencies, times, Sxx


def find_constellation_peaks(Sxx, min_distance=35, amp_min=-40, max_peaks_per_frame=50):
    coordinates = peak_local_max(
        Sxx,
        min_distance=min_distance,
        threshold_abs=amp_min,
        exclude_border=False
    )
    if max_peaks_per_frame is not None:
        from collections import defaultdict
        frame_peaks = defaultdict(list)
        for y, x in coordinates:
            frame_peaks[x].append((x, y, Sxx[y, x]))
        capped_peaks = []
        for peaks in frame_peaks.values():
            peaks.sort(key=lambda tup: tup[2], reverse=True)
            capped_peaks.extend(peaks[:max_peaks_per_frame])
        return capped_peaks
    return [(x, y, Sxx[y, x]) for y, x in coordinates]


def get_constellationpoints(frequencies, times, Sxx):
    local_maxima = maximum_filter(Sxx, size=50) == Sxx
    maxima_coordinates = np.where(local_maxima == True)
    t_vals = times[np.array(maxima_coordinates[1])]
    f_vals = frequencies[np.array(maxima_coordinates[0])]
    return list(zip(t_vals, f_vals))


def hash_generation(peaks, fan_value=5, delta_t_max=200):
    hashes = []
    for i in range(len(peaks)):
        for j in range(1, fan_value + 1):
            if i + j < len(peaks):
                t1, f1 = peaks[i][0], peaks[i][1]
                t2, f2 = peaks[i + j][0], peaks[i + j][1]
                delta_t = t2 - t1
                if 0 <= delta_t <= delta_t_max:
                    hash_str = f"{f1}|{f2}|{delta_t}"
                    h = hashlib.sha1(hash_str.encode("utf-8")).hexdigest()[0:20]
                    hashes.append((h, int(t1)))
    return hashes


def save_spectrogram_image(Sxx, times, freqs, filename, folder):
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(times, freqs, np.log(Sxx + 1e-10), shading='auto', cmap='magma')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram')
    plt.colorbar(label='Log Amplitude')
    plt.tight_layout()
    plt.savefig(os.path.join(folder, filename))
    plt.close()


def save_constellation_image(Sxx, peaks, filename, folder):
    plt.figure(figsize=(12, 6))
    plt.imshow(Sxx, origin='lower', aspect='auto', cmap='inferno')
    plt.colorbar(label='Amplitude (dB)')
    peaks = np.array(peaks)
    if peaks.size > 0:
        freqs, times = peaks[:, 1], peaks[:, 0]
        plt.scatter(times, freqs, color='cyan', s=10, marker='x', label='Peaks')
    plt.title("Constellation Map")
    plt.xlabel("Time Bins")
    plt.ylabel("Frequency Bins")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, filename))
    plt.close()


def clear_output_folders(folders):
    for folder in folders:
        for file in os.listdir(folder):
            if file.endswith(".png"):
                os.remove(os.path.join(folder, file))
