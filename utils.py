import os
import hashlib
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import maximum_filter
from skimage.feature import peak_local_max
from matplotlib.gridspec import GridSpec
from collections import defaultdict


def read_wav(path):
    data, sample_rate = librosa.load(path, sr=None, mono=True)
    return data, sample_rate


def compute_spectrogram(samples, sample_rate, samples_seg=1024, overlap=8):
    frequencies, times, Sxx = signal.spectrogram(samples, fs=sample_rate, window="hann", nperseg=samples_seg, noverlap=int(samples_seg//overlap))
    return frequencies, times, Sxx


def find_constellation_peaks(Sxx, window_size=50):
    local_maxima = maximum_filter(Sxx, size=window_size) == Sxx

    # Get the coordinates of the peaks
    peak_indices = np.argwhere(local_maxima)

    # Convert to list of tuples (freq_idx, time_idx)
    return [tuple(idx) for idx in peak_indices]

def hash_generation(peaks, fan_value=5, delta_t_max=200, delta_f_max=50):
    hashes = []
    for i in range(len(peaks)):
        t1, f1 = peaks[i][0], peaks[i][1]
        for j in range(1, fan_value + 1):
            if i + j < len(peaks):
                t2, f2 = peaks[i + j][0], peaks[i + j][1]
                delta_t = t2 - t1
                delta_f = abs(f2 - f1)
                if 0 <= delta_t <= delta_t_max and delta_f <= delta_f_max:
                    hash_str = f"{f1}|{f2}|{delta_t}"
                    h = hashlib.sha1(hash_str.encode("utf-8")).hexdigest()[0:20]
                    hashes.append((h, int(t1)))
    return hashes



def save_waveform_and_spectrogram(samples, sample_rate, Sxx, times, freqs, filename, folder):    
    if sample_rate <= 0 or len(samples) == 0 or Sxx.size == 0 or len(times) == 0 or len(freqs) == 0 or filename == "":
        print(f"⚠️ Skipped plotting due to invalid input: {filename}")
        return

    try:
        waveform_duration = times[-1]
        time_waveform = np.linspace(0, waveform_duration, len(samples))

        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(2, 1, height_ratios=[1, 2], hspace=0.05)

        # === Waveform ===
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(time_waveform, samples, color='blue', linewidth=0.5)
        ax1.set_xlim([0, waveform_duration])
        ax1.set_ylabel("Amplitude")
        ax1.set_xticks([])
        ax1.grid(True)

        # === Spectrogram ===
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        img = ax2.pcolormesh(times, freqs, Sxx_db, shading='auto', cmap='inferno')
        ax2.set_xlabel("Time (sec)")
        ax2.set_ylabel("Frequency (Hz)")
        ax2.set_ylim(0, 5000)

        cbar = fig.colorbar(img, ax=ax2, orientation="horizontal", pad=0.2)
        cbar.set_label("Log Amplitude (dB)")

        plt.savefig(os.path.join(folder, filename))
        plt.close()

    except Exception as e:
        print(f"Error while plotting {filename}: {e}")


def save_constellation_image(Sxx, peaks, filename, folder):
    plt.figure(figsize=(12, 6))

    # Plot the spectrogram in dB scale with better contrast
    plt.imshow(10 * np.log10(Sxx + 1e-10),  # Avoid log(0)
               origin='lower',
               aspect='auto',
               cmap='inferno')
    plt.colorbar(label='Power (dB)')


    peaks = np.array(peaks)
    freq_indices = peaks[:, 0]
    time_indices = peaks[:, 1]
    plt.scatter(time_indices, freq_indices, color='cyan', s=10, marker='x', label='Peaks')

    plt.title("Constellation Map (Spectrogram with Peaks)")
    plt.xlabel("Time Bins")
    plt.ylabel("Frequency Bins")

    plt.legend()
    plt.tight_layout()

    # Ensure output folder exists
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, filename))
    plt.close()


def clear_output_folders_and_db_files(folders):
    for folder in folders:
        for file in os.listdir(folder):
            if file.endswith(".png"):
                os.remove(os.path.join(folder, file))
        print(f"Cleared folder: {folder}")
    
    for file in os.listdir():
        if file.endswith('.db'):
            os.remove(file)
            print(f"Deleted database file: {file}")


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