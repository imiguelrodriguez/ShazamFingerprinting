import argparse
import os
import sqlite3
from utils import read_wav, compute_spectrogram, find_constellation_peaks, hash_generation, save_waveform_and_spectrogram, save_constellation_image, clear_output_folders_and_db_files, save_spectrogram_image

SPEC_FOLDER = "spectrograms"
PEAKS_FOLDER = os.path.join("constellations", "peaks")
MAXFILT_FOLDER = os.path.join("constellations", "maxfilter")

os.makedirs(SPEC_FOLDER, exist_ok=True)
os.makedirs(PEAKS_FOLDER, exist_ok=True)
os.makedirs(MAXFILT_FOLDER, exist_ok=True)


def build_database(input_folder: str, output: str, **kwargs):
    window_size = kwargs.get("window_size", None)
    delta_t_max = kwargs.get('delta_t_max', None)
    delta_f_max = kwargs.get('delta_f_max', None)

    clear_output_folders_and_db_files([SPEC_FOLDER, PEAKS_FOLDER, MAXFILT_FOLDER])

    conn = sqlite3.connect(output)
    cur = conn.cursor()

    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS songs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT
            )
        """)
    except sqlite3.OperationalError as e:
        print(f"Error creating songs table: {e}")
        conn.close()
        return
    
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fingerprints (
                hash TEXT,
                offset INTEGER,
                song_id INTEGER
            )
        """)
    except sqlite3.OperationalError as e:
        print(f"Error creating fingerprints table: {e}")
        conn.close()
        return

    for filename in sorted(os.listdir(input_folder)):
        if filename.lower().endswith('.wav'):
            print(f"Processing {filename}")
            filepath = os.path.join(input_folder, filename)
            data, sr = read_wav(filepath)
            freqs, times, Sxx = compute_spectrogram(data, sr)
            song_id_base = os.path.splitext(filename)[0]

            # There is a corrupted file in the dataset, so we only process the first one
            if filename == sorted(os.listdir(input_folder))[0]:
                save_waveform_and_spectrogram(data, sr, Sxx, times, freqs, f"{song_id_base}_spectrogram.png", SPEC_FOLDER)
            else:
                save_spectrogram_image(Sxx, times, freqs, f"{song_id_base}_spectrogram.png", SPEC_FOLDER)

            if window_size:
                peaks = find_constellation_peaks(Sxx, times, freqs, window_size=int(window_size))
            else:
                peaks = find_constellation_peaks(Sxx, times, freqs)

            save_constellation_image(Sxx, peaks, f"{song_id_base}_peaks.png", PEAKS_FOLDER, freqs, times)

            try:
                cur.execute("INSERT INTO songs (name) VALUES (?)", (filename,))
            except sqlite3.IntegrityError:
                print(f"Song {filename} already exists in the database. Skipping.")
                continue

            song_id = cur.lastrowid

            if delta_t_max and delta_f_max:
                hashes = hash_generation(peaks, delta_t_max=int(delta_t_max), delta_f_max=int(delta_f_max))
            elif delta_t_max is not None:
                hashes = hash_generation(peaks, delta_t_max=int(delta_t_max))
            elif delta_f_max is not None:
                hashes = hash_generation(peaks, delta_f_max=int(delta_f_max))
            else:
                hashes = hash_generation(peaks)

            for hash_val, offset in hashes:

                try:
                    cur.execute(
                        "INSERT INTO fingerprints (hash, offset, song_id) VALUES (?, ?, ?)",
                        (hash_val, offset, song_id)
                    )
                except sqlite3.IntegrityError:
                    print(f"Hash {hash_val} already exists for song {filename}. Skipping.")
                    continue

            conn.commit()

    conn.close()
    print("Database build complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Builds an audio fingerprint database from a folder of songs.",
        epilog="Example: python builddb.py -i songs-folder -o database.db"
    )
    parser.add_argument('-i', '--input', required=True, help="Path to the folder containing songs.")
    parser.add_argument('-o', '--output', required=True, help="Path to the output SQLite database file.")
    parser.add_argument('-w', '--windowsize', required=False, help="Size of the window of the maximum filter to obtain constellation peaks.")
    parser.add_argument('-dt', '--deltatime', required=False, help="Time offset to build rectangle in the hashing step.")
    parser.add_argument('-df', '--deltafreq', required=False, help="Frequency offset to build rectangle in the hashing step.")


    args = parser.parse_args()

    # Prepare kwargs depending on which optional args are provided
    kwargs = {}
    if args.windowsize is not None:
        kwargs['window_size'] = args.windowsize
    if args.deltatime is not None:
        kwargs['delta_t_max'] = args.deltatime
    if args.deltafreq is not None:
        kwargs['delta_f_max'] = args.deltafreq

    build_database(args.input, args.output, **kwargs)
