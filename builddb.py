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


def build_database(input_folder: str, output: str):
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

            peaks = find_constellation_peaks(Sxx)

            save_constellation_image(Sxx, peaks, f"{song_id_base}_peaks.png", PEAKS_FOLDER)

            try:
                cur.execute("INSERT INTO songs (name) VALUES (?)", (filename,))
            except sqlite3.IntegrityError:
                print(f"Song {filename} already exists in the database. Skipping.")
                continue

            song_id = cur.lastrowid

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
    args = parser.parse_args()

    build_database(args.input, args.output)
