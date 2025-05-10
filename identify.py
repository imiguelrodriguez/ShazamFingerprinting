import argparse
import sqlite3
from utils import read_wav, compute_spectrogram, find_constellation_peaks, hash_generation

MIN_HASHES_FOR_MATCH = 5

def match_hashes(sample_hashes, conn):
    cur = conn.cursor()
    offset_counts = {}
    for h, t_sample in sample_hashes:
        cur.execute("SELECT hash, offset, song_id FROM fingerprints WHERE hash=?", (h,))
        matches = cur.fetchall()
        for _, t_db, song_id in matches:
            delta = t_db - t_sample
            key = (song_id, delta)
            offset_counts[key] = offset_counts.get(key, 0) + 1
    return offset_counts


def identify_song(offset_counts, conn):
    if not offset_counts:
        return None, 0
    best_match = max(offset_counts.items(), key=lambda x: x[1])
    (song_id, delta), count = best_match
    if count < MIN_HASHES_FOR_MATCH:
        return None, count
    cur = conn.cursor()
    cur.execute("SELECT name FROM songs WHERE id=?", (song_id,))
    song_name = cur.fetchone()[0]
    return song_name, count


def identify_sample(database_path, sample_path):
    data, sr = read_wav(sample_path)
    _, _, Sxx = compute_spectrogram(data, sr)
    peaks = find_constellation_peaks(Sxx)
    hashes = hash_generation(peaks)
    conn = sqlite3.connect(database_path)
    offset_counts = match_hashes(hashes, conn)
    song_name, score = identify_song(offset_counts, conn)
    conn.close()
    return song_name, score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identifies a song from a fingerprint database using a sample.")
    parser.add_argument('-d', '--database', required=True, help="Path to the fingerprint database.")
    parser.add_argument('-i', '--input', required=True, help="Path to the sample WAV file.")
    args = parser.parse_args()

    print(f"Identifying {args.input}...")
    song, score = identify_sample(args.database, args.input)

    if song:
        print(f"✅ Match: {song} (score = {score})")
    else:
        print("❌ No match found.")
