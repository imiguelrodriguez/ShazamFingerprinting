import argparse
import sqlite3
from utils import read_wav, compute_spectrogram, find_constellation_peaks, hash_generation, save_spectrogram_image, \
    save_constellation_image

MIN_HASHES_FOR_MATCH = 5


def match_hashes(sample_hashes, conn):
    if not sample_hashes:
        return {}
    cur = conn.cursor()
    # Prepare all hashes and their sample times
    hash_to_time = dict(sample_hashes)
    hashes = list(hash_to_time.keys())
    # Use a single query with IN clause
    placeholders = ','.join('?' for _ in hashes)
    query = f"SELECT hash, offset, song_id FROM fingerprints WHERE hash IN ({placeholders})"
    cur.execute(query, hashes)
    offset_counts = {}
    for h, t_db, song_id in cur.fetchall():
        t_sample = hash_to_time[h]
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


def identify_sample(database_path, sample_path, **kwargs):
    window_size = kwargs.get("window_size", None)
    delta_t_max = kwargs.get('delta_t_max', None)
    delta_f_max = kwargs.get('delta_f_max', None)

    data, sr = read_wav(sample_path)
    f, t, Sxx = compute_spectrogram(data, sr)
    save_spectrogram_image(Sxx, t, f, "test.png", ".")
    if window_size:
        peaks = find_constellation_peaks(Sxx, t, f, window_size=int(window_size))
    else:
        peaks = find_constellation_peaks(Sxx, t, f)
    save_constellation_image(Sxx, peaks, "test_const.png", ".", f, t)

    if delta_t_max and delta_f_max:
        hashes = hash_generation(peaks, delta_t_max=int(delta_t_max), delta_f_max=int(delta_f_max))
    elif delta_t_max is not None:
        hashes = hash_generation(peaks, delta_t_max=int(delta_t_max))
    elif delta_f_max is not None:
        hashes = hash_generation(peaks, delta_f_max=int(delta_f_max))
    else:
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
    parser.add_argument('-w', '--windowsize', required=False,
                        help="Size of the window of the maximum filter to obtain constellation peaks.")
    parser.add_argument('-dt', '--deltatime', required=False,
                        help="Time offset to build rectangle in the hashing step.")
    parser.add_argument('-df', '--deltafreq', required=False,
                        help="Frequency offset to build rectangle in the hashing step.")

    args = parser.parse_args()

    # Prepare kwargs depending on which optional args are provided
    kwargs = {}
    if args.windowsize is not None:
        kwargs['window_size'] = args.windowsize
    if args.deltatime is not None:
        kwargs['delta_t_max'] = args.deltatime
    if args.deltafreq is not None:
        kwargs['delta_f_max'] = args.deltafreq

    print(f"Identifying {args.input}...")

    song, score = identify_sample(args.database, args.input, **kwargs)

    if song:
        print(f"Match: {song} (score = {score})")
    else:
        print("No match found.")
