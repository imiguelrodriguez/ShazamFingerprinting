import itertools
import json
import re
import subprocess
import os
import glob
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# ------------- CONFIGURABLE PARAMETERS -------------
window_sizes = [25, 50]
delta_t_max_values = [2, 5]
delta_f_max_values = [1500, 3000]

PARAM_GRID = [
    {"window_size": w, "delta_t_max": dt, "delta_f_max": df}
    for w, dt, df in itertools.product(window_sizes, delta_t_max_values, delta_f_max_values)
]

SONG_DIR = "songs"
SAMPLES_DIR = "samples"
SAMPLE_SUBDIRS = ["clean_samples", "filtered_samples", "noisy_samples", "noisy_filtered_samples"]
GT_METHOD = "filename"  # assumes ground truth in filename like: "10_Traveller_1.wav" → should match "10_Traveller"
IDENTIFY_SCRIPT = "identify.py"
BUILD_SCRIPT = "builddb.py"

# ------------- HELPER FUNCTIONS -------------


def evaluate_confusion_matrix(sample_files, db_path, config):
    """
    Returns
    {ground_truth: {predicted: count}}
    """
    confusion = defaultdict(lambda: defaultdict(int))
    all_labels = set()

    for sample in sample_files:
        print(f"Identifying {sample}")
        t_ini = time.time()
        cmd = [
            sys.executable, IDENTIFY_SCRIPT,
            "-d", db_path,
            "-i", sample,
            "-w", str(config["window_size"]),
            "-dt", str(config["delta_t_max"]),
            "-df", str(config["delta_f_max"])
        ]
        gt = extract_ground_truth(os.path.basename(sample))
        stdout, stderr = run_command(cmd)
        t = time.time() - t_ini
        if stderr:
            print(f"Error in sample {sample}: {stderr}")
        # Default prediction if no match is found
        pred = "No match"

        # Try to extract song name from stdout
        match = re.search(r"Match:\s*(.+?)\s*\(score", stdout)
        if match:
            pred = match.group(1).strip()

        confusion[gt][pred] += 1
        all_labels.update([gt, pred])

    # Convertir a lista ordenada de etiquetas
    labels = sorted(all_labels)

    # Crear matriz numpy para seaborn/matplotlib
    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for i, gt_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            matrix[i, j] = confusion[gt_label].get(pred_label, 0)

    return labels, matrix, t

def plot_confusion_matrix(labels, matrix, title="Confusion Matrix"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def extract_ground_truth(filepath):
    filename = os.path.basename(filepath)
    match = re.match(r'(.+)_\d+\.wav$', filename)
    if match:
        return match.group(1) + '.wav'
    return filename  # fallback if pattern doesn't match

def run_command(command):
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout, result.stderr

def evaluate_identifications(sample_files, db_path):
    correct = 0
    total = len(sample_files)

    for sample in sample_files:
        gt = extract_ground_truth(os.path.basename(sample))
        stdout, stderr = run_command(["python", IDENTIFY_SCRIPT, "-d", db_path, "-i", sample])
        if stderr:
            print(f"Error in sample {sample}: {stderr}")
        if gt.lower() in stdout.lower():
            correct += 1
    return correct / total if total > 0 else 0

# ------------- MAIN EXPERIMENT LOOP -------------

def run_grid_search():
    results = []

    for i, config in enumerate(PARAM_GRID):
        print(f"\n🔧 Testing config {i+1}/{len(PARAM_GRID)}: {config}")
        db_name = f"db_w{config['window_size']}_dt{config['delta_t_max']}_df{config['delta_f_max']}.db"
        db_path = os.path.join("dbs", db_name)
        os.makedirs("dbs", exist_ok=True)

        # Build the database
        cmd = [
            sys.executable, BUILD_SCRIPT,
            "-i", SONG_DIR,
            "-o", db_path,
            "-w", str(config["window_size"]),
            "-dt", str(config["delta_t_max"]),
            "-df", str(config["delta_f_max"])
        ]
        t_ini= time.time()
        stdout, stderr = run_command(cmd)
        t_end = time.time() - t_ini

        print(stdout)
        print(f"Database built in {t_end:.2f} seconds")

        print("\nIdentifying samples...")
            # Store accuracy per subdir
        times = []
        subdir_accuracies = {}
        for subdir in SAMPLE_SUBDIRS:
            subdir_path = os.path.join(SAMPLES_DIR, subdir)
            sample_files = glob.glob(os.path.join(subdir_path, "*", "*.wav"))

            if not sample_files:
                print(f"⚠️ No samples found in {subdir}")
                subdir_accuracies[subdir] = 0.0
                continue

            labels, matrix, t = evaluate_confusion_matrix(sample_files, db_path, config)
            plot_confusion_matrix(labels, matrix, title=f"Confusion Matrix - {subdir}")
            times.append(t)
            correct = sum(matrix[i, i] for i in range(len(labels)))
            total = matrix.sum()
            acc = correct / total if total > 0 else 0
            subdir_accuracies[subdir] = acc
            print(f"  🧪 Accuracy in {subdir}: {acc:.2%}")
        avg_time = np.mean(times)
        results.append({
            "config": config,
            "accuracies": subdir_accuracies,
            "avg_time": avg_time
        })
        print(f"  ⏱️ Average time per sample: {avg_time:.2f} seconds")



    # Print summary
    print("\n📊 Final Grid Search Results:")
    for entry in results:
        print(f"\nConfig: {json.dumps(entry['config'])}")
        for subdir, acc in entry["accuracies"].items():
            print(f"  - {subdir}: {acc:.2%}")
        print(f"  ⏱️ Avg time: {entry['avg_time']:.2f} s")

    # Save to JSON
    with open("grid_search_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Results saved to grid_search_results.json")

# ------------- ENTRY POINT -------------

if __name__ == "__main__":
    run_grid_search()
