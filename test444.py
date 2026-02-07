import os
import re
import csv
import matplotlib.pyplot as plt
import pandas as pd

# === Configuration ===
base_folders = [
    r"C:\Users\jimmy\Documents\NetBeansProjects\DNNSG\Expirement445566",
    r"C:\Users\jimmy\Documents\NetBeansProjects\DNNSG\Expirement3h",
    r"C:\Users\jimmy\Documents\NetBeansProjects\DNNSG\Expirement4h",
]

log_file_name = "training_log.txt"  # Your log file name
results_csv = "results/experiment_summary.csv"  # Output CSV
plots_dir = "results/plots"  # Folder for saving plots
summary_dir = "results/summaries"  # Folder for grouped summaries

# === Log Parser ===


def parse_log_file(file_path):
    epochs, train_loss, train_accuracy, val_loss, val_accuracy = [], [], [], [], []
    current_epoch = None
    validation_section = False
    val_loss_count = 0

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()

            # Detect new epoch
            epoch_match = re.match(r"Epoch (\d+)", line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                epochs.append(current_epoch)
                validation_section = False
                val_loss_count = 0
                continue

            if "Validation dataset" in line:
                validation_section = True
                continue

            if any(key in line for key in ["CPU Load", "Memory", "Average Epoch Time", "Edge Count"]):
                validation_section = False

            # Extract loss
            loss_match = re.search(r'Loss on Dataset\s*=\s*([\d\.E-]+)', line)
            if loss_match:
                value = float(loss_match.group(1))
                if validation_section:
                    val_loss_count += 1
                    if val_loss_count == 1:
                        val_loss.append(value)
                else:
                    train_loss.append(value)

            # Extract accuracy
            acc_match = re.search(
                r'Accuracy on Dataset\s*=\s*([\d\.]+)\s*%', line)
            if acc_match:
                value = float(acc_match.group(1))
                if validation_section:
                    if val_loss_count == 1:
                        val_accuracy.append(value)
                else:
                    train_accuracy.append(value)

    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'path': file_path
    }


# === Collect Results ===
summary_rows = []

for base in base_folders:
    for root, _, files in os.walk(base):
        if log_file_name in files:
            full_path = os.path.join(root, log_file_name)
            print(f"Parsing: {full_path}")
            data = parse_log_file(full_path)

            if not data['epochs']:
                print(f"Skipping empty log: {full_path}")
                continue

            # Key metrics
            best_val_acc = max(data['val_accuracy']
                               ) if data['val_accuracy'] else None
            best_epoch = data['val_accuracy'].index(
                best_val_acc) if best_val_acc else None
            final_val_acc = data['val_accuracy'][-1] if data['val_accuracy'] else None
            final_val_loss = data['val_loss'][-1] if data['val_loss'] else None
            initial_val_acc = data['val_accuracy'][0] if data['val_accuracy'] else None
            initial_train_acc = data['train_accuracy'][0] if data['train_accuracy'] else None
            final_train_acc = data['train_accuracy'][-1] if data['train_accuracy'] else None

            # Experiment details from folder structure
            parts = full_path.split(os.sep)
            optimizer = parts[-2]  # e.g. Adam, SGD
            sp = [p for p in parts if "SP" in p][0] if any(
                "SP" in p for p in parts) else "NA"
            nodes = [p for p in parts if p.endswith("N")][0] if any(
                p.endswith("N") for p in parts) else "NA"
            inputs = [p for p in parts if "inputs" in p][0] if any(
                "inputs" in p for p in parts) else "NA"
            outputs = [p for p in parts if "outputs" in p][0] if any(
                "outputs" in p for p in parts) else "NA"

            summary_rows.append([
                inputs, outputs, nodes, sp, optimizer,
                initial_train_acc, initial_val_acc,
                best_val_acc, best_epoch,
                final_train_acc, final_val_acc, final_val_loss
            ])

            # === Save Plots ===
            plt.figure(figsize=(12, 5))

            # Loss plot
            plt.subplot(1, 2, 1)
            if data['train_loss']:
                plt.plot(data['epochs'][:len(data['train_loss'])],
                         data['train_loss'], label="Train Loss")
            if data['val_loss']:
                plt.plot(data['epochs'][:len(data['val_loss'])],
                         data['val_loss'], '--', label="Val Loss")
            plt.title("Loss Curve")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()

            # Accuracy plot
            plt.subplot(1, 2, 2)
            if data['train_accuracy']:
                plt.plot(data['epochs'][:len(data['train_accuracy'])],
                         data['train_accuracy'], label="Train Acc")
            if data['val_accuracy']:
                plt.plot(data['epochs'][:len(data['val_accuracy'])],
                         data['val_accuracy'], '--', label="Val Acc")
            plt.title("Accuracy Curve")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy (%)")
            plt.legend()

            # Save
            exp_name = "_".join(
                [inputs, outputs, nodes, sp, optimizer]).replace(os.sep, "_")
            save_path = os.path.join(plots_dir, exp_name + ".png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

# === Write CSV Summary ===
os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)
with open(results_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Inputs", "Outputs", "Nodes", "Skip %", "Optimizer",
        "Initial Train Acc (%)", "Initial Val Acc (%)",
        "Best Val Acc (%)", "Best Epoch",
        "Final Train Acc (%)", "Final Val Acc (%)", "Final Val Loss"
    ])
    writer.writerows(summary_rows)

print(f"Summary saved to {results_csv}")
print(f"Plots saved under {plots_dir}")

# === Grouped Summaries ===
df = pd.DataFrame(summary_rows, columns=[
    "Inputs", "Outputs", "Nodes", "Skip %", "Optimizer",
    "Initial Train Acc (%)", "Initial Val Acc (%)",
    "Best Val Acc (%)", "Best Epoch",
    "Final Train Acc (%)", "Final Val Acc (%)", "Final Val Loss"
])

os.makedirs(summary_dir, exist_ok=True)

# By optimizer
df.groupby("Optimizer").agg(["mean", "std", "min", "max"]).to_csv(
    os.path.join(summary_dir, "by_optimizer.csv"))

# By skip %
df.groupby("Skip %").agg(["mean", "std", "min", "max"]).to_csv(
    os.path.join(summary_dir, "by_skip.csv"))

# By nodes
df.groupby("Nodes").agg(["mean", "std", "min", "max"]).to_csv(
    os.path.join(summary_dir, "by_nodes.csv"))

# By inputs/outputs
df.groupby(["Inputs", "Outputs"]).agg(["mean", "std", "min", "max"]
                                      ).to_csv(os.path.join(summary_dir, "by_io.csv"))

print(f"Grouped summaries saved under {summary_dir}")
