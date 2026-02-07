import os
import re
import csv
import matplotlib.pyplot as plt

# === Configuration ===
base_folders = [
    r"C:\Users\jimmy\Documents\NetBeansProjects\DNNSG\ExpirementSet2_55",
]

log_file_name = "training_log.txt"  # Your log file name
results_csv = "ExpirementSet2_55.1.csv"  # Output CSV
plots_dir = "results/ExpirementSet2_55.1"  # Folder for saving plots


# === Log Parser ===
def parse_log_file(file_path):
    epochs, train_loss, train_accuracy, val_loss, val_accuracy = [], [], [], [], []
    cpu_load, memory_used, memory_free = [], [], []
    start_times, end_times, avg_times = [], [], []
    weights, edge_counts = [], []
    current_epoch = None
    validation_section = False
    val_loss_count = 0

    # Hyperparameters
    learning_rate = None
    dropout_rate = None
    l2_lambda = None
    adam_beta1 = None
    adam_beta2 = None
    adam_epsilon = None
    rmsprop_decay = None
    rmsprop_epsilon = None

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()

            # Extract hyperparameters
            if "Learning Rate (eta):" in line:
                learning_rate = float(
                    re.search(r'Learning Rate \(eta\):\s*([\d\.]+)', line).group(1))
            elif "Dropout Rate:" in line:
                dropout_rate = float(
                    re.search(r'Dropout Rate:\s*([\d\.]+)', line).group(1))
            elif "L2 Lambda:" in line:
                l2_lambda = float(
                    re.search(r'L2 Lambda:\s*([\d\.]+)', line).group(1))
            elif "Adam Optimizer:" in line:
                adam_match = re.search(
                    r'beta1=([\d\.]+), beta2=([\d\.]+), epsilon=([\d\.E-]+)', line)
                if adam_match:
                    adam_beta1 = float(adam_match.group(1))
                    adam_beta2 = float(adam_match.group(2))
                    adam_epsilon = float(adam_match.group(3))
            elif "RMSprop Optimizer:" in line:
                rmsprop_match = re.search(
                    r'decay=([\d\.]+), epsilon=([\d\.E-]+)', line)
                if rmsprop_match:
                    rmsprop_decay = float(rmsprop_match.group(1))
                    rmsprop_epsilon = float(rmsprop_match.group(2))

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

            # CPU Load
            cpu_match = re.search(r'CPU Load:\s*([\d\.]+)\s*%', line)
            if cpu_match:
                cpu_load.append(float(cpu_match.group(1)))

            # Memory
            mem_match = re.search(
                r'Memory Used:\s*(\d+)\s*MB \| Free Memory:\s*(\d+)\s*MB', line)
            if mem_match:
                memory_used.append(int(mem_match.group(1)))
                memory_free.append(int(mem_match.group(2)))

            # Times
            time_match = re.search(
                r'Start Time:\s*(\d+)\s*End Time:\s*(\d+).*Average Time \(ns\):\s*([\d\.E-]+)', line)
            if time_match:
                start_times.append(int(time_match.group(1)))
                end_times.append(int(time_match.group(2)))
                avg_times.append(float(time_match.group(3)))

            # Weights + Edge counts
            weight_match = re.search(r'Average Weight\s*=\s*([\d\.E-]+)', line)
            edge_match = re.search(r'Edge Count\s*=\s*(\d+)', line)
            if weight_match and edge_match:
                weights.append(float(weight_match.group(1)))
                edge_counts.append(int(edge_match.group(1)))

    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'cpu_load': cpu_load,
        'memory_used': memory_used,
        'memory_free': memory_free,
        'start_times': start_times,
        'end_times': end_times,
        'avg_times': avg_times,
        'weights': weights,
        'edge_counts': edge_counts,
        'learning_rate': learning_rate,
        'dropout_rate': dropout_rate,
        'l2_lambda': l2_lambda,
        'adam_beta1': adam_beta1,
        'adam_beta2': adam_beta2,
        'adam_epsilon': adam_epsilon,
        'rmsprop_decay': rmsprop_decay,
        'rmsprop_epsilon': rmsprop_epsilon,
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

            # Extract key metrics
            best_val_acc = max(data['val_accuracy']
                               ) if data['val_accuracy'] else None
            best_epoch = data['val_accuracy'].index(
                best_val_acc) if best_val_acc else None
            final_val_acc = data['val_accuracy'][-1] if data['val_accuracy'] else None
            final_val_loss = data['val_loss'][-1] if data['val_loss'] else None

            # Get experiment details from path
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
                best_val_acc, best_epoch, final_val_acc, final_val_loss,
                data['cpu_load'][-1] if data['cpu_load'] else None,
                data['memory_used'][-1] if data['memory_used'] else None,
                data['memory_free'][-1] if data['memory_free'] else None,
                data['start_times'][-1] if data['start_times'] else None,
                data['end_times'][-1] if data['end_times'] else None,
                data['avg_times'][-1] if data['avg_times'] else None,
                data['weights'][-1] if data['weights'] else None,
                data['edge_counts'][-1] if data['edge_counts'] else None,
                # Hyperparameters
                data['learning_rate'],
                data['dropout_rate'],
                data['l2_lambda'],
                data['adam_beta1'],
                data['adam_beta2'],
                data['adam_epsilon'],
                data['rmsprop_decay'],
                data['rmsprop_epsilon']
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
        "Best Val Acc (%)", "Best Epoch", "Final Val Acc (%)", "Final Val Loss",
        "CPU Load (%)", "Memory Used (MB)", "Memory Free (MB)",
        "Start Time", "End Time", "Avg Time (ns)",
        "Average Weight", "Edge Count",
        # Hyperparameter columns
        "Learning Rate", "Dropout Rate", "L2 Lambda",
        "Adam Beta1", "Adam Beta2", "Adam Epsilon",
        "RMSprop Decay", "RMSprop Epsilon"
    ])
    writer.writerows(summary_rows)

print(f"Summary saved to {results_csv}")
print(f"Plots saved under {plots_dir}")
