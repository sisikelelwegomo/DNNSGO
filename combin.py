import os
import re
import csv
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict
import pandas as pd
import numpy as np

# === Configuration ===
base_folders = [
    r"C:\Users\jimmy\Documents\NetBeansProjects\DNNSG\ExperimentSet_Phase3_Combined7",
]

log_file_name = "combined_training.txt"
common_log_name = "common_log.txt"
results_csv = "ExpirementCombined7_Complete_Analysis.csv"
plots_dir = "results/ExpirementCombined7_Analysis"

# === IMPROVED: Extract path information ===


def extract_path_info(file_path):
    """
    Extract experiment metadata from the directory path.
    Expected structure: .../SkipPercent_XX/XinputXoutput/Teacher_Depth/RepX/Student_X/HP_XXX/
    """
    parts = file_path.split(os.sep)
    info = {}

    for i, part in enumerate(parts):
        if part.startswith("SkipPercent_"):
            info['skip_from_path'] = int(part.replace("SkipPercent_", ""))
        elif "input_" in part and "output" in part:
            match = re.search(r'(\d+)input_(\d+)output', part)
            if match:
                info['input_from_path'] = int(match.group(1))
                info['output_from_path'] = int(match.group(2))
        elif part.startswith("Teacher_"):
            info['teacher_depth'] = part.replace("Teacher_", "")
        elif part.startswith("Rep"):
            match = re.search(r'Rep(\d+)', part)
            if match:
                info['replication'] = int(match.group(1))
        elif part.startswith("Student_"):
            match = re.search(r'Student_(\d+)', part)
            if match:
                info['student_idx'] = int(match.group(1))
        elif part.startswith("HP_"):
            info['hp_name'] = part.replace("HP_", "")

    return info


# === IMPROVED: Enhanced Log Parser ===
def parse_log_file(file_path):
    epochs, train_loss, train_accuracy, val_loss, val_accuracy = [], [], [], [], []
    cpu_load, memory_used, memory_free = [], [], []
    start_times, end_times, avg_times = [], [], []
    weights, edge_counts = [], []
    current_epoch = None
    validation_section = False
    val_loss_count = 0

    learning_rate = None
    dropout_rate = None
    l2_lambda = None
    adam_beta1 = None
    adam_beta2 = None
    adam_epsilon = None
    rmsprop_decay = None
    rmsprop_epsilon = None
    optimizer_type = None
    hp_name = None

    architecture = None
    student_architecture = None
    teacher_architecture = None
    hidden_nodes = None
    layer_config = []
    total_nodes = 0
    skip_percentage = None
    input_size = None
    output_size = None
    network_structure = []

    teacher_params = None
    student_params = None
    baseline_params = None

    teacher_idx = None
    replication = None
    student_idx = None

    with open(file_path, "r", encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            line = line.strip()

            # Extract HP Configuration Name
            if "Name:" in line and "HYPERPARAMETER CONFIGURATION" in lines[max(0, i-5):i]:
                hp_match = re.search(r'Name:\s*(.+)', line)
                if hp_match:
                    hp_name = hp_match.group(1).strip()

            # Extract Optimizer Type
            if "Optimizer:" in line and "HYPERPARAMETER CONFIGURATION" in lines[max(0, i-5):i]:
                opt_match = re.search(r'Optimizer:\s*(.+)', line)
                if opt_match:
                    optimizer_type = opt_match.group(1).strip()
            elif "ADAM OPTIMIZER" in line:
                optimizer_type = "ADAM"
            elif "RMSPROP OPTIMIZER" in line:
                optimizer_type = "RMSPROP"
            elif "STOCHASTIC GRADIENT DESCENT" in line:
                optimizer_type = "SGD"

            # Extract indices from REPLICATION & INDICES section
            if "Teacher Index:" in line:
                idx_match = re.search(r'Teacher Index:\s*(\d+)', line)
                if idx_match:
                    teacher_idx = int(idx_match.group(1))

            if "Replication:" in line and "Teacher Index" not in line:
                rep_match = re.search(r'Replication:\s*(\d+)', line)
                if rep_match:
                    replication = int(rep_match.group(1))

            if "Student Index:" in line:
                sidx_match = re.search(r'Student Index:\s*(\d+)', line)
                if sidx_match:
                    student_idx = int(sidx_match.group(1))

            # Extract Skip Percentage
            if "Skip Percentage:" in line:
                skip_match = re.search(
                    r'Skip Percentage\s*(?:\(FIXED\))?:\s*([\d\.]+)%', line)
                if skip_match:
                    skip_percentage = float(skip_match.group(1))

            # Extract Input and Output sizes
            if "Input Size:" in line:
                input_match = re.search(r'Input Size:\s*(\d+)', line)
                if input_match:
                    input_size = int(input_match.group(1))
            elif "Output Size:" in line:
                output_match = re.search(r'Output Size:\s*(\d+)', line)
                if output_match:
                    output_size = int(output_match.group(1))

            # Extract TEACHER architecture
            if "Teacher:" in line and "ARCHITECTURAL CONFIGURATION" in lines[max(0, i-10):i]:
                teacher_match = re.search(r'Teacher:\s*(.+)', line)
                if teacher_match:
                    teacher_architecture = teacher_match.group(1).strip()

            # Extract STUDENT architecture (the one being trained)
            if "Student:" in line and "ARCHITECTURAL CONFIGURATION" in lines[max(0, i-10):i]:
                student_match = re.search(r'Student:\s*(.+)', line)
                if student_match:
                    student_architecture = student_match.group(1).strip()

            # Extract architecture string for layer config parsing
            if "Architecture:" in line:
                architecture = re.search(r'Architecture:\s*(.+)', line)
                if architecture:
                    architecture = architecture.group(1)

            # Parse layer config from STUDENT architecture
            if student_architecture and not layer_config:
                arch_parts = student_architecture.split('-')
                for part in arch_parts:
                    if 'n' in part or 'o' in part:
                        try:
                            nodes = int(re.search(r'(\d+)', part).group(1))
                            layer_config.append(nodes)
                            total_nodes += nodes
                        except:
                            pass

            # Extract Network Structure details
            if "Network Structure:" in line:
                for j in range(i + 1, min(i + 10, len(lines))):
                    struct_line = lines[j].strip()
                    if not struct_line or "===" in struct_line:
                        break
                    layer_match = re.search(
                        r'(\w+ Layer):\s*(\d+)\s*nodes,\s*(\d+)\s*edges', struct_line)
                    if layer_match:
                        layer_type = layer_match.group(1)
                        nodes = int(layer_match.group(2))
                        edges = int(layer_match.group(3))
                        network_structure.append({
                            'type': layer_type,
                            'nodes': nodes,
                            'edges': edges
                        })

            # Extract hyperparameters
            if "Learning Rate (eta):" in line or "Learning Rate:" in line:
                lr_match = re.search(
                    r'Learning Rate\s*(?:\(eta\))?:\s*([\d\.]+)', line)
                if lr_match:
                    learning_rate = float(lr_match.group(1))
            elif "Dropout Rate:" in line:
                dropout_rate = float(
                    re.search(r'Dropout Rate:\s*([\d\.]+)', line).group(1))
            elif "L2 Lambda:" in line:
                l2_match = re.search(r'L2 Lambda:\s*([\d\.E-]+)', line)
                if l2_match:
                    l2_lambda = float(l2_match.group(1))
            elif "Adam Optimizer:" in line or "Adam Beta" in line:
                adam_match = re.search(
                    r'beta1=([\d\.]+),?\s*beta2=([\d\.]+),?\s*epsilon=([\d\.E-]+)', line)
                if adam_match:
                    adam_beta1 = float(adam_match.group(1))
                    adam_beta2 = float(adam_match.group(2))
                    adam_epsilon = float(adam_match.group(3))
            elif "RMSprop Optimizer:" in line or "RMSprop Decay:" in line:
                rmsprop_match = re.search(
                    r'decay=([\d\.]+),?\s*epsilon=([\d\.E-]+)', line)
                if rmsprop_match:
                    rmsprop_decay = float(rmsprop_match.group(1))
                    rmsprop_epsilon = float(rmsprop_match.group(2))

            # Extract parameter counts
            if "Final Student Parameters:" in line:
                param_match = re.search(
                    r'Final Student Parameters:\s*(\d+)', line)
                if param_match:
                    student_params = int(param_match.group(1))

            if "Class Baseline:" in line:
                baseline_match = re.search(r'Class Baseline:\s*(\d+)', line)
                if baseline_match:
                    baseline_params = int(baseline_match.group(1))

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

    # Create architecture description
    if layer_config:
        arch_description = "-".join([f"{n}N" for n in layer_config])
        depth = len(layer_config)
    else:
        arch_description = student_architecture if student_architecture else architecture
        depth = None

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
        'hidden_nodes': hidden_nodes,
        'architecture': student_architecture,
        'teacher_architecture': teacher_architecture,
        'layer_config': layer_config,
        'total_nodes': total_nodes,
        'skip_percentage': skip_percentage,
        'input_size': input_size,
        'output_size': output_size,
        'network_structure': network_structure,
        'arch_description': arch_description,
        'depth': depth,
        'teacher_params': teacher_params,
        'student_params': student_params,
        'baseline_params': baseline_params,
        'optimizer_type': optimizer_type,
        'hp_name': hp_name,
        'teacher_idx': teacher_idx,
        'replication': replication,
        'student_idx': student_idx,
        'path': file_path
    }


# === Main Processing ===
summary_rows = []

print("\n" + "="*70)
print("PARSING LOG FILES WITH IMPROVED EXTRACTION")
print("="*70 + "\n")

for base in base_folders:
    for root, _, files in os.walk(base):
        if log_file_name in files:
            full_path = os.path.join(root, log_file_name)
            print(f"Parsing: {full_path}")

            # Extract path information
            path_info = extract_path_info(full_path)

            # Parse log file
            data = parse_log_file(full_path)

            if not data['epochs']:
                print(f"  ⚠️  Skipping empty log")
                continue

            best_val_acc = max(data['val_accuracy']
                               ) if data['val_accuracy'] else None
            best_epoch = data['val_accuracy'].index(
                best_val_acc) if best_val_acc else None
            final_val_acc = data['val_accuracy'][-1] if data['val_accuracy'] else None
            final_val_loss = data['val_loss'][-1] if data['val_loss'] else None

            # Use path info as fallback
            skip_percentage = data['skip_percentage'] if data['skip_percentage'] is not None else path_info.get(
                'skip_from_path', "NA")
            input_size = data['input_size'] if data['input_size'] else path_info.get(
                'input_from_path', "NA")
            output_size = data['output_size'] if data['output_size'] else path_info.get(
                'output_from_path', "NA")

            # HP name and optimizer from data or path
            hp_name = data['hp_name'] if data['hp_name'] else path_info.get(
                'hp_name', "Unknown")
            optimizer = data['optimizer_type'] if data['optimizer_type'] else "Unknown"

            # Indices from data or path
            teacher_idx = data['teacher_idx'] if data['teacher_idx'] is not None else "NA"
            replication = data['replication'] if data['replication'] is not None else path_info.get(
                'replication', "NA")
            student_idx = data['student_idx'] if data['student_idx'] is not None else path_info.get(
                'student_idx', "NA")

            teacher_depth = path_info.get('teacher_depth', 'Unknown')

            nodes = data['arch_description'] if data['arch_description'] else "NA"
            depth = data['depth'] if data['depth'] else "NA"
            total_nodes = data['total_nodes'] if data['total_nodes'] else "NA"
            layer_config_str = str(
                data['layer_config']) if data['layer_config'] else "NA"

            teacher_arch = data['teacher_architecture'] if data['teacher_architecture'] else "NA"
            student_arch = data['architecture'] if data['architecture'] else "NA"

            summary_rows.append([
                input_size, output_size, nodes, skip_percentage,
                hp_name, optimizer, teacher_depth,
                teacher_idx, replication, student_idx,
                best_val_acc, best_epoch, final_val_acc, final_val_loss,
                data['cpu_load'][-1] if data['cpu_load'] else None,
                data['memory_used'][-1] if data['memory_used'] else None,
                data['memory_free'][-1] if data['memory_free'] else None,
                data['start_times'][-1] if data['start_times'] else None,
                data['end_times'][-1] if data['end_times'] else None,
                data['avg_times'][-1] if data['avg_times'] else None,
                data['weights'][-1] if data['weights'] else None,
                data['edge_counts'][-1] if data['edge_counts'] else None,
                data['learning_rate'],
                data['dropout_rate'],
                data['l2_lambda'],
                data['adam_beta1'],
                data['adam_beta2'],
                data['adam_epsilon'],
                data['rmsprop_decay'],
                data['rmsprop_epsilon'],
                depth,
                total_nodes,
                layer_config_str,
                teacher_arch,
                student_arch,
                data['teacher_params'],
                data['student_params'],
                data['baseline_params']
            ])

            # === Save Plots ===
            if data['train_loss'] and data['val_loss']:
                plt.figure(figsize=(12, 5))

                plt.subplot(1, 2, 1)
                plt.plot(data['epochs'][:len(data['train_loss'])],
                         data['train_loss'], label="Train Loss")
                plt.plot(data['epochs'][:len(data['val_loss'])],
                         data['val_loss'], '--', label="Val Loss")
                plt.title("Loss Curve")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()

                plt.subplot(1, 2, 2)
                plt.plot(data['epochs'][:len(data['train_accuracy'])],
                         data['train_accuracy'], label="Train Acc")
                plt.plot(data['epochs'][:len(data['val_accuracy'])],
                         data['val_accuracy'], '--', label="Val Acc")
                plt.title("Accuracy Curve")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy (%)")
                plt.legend()

                exp_name = f"{input_size}in_{output_size}out_{nodes}_{skip_percentage}SP_{hp_name}"
                save_path = os.path.join(plots_dir, exp_name + ".png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.tight_layout()
                plt.savefig(save_path)
                plt.close()

            print(
                f"  ✓ Processed | HP: {hp_name} | Arch: {student_arch} | Params: {data['student_params']} | ValAcc: {final_val_acc:.2f}%")

# === Write CSV Summary ===
print("\n" + "="*70)
print("WRITING RESULTS")
print("="*70)

os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)
with open(results_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Inputs", "Outputs", "Nodes", "Skip %",
        "HP Name", "Optimizer Type", "Teacher Depth",
        "Teacher Index", "Replication", "Student Index",
        "Best Val Acc (%)", "Best Epoch", "Final Val Acc (%)", "Final Val Loss",
        "CPU Load (%)", "Memory Used (MB)", "Memory Free (MB)",
        "Start Time", "End Time", "Avg Time (ns)",
        "Average Weight", "Edge Count",
        "Learning Rate", "Dropout Rate", "L2 Lambda",
        "Adam Beta1", "Adam Beta2", "Adam Epsilon",
        "RMSprop Decay", "RMSprop Epsilon",
        "Depth", "Total Nodes", "Layer Config",
        "Teacher Architecture", "Student Architecture",
        "Teacher Params", "Student Params", "Baseline Params"
    ])
    writer.writerows(summary_rows)

print(f"✅ Summary saved to: {results_csv}")
print(f"✅ Total experiments processed: {len(summary_rows)}")

# === Data Quality Check ===
print("\n" + "="*70)
print("DATA QUALITY CHECK")
print("="*70)

df = pd.read_csv(results_csv)
print("\nNull values per column:")
print(df.isnull().sum())
print("\nPercentage of null values per column:")
print((df.isnull().sum() / len(df) * 100).round(2))
print(f"\nTotal rows: {len(df)}")
print(f"Total null values: {df.isnull().sum().sum()}")

# === HYPERPARAMETER ANALYSIS ===
print("\n" + "="*70)
print("HYPERPARAMETER ANALYSIS")
print("="*70)

# Group by HP name
hp_groups = df.groupby('HP Name')

print("\nPerformance by Hyperparameter Set:")
print("-" * 70)
for hp_name, group in hp_groups:
    print(f"\n{hp_name}:")
    print(f"  Count: {len(group)}")
    print(
        f"  Avg Best Val Acc: {group['Best Val Acc (%)'].mean():.2f}% (±{group['Best Val Acc (%)'].std():.2f})")
    print(
        f"  Avg Final Val Acc: {group['Final Val Acc (%)'].mean():.2f}% (±{group['Final Val Acc (%)'].std():.2f})")
    print(f"  Max Best Val Acc: {group['Best Val Acc (%)'].max():.2f}%")
    print(f"  Min Best Val Acc: {group['Best Val Acc (%)'].min():.2f}%")

# === OPTIMIZER COMPARISON ===
print("\n" + "="*70)
print("OPTIMIZER TYPE COMPARISON")
print("="*70)

opt_groups = df.groupby('Optimizer Type')
print("\nPerformance by Optimizer:")
print("-" * 70)
for opt_type, group in opt_groups:
    print(f"\n{opt_type}:")
    print(f"  Count: {len(group)}")
    print(
        f"  Avg Best Val Acc: {group['Best Val Acc (%)'].mean():.2f}% (±{group['Best Val Acc (%)'].std():.2f})")
    print(
        f"  Avg Final Val Acc: {group['Final Val Acc (%)'].mean():.2f}% (±{group['Final Val Acc (%)'].std():.2f})")

# === SKIP PERCENTAGE ANALYSIS ===
print("\n" + "="*70)
print("SKIP PERCENTAGE ANALYSIS")
print("="*70)

skip_groups = df.groupby('Skip %')
print("\nPerformance by Skip Percentage:")
print("-" * 70)
for skip, group in skip_groups:
    print(f"\nSkip {skip}%:")
    print(f"  Count: {len(group)}")
    print(
        f"  Avg Best Val Acc: {group['Best Val Acc (%)'].mean():.2f}% (±{group['Best Val Acc (%)'].std():.2f})")
    print(
        f"  Avg Final Val Acc: {group['Final Val Acc (%)'].mean():.2f}% (±{group['Final Val Acc (%)'].std():.2f})")

# === ARCHITECTURE ANALYSIS ===
print("\n" + "="*70)
print("ARCHITECTURE ANALYSIS")
print("="*70)

# Best performing architectures
best_architectures = df.nlargest(10, 'Best Val Acc (%)')
print("\nTop 10 Best Performing Configurations:")
print("-" * 70)
for idx, row in best_architectures.iterrows():
    print(f"\n{row['Student Architecture']}:")
    print(f"  HP: {row['HP Name']}")
    print(f"  Skip: {row['Skip %']}%")
    print(f"  Best Val Acc: {row['Best Val Acc (%)']}%")
    print(f"  Student Params: {row['Student Params']}")

# === PARAMETER BASELINE CHECK ===
print("\n" + "="*70)
print("PARAMETER BASELINE VERIFICATION")
print("="*70)

# Check if students match their baseline
df['Param Deviation (%)'] = ((df['Student Params'] -
                              df['Baseline Params']) / df['Baseline Params'] * 100).abs()
high_deviation = df[df['Param Deviation (%)'] > 5]

print(
    f"\nExperiments with >5% parameter deviation from baseline: {len(high_deviation)}")
if len(high_deviation) > 0:
    print("\nHigh deviation cases:")
    print(high_deviation[['Student Architecture', 'Student Params',
          'Baseline Params', 'Param Deviation (%)']].head(10))

print("\n" + "="*70)
print("PROCESSING COMPLETE")
print("="*70)
