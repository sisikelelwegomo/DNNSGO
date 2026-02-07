import os
import re
import csv
import matplotlib.pyplot as plt
from glob import glob

# === Configuration ===
base_folders = [
    r"C:\Users\jimmy\Documents\NetBeansProjects\DNNSG\ExperimentSet_Phase3_Combined6",
]

log_file_name = "training_log.txt"
common_log_name = "common_log.txt"  # NEW: Common log file name
results_csv = "ExpirementCombined6_55.6_arch.csv"
plots_dir = "results/ExpirementCombined6_55.6_arch"

# NEW: CSV Combining Configuration
combine_csvs = True  # Set to True to combine multiple CSVs
csv_files_to_combine = [
    "ExpirementCombined6_55.6_arch.csv",
    # Add more CSV files here
    # "ExpirementCombined6_old.csv",
    # "ExpirementCombined6_another.csv",
]
combined_output_csv = "ExpirementSet6_Combined.csv"

# === NEW: Common Log Parser ===


def parse_common_log(file_path):
    """
    Parse common_log.txt to extract teacher and student parameters
    """
    teacher_params = None
    student_params = None
    baseline_params = None

    if not os.path.exists(file_path):
        print(f"  ⚠️  Common log not found: {file_path}")
        return teacher_params, student_params, baseline_params

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()

            # Extract teacher parameters
            teacher_match = re.search(
                r'Teacher Parameters \(A\):\s*(\d+)', content)
            if teacher_match:
                teacher_params = int(teacher_match.group(1))

            # Extract student parameters
            student_match = re.search(
                r'Student Parameters \(B\):\s*(\d+)', content)
            if student_match:
                student_params = int(student_match.group(1))

            # Extract baseline parameters
            baseline_match = re.search(
                r'Baseline Parameters \(0% skip\):\s*(\d+)', content)
            if baseline_match:
                baseline_params = int(baseline_match.group(1))

            # Alternative patterns if above don't match
            if not teacher_params:
                alt_teacher_match = re.search(
                    r'Teacher Parameters:\s*(\d+)', content)
                if alt_teacher_match:
                    teacher_params = int(alt_teacher_match.group(1))

            if not student_params:
                alt_student_match = re.search(
                    r'Actual Student Parameters \(B\):\s*(\d+)', content)
                if alt_student_match:
                    student_params = int(alt_student_match.group(1))

            print(
                f"  ✓ Teacher Params: {teacher_params}, Student Params: {student_params}, Baseline: {baseline_params}")

    except Exception as e:
        print(f"  ⚠️  Error parsing common log: {e}")

    return teacher_params, student_params, baseline_params

# === Enhanced Log Parser ===


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

    # Architecture info
    architecture = None
    hidden_nodes = None
    layer_config = []
    total_nodes = 0
    skip_percentage = None
    input_size = None
    output_size = None
    network_structure = []

    # NEW: Teacher/Student parameters
    teacher_params = None
    student_params = None
    baseline_params = None

    with open(file_path, "r", encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            line = line.strip()

            # Extract Skip Percentage - FIXED REGEX
            if "Skip Percentage" in line:
                # This pattern handles both "Skip Percentage:" and "Skip Percentage (FIXED):"
                skip_match = re.search(
                    r'Skip Percentage\s*(?:\(FIXED\))?:\s*([\d\.]+)%', line)
                if skip_match:
                    skip_percentage = float(skip_match.group(1))
                    # Debug
                    print(f"  ✓ Found skip percentage: {skip_percentage}%")

            # Extract Input and Output sizes
            if "Input Size:" in line:
                input_match = re.search(r'Input Size:\s*(\d+)', line)
                if input_match:
                    input_size = int(input_match.group(1))
            elif "Output Size:" in line:
                output_match = re.search(r'Output Size:\s*(\d+)', line)
                if output_match:
                    output_size = int(output_match.group(1))

            # Extract architecture string
            if "Architecture:" in line:
                architecture = re.search(r'Architecture:\s*(.+)', line)
                if architecture:
                    architecture = architecture.group(1)
                    arch_parts = architecture.split('-')
                    for part in arch_parts:
                        if 'n' in part or 'o' in part:
                            nodes = int(re.search(r'(\d+)', part).group(1))
                            layer_config.append(nodes)
                            total_nodes += nodes

            # Extract Network Structure details
            if "Network Structure:" in line:
                for j in range(i+1, min(i+10, len(lines))):
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

    # NEW: Parse common_log.txt for teacher/student parameters
    common_log_path = file_path.replace(log_file_name, common_log_name)
    teacher_params, student_params, baseline_params = parse_common_log(
        common_log_path)

    # Create architecture description
    if layer_config:
        arch_description = "-".join([f"{n}N" for n in layer_config])
        depth = len(layer_config)
    else:
        arch_description = architecture
        depth = None

    # Debug output
    print(f"  Debug - Skip % extracted: {skip_percentage}")
    print(f"  Debug - Input size: {input_size}")
    print(f"  Debug - Output size: {output_size}")
    print(f"  Debug - Architecture: {architecture}")
    print(f"  Debug - Teacher Params: {teacher_params}")
    print(f"  Debug - Student Params: {student_params}")

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
        'architecture': architecture,
        'layer_config': layer_config,
        'total_nodes': total_nodes,
        'skip_percentage': skip_percentage,
        'input_size': input_size,
        'output_size': output_size,
        'network_structure': network_structure,
        'arch_description': arch_description,
        'depth': depth,
        'teacher_params': teacher_params,  # NEW
        'student_params': student_params,  # NEW
        'baseline_params': baseline_params,  # NEW
        'path': file_path
    }

# === CSV Combining Function ===


def combine_csv_files(csv_files, output_file):
    """
    Combines multiple CSV files into one, removing duplicates based on all columns.
    """
    print("\n" + "="*60)
    print("COMBINING CSV FILES")
    print("="*60)

    all_rows = []
    headers = None
    seen_rows = set()

    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"⚠️  Warning: {csv_file} not found, skipping...")
            continue

        print(f"Reading: {csv_file}")
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            file_headers = next(reader)

            if headers is None:
                headers = file_headers
            elif headers != file_headers:
                print(f"⚠️  Warning: Headers don't match in {csv_file}")
                print(f"   Expected: {headers}")
                print(f"   Got: {file_headers}")
                continue

            rows_added = 0
            duplicates_skipped = 0

            for row in reader:
                # Create a tuple of the row for hashing (to detect duplicates)
                row_tuple = tuple(row)

                if row_tuple not in seen_rows:
                    seen_rows.add(row_tuple)
                    all_rows.append(row)
                    rows_added += 1
                else:
                    duplicates_skipped += 1

            print(f"   ✓ Added {rows_added} unique rows")
            if duplicates_skipped > 0:
                print(f"   ⊗ Skipped {duplicates_skipped} duplicate rows")

    # Write combined file
    if all_rows:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(all_rows)

        print(f"\n✅ Combined CSV saved: {output_file}")
        print(f"   Total unique rows: {len(all_rows)}")
        print(
            f"   Total files combined: {len([f for f in csv_files if os.path.exists(f)])}")
    else:
        print("\n❌ No data to combine!")

    print("="*60 + "\n")
    return len(all_rows)


# === Main Processing ===
summary_rows = []

print("\n" + "="*60)
print("PARSING LOG FILES")
print("="*60 + "\n")

for base in base_folders:
    for root, _, files in os.walk(base):
        if log_file_name in files:
            full_path = os.path.join(root, log_file_name)
            print(f"Parsing: {full_path}")
            data = parse_log_file(full_path)

            if not data['epochs']:
                print(f"  ⚠️  Skipping empty log")
                continue

            # Extract key metrics
            best_val_acc = max(data['val_accuracy']
                               ) if data['val_accuracy'] else None
            best_epoch = data['val_accuracy'].index(
                best_val_acc) if best_val_acc else None
            final_val_acc = data['val_accuracy'][-1] if data['val_accuracy'] else None
            final_val_loss = data['val_loss'][-1] if data['val_loss'] else None

            # Get experiment details
            parts = full_path.split(os.sep)
            optimizer = parts[-2] if len(parts) > 2 else "Unknown"

            # Use extracted architecture info
            nodes = data['arch_description'] if data['arch_description'] else "NA"
            skip_percentage = data['skip_percentage'] if data['skip_percentage'] is not None else "NA"
            input_size = data['input_size'] if data['input_size'] else "NA"
            output_size = data['output_size'] if data['output_size'] else "NA"
            depth = data['depth'] if data['depth'] else "NA"
            total_nodes = data['total_nodes'] if data['total_nodes'] else "NA"
            layer_config_str = str(
                data['layer_config']) if data['layer_config'] else "NA"

            # NEW: Add teacher/student parameters to CSV row
            summary_rows.append([
                input_size, output_size, nodes, skip_percentage, optimizer,
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
                data['architecture'] if data['architecture'] else "NA",
                data['teacher_params'],  # NEW: Teacher parameters
                data['student_params'],  # NEW: Student parameters
                data['baseline_params']  # NEW: Baseline parameters
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
            exp_name = f"{input_size}in_{output_size}out_{nodes}_{skip_percentage}SP_{optimizer}"
            save_path = os.path.join(plots_dir, exp_name + ".png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

            print(f"  ✓ Processed successfully")

# === Write CSV Summary ===
print("\n" + "="*60)
print("WRITING RESULTS")
print("="*60)

os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)
with open(results_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Inputs", "Outputs", "Nodes", "Skip %", "Optimizer",
        "Best Val Acc (%)", "Best Epoch", "Final Val Acc (%)", "Final Val Loss",
        "CPU Load (%)", "Memory Used (MB)", "Memory Free (MB)",
        "Start Time", "End Time", "Avg Time (ns)",
        "Average Weight", "Edge Count",
        "Learning Rate", "Dropout Rate", "L2 Lambda",
        "Adam Beta1", "Adam Beta2", "Adam Epsilon",
        "RMSprop Decay", "RMSprop Epsilon",
        "Depth", "Total Nodes", "Layer Config", "Architecture String",
        "Teacher Params", "Student Params", "Baseline Params"  # NEW COLUMNS
    ])
    writer.writerows(summary_rows)

print(f"✅ Summary saved to: {results_csv}")
print(f"✅ Plots saved under: {plots_dir}")
print(f"✅ Total experiments processed: {len(summary_rows)}")

# === Combine CSVs (if enabled) ===
if combine_csvs and len(csv_files_to_combine) > 0:
    total_combined = combine_csv_files(
        csv_files_to_combine, combined_output_csv)

print("\n" + "="*60)
print("PROCESSING COMPLETE")
print("="*60)
