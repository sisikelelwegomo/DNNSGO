from glob import glob
import matplotlib.pyplot as plt
import os
import re
import csv
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend BEFORE importing pyplot

# === Configuration ===
base_folders = [
    r"C:\Users\jimmy\Documents\NetBeansProjects\DNNSGO\ParkinsonExperiments",
]

log_file_name = "training_log.txt"
common_log_name = "common_log.txt"
results_csv = "ExpirementSetFinal_Complete.csv"
plots_dir = "results/ExpirementFinal"

combine_csvs = True
csv_files_to_combine = [
    "ExpirementSetFinal_Complete.csv",
]
combined_output_csv = "ExpirementSetFinal_Complete_Combined.csv"

# ============================================================================
# EXTRACT SKIP % FROM PATH (Most Reliable!)
# ============================================================================


def extract_skip_from_path(file_path):
    """
    Extract skip percentage directly from folder path.
    Example: ...\\Skip_25%\\...
    """
    patterns = [
        r'Skip_(\d+(?:\.\d+)?)%',  # Skip_25%
        r'Skip[_\-](\d+(?:\.\d+)?)',  # Skip_25 or Skip-25
    ]

    for pattern in patterns:
        match = re.search(pattern, file_path)
        if match:
            return float(match.group(1))

    return None


# ============================================================================
# PARSE COMMON LOG
# ============================================================================


def parse_common_log(file_path):
    """Parse common_log.txt"""
    teacher_params = None
    student_params = None
    baseline_params = None

    if not os.path.exists(file_path):
        return teacher_params, student_params, baseline_params

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()

            if not content.strip():
                return teacher_params, student_params, baseline_params

            # Teacher patterns
            teacher_patterns = [
                r'Teacher Parameters \(A\):\s*(\d+)',
                r'Teacher Parameters:\s*(\d+)',
                r'Teacher\s*(?:\(A\))?:\s*(\d+)',
                r'TEACHER[:\s]+(\d+)',
            ]

            for pattern in teacher_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    teacher_params = int(match.group(1))
                    break

            # Student patterns
            student_patterns = [
                r'Student Parameters \(B\):\s*(\d+)',
                r'Actual Student Parameters \(B\):\s*(\d+)',
                r'Student Parameters:\s*(\d+)',
                r'Student\s*(?:\(B\))?:\s*(\d+)',
                r'STUDENT[:\s]+(\d+)',
            ]

            for pattern in student_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    student_params = int(match.group(1))
                    break

            # Baseline patterns
            baseline_patterns = [
                r'Baseline Parameters \(0%\s*skip\):\s*(\d+)',
                r'Baseline Parameters:\s*(\d+)',
                r'Baseline\s*(?:\(0%\s*skip\))?:\s*(\d+)',
                r'BASELINE[:\s]+(\d+)',
            ]

            for pattern in baseline_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    baseline_params = int(match.group(1))
                    break

    except Exception as e:
        pass

    return teacher_params, student_params, baseline_params


# ============================================================================
# PARSE LOG FILE
# ============================================================================


def parse_log_file(file_path):
    """Parse training log file"""
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

    # Teacher/Student parameters
    teacher_params = None
    student_params = None
    baseline_params = None

    with open(file_path, "r", encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            line = line.strip()

            # Extract Skip Percentage from log
            if "Skip Percentage" in line:
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

    # Parse common_log.txt
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
        'teacher_params': teacher_params,
        'student_params': student_params,
        'baseline_params': baseline_params,
        'path': file_path
    }


# ============================================================================
# MAIN PROCESSING
# ============================================================================

summary_rows = []
skip_percentages_found = set()

print("\n" + "="*100)
print("PARSING LOG FILES - ALL SKIP PERCENTAGES")
print("="*100 + "\n")

log_count = 0
processed_count = 0

for base in base_folders:
    print(f"Scanning: {base}\n")

    for root, _, files in os.walk(base):
        if log_file_name in files:
            full_path = os.path.join(root, log_file_name)
            log_count += 1

            # Extract skip % from PATH first (most reliable!)
            skip_from_path = extract_skip_from_path(full_path)

            print(f"[{log_count}] {full_path[-100:]}")
            print(f"      Skip % from path: {skip_from_path}")

            # Parse the log file
            try:
                data = parse_log_file(full_path)

                if not data['epochs']:
                    print(f"      ⚠️  Empty log, skipping\n")
                    continue

                # Use skip % from path (fallback to log if not in path)
                skip_percentage = skip_from_path if skip_from_path is not None else data[
                    'skip_percentage']

                if skip_percentage is not None:
                    skip_percentages_found.add(skip_percentage)
                    print(f"      ✓ Skip %: {skip_percentage}%")

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
                input_size = data['input_size'] if data['input_size'] else "NA"
                output_size = data['output_size'] if data['output_size'] else "NA"
                depth = data['depth'] if data['depth'] else "NA"
                total_nodes = data['total_nodes'] if data['total_nodes'] else "NA"
                layer_config_str = str(
                    data['layer_config']) if data['layer_config'] else "NA"

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
                    data['teacher_params'],
                    data['student_params'],
                    data['baseline_params']
                ])

                # === Save Plots (NO DISPLAY) ===
                try:
                    fig = plt.figure(figsize=(12, 5))

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

                    exp_name = f"{input_size}in_{output_size}out_{nodes}_{skip_percentage}SP_{optimizer}"
                    save_path = os.path.join(plots_dir, exp_name + ".png")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    plt.tight_layout()
                    plt.savefig(save_path)
                    plt.close(fig)  # Close without displaying

                except Exception as plot_error:
                    print(f"      ⚠️  Plot save error: {plot_error}")
                    plt.close('all')

                processed_count += 1
                print(f"      ✓ Processed successfully\n")

            except Exception as e:
                print(f"      ❌ Error: {e}\n")
                continue

# ============================================================================
# WRITE CSV SUMMARY
# ============================================================================

print("\n" + "="*100)
print("WRITING RESULTS")
print("="*100)

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
        "Teacher Params", "Student Params", "Baseline Params"
    ])
    writer.writerows(summary_rows)

print(f"\n✅ Summary saved to: {results_csv}")
print(f"✅ Plots saved under: {plots_dir}")
print(f"\n📊 SUMMARY STATISTICS:")
print(f"   Total log files found: {log_count}")
print(f"   Successfully processed: {processed_count}")
print(f"   Failed: {log_count - processed_count}")

print(f"\n📊 Skip Percentages Found in Dataset:")
for skip_pct in sorted(skip_percentages_found):
    count = sum(1 for row in summary_rows if row[3] == skip_pct)
    print(f"   • {skip_pct}%: {count} experiments")

print("\n" + "="*100)
print("✅ PROCESSING COMPLETE - NO DISPLAY ERRORS")
print("="*100)
