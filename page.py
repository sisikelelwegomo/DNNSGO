import os
import re
import matplotlib.pyplot as plt

# === Configuration ===
node_class_folders = [
    r"C:\Users\jimmy\Documents\NetBeansProjects\DNNSG\ExpirementDORMS.f\Static\8inputs\4outputs\5N",
    r"C:\Users\jimmy\Documents\NetBeansProjects\DNNSG\ExpirementDORMS.f\Static\8inputs\4outputs\10N",
    r"C:\Users\jimmy\Documents\NetBeansProjects\DNNSG\ExpirementDORMS.f\Static\8inputs\4outputs\15N",
    r"C:\Users\jimmy\Documents\NetBeansProjects\DNNSG\ExpirementDORMS.f\Static\8inputs\4outputs\20N",
    r"C:\Users\jimmy\Documents\NetBeansProjects\DNNSG\ExpirementDORMS.f\Static\8inputs\4outputs\25N",
    r"C:\Users\jimmy\Documents\NetBeansProjects\DNNSG\ExpirementDORMS.f\Static\8inputs\4outputs\30N",
    r"C:\Users\jimmy\Documents\NetBeansProjects\DNNSG\ExpirementDORMS.f\Static\8inputs\4outputs\60N",
    r"C:\Users\jimmy\Documents\NetBeansProjects\DNNSG\ExpirementDORMS.f\Static\8inputs\4outputs\100N",
]

log_file_name = "training_and_evaluation_log.txt"

skip_colors = {
    '0.0SP': 'blue',
    '0.3SP': 'green',
    '0.5SP': 'red',
    '0.8SP': 'purple',
    '1.0SP': 'orange',
}

# === Log Parser ===


def parse_log_file(file_path):
    epochs, train_loss, train_accuracy, val_loss, val_accuracy = [], [], [], [], []
    current_epoch = None
    teacher_layers, student_layers = 0, 0
    skip_connection_type = "No Skip"
    validation_section = False
    val_loss_count = 0  # Track how many validation losses we've seen in current epoch

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()

            if "Number of layers in Teacher:" in line:
                teacher_layers = int(line.split(":")[1].strip())
            elif "Number of layers in Student:" in line:
                student_layers = int(line.split(":")[1].strip())
            elif "Skip Connection Type:" in line:
                skip_connection_type = line.split(":")[1].strip()

            # Detect new epoch
            epoch_match = re.match(r"Epoch (\d+)", line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                epochs.append(current_epoch)
                validation_section = False
                val_loss_count = 0  # Reset validation loss counter
                continue

            # Entering validation section
            if "Validation dataset" in line:
                validation_section = True
                continue

            # Leave validation after any summary line
            if any(key in line for key in ["CPU Load", "Memory", "Average Epoch Time", "Edge Count"]):
                validation_section = False

            # Extract loss
            loss_match = re.search(r'Loss on Dataset\s*=\s*([\d\.E-]+)', line)
            if loss_match:
                value = float(loss_match.group(1))
                if validation_section:
                    val_loss_count += 1
                    # Only take the first validation loss per epoch
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
                    # Only take the first validation accuracy per epoch
                    if val_loss_count == 1:
                        val_accuracy.append(value)
                else:
                    train_accuracy.append(value)

    # Infer skip connection from folder if needed
    skip_folder = os.path.basename(os.path.dirname(file_path))
    for key in skip_colors:
        if key in skip_folder:
            skip_connection_type = key
            break

    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'teacher_layers': teacher_layers,
        'student_layers': student_layers,
        'skip_type': skip_connection_type,
        'path': file_path,
        'folder_structure': os.path.relpath(os.path.dirname(file_path), os.path.commonpath(node_class_folders))
    }


# === Main Loop ===
for node_folder in node_class_folders:
    experiments = []

    for root, _, files in os.walk(node_folder):
        if log_file_name in files:
            full_path = os.path.join(root, log_file_name)
            print(f"Parsing: {full_path}")
            data = parse_log_file(full_path)

            if data['epochs'] and (data['train_loss'] or data['train_accuracy']):
                experiments.append(data)
            else:
                print(f"Skipping (insufficient data): {full_path}")

    if not experiments:
        continue

    plt.figure(figsize=(12, 5))

    # === LOSS PLOT ===
    plt.subplot(1, 2, 1)
    for exp in experiments:
        color = skip_colors.get(exp['skip_type'], 'black')

        # Training loss
        if exp['train_loss']:
            train_epochs = exp['epochs'][:len(exp['train_loss'])]
            plt.plot(train_epochs, exp['train_loss'],
                     color=color, label=f"{exp['skip_type']} Train")
        else:
            print(f"Skipping train loss plot for: {exp['path']}")

        # Validation loss
        if exp['val_loss']:
            val_epochs = exp['epochs'][:len(exp['val_loss'])]
            plt.plot(val_epochs, exp['val_loss'], '--',
                     color=color, label=f"{exp['skip_type']} Val")

    title_parts = experiments[0]['folder_structure'].split(os.sep)
    title = " | ".join(
        title_parts[-3:]) if len(title_parts) >= 3 else experiments[0]['folder_structure']
    plt.title(
        f"{title} - Loss\nTL: {experiments[0]['teacher_layers']}, SL: {experiments[0]['student_layers']}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # === ACCURACY PLOT ===
    plt.subplot(1, 2, 2)
    for exp in experiments:
        color = skip_colors.get(exp['skip_type'], 'black')

        # Training accuracy
        if exp['train_accuracy']:
            acc_epochs = exp['epochs'][:len(exp['train_accuracy'])]
            plt.plot(acc_epochs, exp['train_accuracy'],
                     color=color, label=f"{exp['skip_type']} Train")
        else:
            print(f"Skipping train accuracy plot for: {exp['path']}")

        # Validation accuracy
        if exp['val_accuracy']:
            val_epochs = exp['epochs'][:len(exp['val_accuracy'])]
            plt.plot(val_epochs, exp['val_accuracy'], '--',
                     color=color, label=f"{exp['skip_type']} Val")

    plt.title(
        f"{title} - Accuracy\nTL: {experiments[0]['teacher_layers']}, SL: {experiments[0]['student_layers']}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.show()
