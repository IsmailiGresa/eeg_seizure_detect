import torch.optim as optim
from models.detector_models.cnn_bigru_selfattention import CNN2D_BiGRU
from eval_model import *
from utils.logger import Logger
from data.data_preprocess import get_data_preprocessed

def parse_args():
    parser = argparse.ArgumentParser(description='Train CNN2D_BiGRU Model')

    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training and validation')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=4, help='Number of epochs for training')
    parser.add_argument('--output_dim', type=int, default=2, help='Number of output classes')
    parser.add_argument('--num_channel', type=int, default=128, help='Number of input channels (e.g., EEG channels)')
    parser.add_argument('--enc_model', type=str, default='raw', help='Feature extraction model (e.g., stft or raw)')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of GRU layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate for regularization')
    parser.add_argument('--attention_dim', type=int, default=128, help='Dimension for self-attention layer')
    parser.add_argument('--multi_head_num', type=int, default=4, help='Number of heads for multi-head attention')
    parser.add_argument('--calibration', action='store_true', help='Enable model calibration (default: False)')
    parser.add_argument('--binary_target_groups', type=int, default=1, help='Number of binary target groups for EEG classification')
    parser.add_argument('--feature_sample_rate', type=int, default=100, help='Feature extraction sample rate')
    parser.add_argument('--margin_test', action='store_true', help='Enable margin testing for evaluation')

    # Sliding Window Parameters
    parser.add_argument('--window_shift_label', type=int, default=100, help='Sliding window shift for labels')
    parser.add_argument('--window_shift', type=int, default=100, help='Sliding window shift for signals')
    parser.add_argument('--window_size_label', type=int, default=200, help='Sliding window size for labels')
    parser.add_argument('--window_size', type=int, default=200, help='Sliding window size for signals')
    parser.add_argument('--requirement_target', type=float, default=None, help='Threshold for required target samples in a window (0 to 1)')
    parser.add_argument("--sample_rate", type=int, default=256)

    # Logging and Checkpointing
    parser.add_argument('--dir_result', type=str, default='results', help='Root directory for saving results')
    parser.add_argument('--project_name', type=str, default='CNN2D_BiGRU_Project', help='Project name for logging')
    parser.add_argument('--log_iter', type=int, default=10, help='Logging frequency per iteration')
    parser.add_argument('--reset', action='store_true', help='Reset the logging and checkpoint directories')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--eeg_type', type=str, default='bipolar', choices=['unipolar', 'bipolar', 'uni_bipolar'],
                        help='Type of EEG signals to use')
    parser.add_argument('--data_path', type=str, default='/Users/gresaismaili/Desktop/TUH-EEG',
                        help='Path to the root directory of the EEG dataset')
    parser.add_argument('--loss_decision', type=str, default='mean', choices=['max_division', 'mean'],
                        help='Loss decision type: max_division or mean')

    args = parser.parse_args([])
    return args

# Parse Arguments and Set Device
args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========== Data Loading ========== #
print("\n=== Loading Data ===")
train_loader, val_loader, test_loader, train_size, val_size, test_size = get_data_preprocessed(args)
print(f"Train Size: {train_size}, Validation Size: {val_size}, Test Size: {test_size}")

# ========== Model Initialization ========== #
print("\n=== Initializing Model ===")
model = CNN2D_BiGRU(args, device).to(device)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.0005
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

logger = Logger(args)

# ========== Training and Validation Loop ========== #
best_val_accuracy = 0.0
for epoch in range(args.num_epochs):
    print(f"\n=== Epoch {epoch+1}/{args.num_epochs} ===")

    # ========== Training ========== #
    model.train()
    train_loss = []

    for batch_idx, (inputs, targets, seq_lengths, filenames) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        model, iter_loss = sliding_window_v1(
            args=args,
            iteration=batch_idx,
            train_x=inputs,
            train_y=targets,
            seq_lengths=seq_lengths,
            model=model,
            logger=logger,
            device=device,
            scheduler=scheduler,
            optimizer=optimizer,
            criterion=criterion,
            flow_type="train"
        )
        train_loss.extend(iter_loss)

    avg_train_loss = np.mean(train_loss)
    print(f"Train Loss: {avg_train_loss:.4f}")

    # ========== Validation ========== #
    model.eval()
    val_loss = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets, seq_lengths, filenames in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            model, val_iter_loss = sliding_window_v1(
                args=args,
                iteration=batch_idx,
                train_x=inputs,
                train_y=targets,
                seq_lengths=seq_lengths,
                model=model,
                logger=logger,
                device=device,
                scheduler=scheduler,
                optimizer=optimizer,
                criterion=criterion,
                flow_type="validation"
            )
            val_loss.extend(val_iter_loss)

            outputs = model(inputs)
            _, predicted = outputs.max(1)

            if targets.ndim > 1:
                target_temp = targets.float()
                target, _ = torch.max(target_temp, dim=1)
                final_target = torch.round(target).long()
            else:
                final_target = targets

            if predicted.shape != final_target.shape:
                print(f"⚠️ Shape mismatch: predicted {predicted.shape}, target {final_target.shape}")
                min_len = min(predicted.shape[0], final_target.shape[0])
                predicted = predicted[:min_len]
                final_target = final_target[:min_len]

            correct += predicted.eq(final_target).sum().item()
            total += final_target.size(0)

    avg_val_loss = np.mean(val_loss)
    val_accuracy = 100. * correct / total
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # ========== Save Best Model ========== #
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Model saved with accuracy: {best_val_accuracy:.2f}%")

print("\n=== Training Complete ===")

print("\n=== Evaluating on Test Set (/eval/) ===")
model.eval()
test_loss = []
correct = 0
total = 0

with torch.no_grad():
    for inputs, targets, seq_lengths, filenames in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        model, batch_loss = sliding_window_v1(
            args=args,
            iteration=0,
            train_x=inputs,
            train_y=targets,
            seq_lengths=seq_lengths,
            model=model,
            logger=logger,
            device=device,
            scheduler=scheduler,
            optimizer=optimizer,
            criterion=criterion,
            flow_type="test"
        )

        test_loss.extend(batch_loss)

        outputs = model(inputs)
        _, predicted = outputs.max(1)

        # Handle sequence targets
        if targets.ndim > 1:
            targets, _ = torch.max(targets.float(), dim=1)
            targets = torch.round(targets).long()

        if predicted.shape != targets.shape:
            min_len = min(predicted.shape[0], targets.shape[0])
            predicted = predicted[:min_len]
            targets = targets[:min_len]

        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

avg_test_loss = np.mean(test_loss)
test_accuracy = 100. * correct / total

print(f"🧪 Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
