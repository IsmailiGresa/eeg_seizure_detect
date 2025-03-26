# import os
# import numpy as np
# import torch
# import csv
# import torch.nn as nn
# import torch.optim as optim
# import argparse
# from torch.utils.data import DataLoader
# from models.detector_models.cnn_bigru_selfattention import CNN2D_BiGRU
# from train.trainer import sliding_window_v1
# from control.config import args
# from utils.logger import Logger
# from data.data_preprocess import get_data_preprocessed
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt
#
#
# # ========== Parse Arguments for Evaluation ========== #
# def parse_args():
#     parser = argparse.ArgumentParser(description='Evaluate CNN2D_BiGRU Model')
#
#     parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training and validation')
#     parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate for the optimizer')
#     parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs for training')
#     parser.add_argument('--output_dim', type=int, default=2, help='Number of output classes')
#     parser.add_argument('--num_channel', type=int, default=128, help='Number of input channels (e.g., EEG channels)')
#     parser.add_argument('--enc_model', type=str, default='raw', help='Feature extraction model (e.g., stft or raw)')
#     parser.add_argument('--num_layers', type=int, default=1, help='Number of GRU layers')
#     parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate for regularization')
#     parser.add_argument('--attention_dim', type=int, default=128, help='Dimension for self-attention layer')
#     parser.add_argument('--multi_head_num', type=int, default=4, help='Number of heads for multi-head attention')
#     parser.add_argument('--calibration', action='store_true', help='Enable model calibration (default: False)')
#     parser.add_argument('--binary_target_groups', type=int, default=1,
#                         help='Number of binary target groups for EEG classification')
#     parser.add_argument('--feature_sample_rate', type=int, default=100, help='Feature extraction sample rate')
#     parser.add_argument('--margin_test', action='store_true', help='Enable margin testing for evaluation')
#
#     # Sliding Window Parameters
#     parser.add_argument('--window_shift_label', type=int, default=100, help='Sliding window shift for labels')
#     parser.add_argument('--window_shift', type=int, default=100, help='Sliding window shift for signals')
#     parser.add_argument('--window_size_label', type=int, default=200, help='Sliding window size for labels')
#     parser.add_argument('--window_size', type=int, default=200, help='Sliding window size for signals')
#     parser.add_argument('--requirement_target', type=float, default=None,
#                         help='Threshold for required target samples in a window (0 to 1)')
#     parser.add_argument("--sample_rate", type=int, default=256)
#
#     # Logging and Checkpointing
#     parser.add_argument('--dir_result', type=str, default='results', help='Root directory for saving results')
#     parser.add_argument('--project_name', type=str, default='CNN2D_BiGRU_Project', help='Project name for logging')
#     parser.add_argument('--log_iter', type=int, default=10, help='Logging frequency per iteration')
#     parser.add_argument('--reset', action='store_true', help='Reset the logging and checkpoint directories')
#     parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
#     parser.add_argument('--eeg_type', type=str, default='bipolar', choices=['unipolar', 'bipolar', 'uni_bipolar'],
#                         help='Type of EEG signals to use')
#     parser.add_argument('--data_path', type=str, default='/Users/gresaismaili/Desktop/TUH-EEG',
#                         help='Path to the root directory of the EEG dataset')
#     parser.add_argument('--loss_decision', type=str, default='mean', choices=['max_division', 'mean'],
#                         help='Loss decision type: max_division or mean')
#
#     args = parser.parse_args([])
#     return args
#
# # Parse Arguments and Set Device
# args = parse_args()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
#
# # ========== Load Only Test Data ========== #
# print("\n=== Loading /eval/ Data Only ===")
# _, _, test_loader, _, _, _ = get_data_preprocessed(args)
#
# # ========== Load Trained Model ========== #
# print("\n=== Loading Model ===")
# model = CNN2D_BiGRU(args, device).to(device)
# model.load_state_dict(torch.load('best_model.pth', map_location=device))
# model.eval()
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
# logger = Logger(args)
#
# # ========== Evaluation on Test Set ========== #
# print("\n=== Evaluating on Test Set (/eval/) ===")
# test_loss = []
# correct = 0
# total = 0
#
# all_filenames = []
# all_targets = []
# all_predictions = []
#
# with torch.no_grad():
#     for inputs, targets, seq_lengths, filenames in test_loader:
#         inputs, targets = inputs.to(device), targets.to(device)
#
#         # Forward pass
#         model, batch_loss = sliding_window_v1(
#             args=args,
#             iteration=0,
#             train_x=inputs,
#             train_y=targets,
#             seq_lengths=seq_lengths,
#             model=model,
#             logger=logger,
#             device=device,
#             scheduler=scheduler,
#             optimizer=optimizer,
#             criterion=criterion,
#             flow_type="test"
#         )
#
#         test_loss.extend(batch_loss)
#         outputs = model(inputs)
#         _, predicted = outputs.max(1)
#
#         # Handle sequence targets
#         if targets.ndim > 1:
#             targets, _ = torch.max(targets.float(), dim=1)
#             targets = torch.round(targets).long()
#
#         if predicted.shape != targets.shape:
#             min_len = min(predicted.shape[0], targets.shape[0])
#             predicted = predicted[:min_len]
#             targets = targets[:min_len]
#
#         correct += predicted.eq(targets).sum().item()
#         total += targets.size(0)
#
# avg_test_loss = np.mean(test_loss)
# test_accuracy = 100. * correct / total
#
# all_filenames.extend(filenames)
# all_targets.extend(targets.cpu().numpy())
# all_predictions.extend(predicted.cpu().numpy())
#
# print(f"🧪 Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
#
# print("\n🧾 Predictions on /eval/ set:")
# for fname, target, pred in zip(all_filenames, all_targets, all_predictions):
#     print(f"📄 {os.path.basename(fname)} | Target: {target} | Predicted: {pred}")
#
# # ========== Save Predictions to CSV ========== #
# with open("eval_predictions.csv", "w", newline="") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["filename", "target", "predicted"])
#     for fname, target, pred in zip(all_filenames, all_targets, all_predictions):
#         writer.writerow([os.path.basename(fname), target, pred])
#
# print("\n✅ Predictions saved to eval_predictions.csv")
#
# print("\n📊 Confusion Matrix:")
# cm = confusion_matrix(all_targets, all_predictions, labels=[0, 1])
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
# disp.plot(cmap='Blues', values_format='d')
# plt.title("Confusion Matrix (Eval Set)")
# plt.show()

import os
import torch
import numpy as np
import random
from tqdm import tqdm
import torch.nn as nn
from data.data_preprocess import get_data_preprocessed
from models import get_detector_model
from utils.metrics import Evaluator
from utils.logger import Logger
from utils.utils import set_seeds, set_devices
from train.trainer import sliding_window_v1
import argparse
from control.config import args

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CNN2D_BiGRU Model')

    # Model Hyperparameters
    parser.add_argument('--model', type=str, default='cnn_bigru_selfattention',
                        help='Model name to use from models/detector_models/')
    parser.add_argument('--cpu', action='store_true', help='Force using CPU even if CUDA is available')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training and validation')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs for training')
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
    parser.add_argument('--task_type', type=str, default='binary',
                        choices=['binary', 'binary_noslice', 'multiclass'],
                        help='Type of classification task')
    parser.add_argument('--tnr_for_margintest', nargs='+', type=float, default=[0.98],
                        help='TNR thresholds for margin-based evaluation')
    parser.add_argument('--seizure_wise_eval_for_binary', action='store_true',
                        help='Enable seizure-wise evaluation for binary tasks')

    args = parser.parse_args([])
    return args

# Parse Arguments and Set Device
args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def run_evaluation():
    # === Setup
    set_seeds(args)
    device = set_devices(args)
    logger = Logger(args)
    logger.loss = 0

    print("Project name:", args.project_name)
    print("Using device:", device)

    # === Reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    # === Load Data
    train_loader, val_loader, test_loader, _, _, _ = get_data_preprocessed(args)
    model_class = get_detector_model(args)
    model = model_class(args, device).to(device)

    # === Load checkpoint
    ckpt_path = os.path.join(f'best_model.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    print("✅ Loaded model from checkpoint:", ckpt_path)

    # === Evaluation Mode
    model.eval()
    logger.evaluator = Evaluator(args)
    logger.evaluator.reset()
    criterion = nn.CrossEntropyLoss(reduction='none')
    scheduler = None
    optimizer = None

    print("🧪 Beginning test evaluation...")

    iteration = 0
    with torch.no_grad():
        for test_batch in tqdm(test_loader, total=len(test_loader), desc="Evaluating"):
            test_x, test_y, seq_lengths, signal_name_list = test_batch
            test_x = test_x.to(device)
            iteration += 1

            model, _ = sliding_window_v1(
                args=args,
                iteration=iteration,
                train_x=test_x,
                train_y=test_y,
                seq_lengths=seq_lengths,
                model=model,
                logger=logger,
                device=device,
                scheduler=scheduler,
                optimizer=optimizer,
                criterion=criterion,
                signal_name_list=signal_name_list,
                flow_type="test"
            )

    logger.test_result_only()

if __name__ == "__main__":
    run_evaluation()



