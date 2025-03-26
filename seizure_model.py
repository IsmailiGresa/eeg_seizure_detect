import os
from tqdm import tqdm
import random
import math
from data.data_preprocess import get_data_preprocessed
from models import get_detector_model
from utils.metrics import Evaluator
from utils.logger import Logger
from train.trainer import *
from utils.utils import set_seeds, set_devices
from utils.binary_performance_estimator import binary_detector_evaluator
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CNN2D_BiGRU Model')

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
    parser.add_argument('--seed_list', nargs='+', type=int, default=[42], help='List of seeds for evaluation')
    parser.add_argument('--margin_list', nargs='+', type=int, default=[3, 5], help='List of margins to test')
    parser.add_argument('--model', type=str, default='cnn_bigru_selfattention', help='Model file to use')
    parser.add_argument('--test_type', type=str, default='full', choices=['full', 'partial'], help='Test mode selector')
    parser.add_argument('--last', action='store_true', help='Use the last checkpoint')
    parser.add_argument('--best', action='store_true', help='Use the best checkpoint')

    args = parser.parse_args([])
    return args

args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
label_method_max = True
scheduler = None
optimizer = None
criterion = nn.CrossEntropyLoss(reduction='none')


def calc_hf(ref, hyp):

    start_r_a = ref[0]
    stop_r_a = ref[1]
    start_h_a = hyp[0]
    stop_h_a = hyp[1]

    ref_dur = stop_r_a - start_r_a
    hyp_dur = stop_h_a - start_h_a
    hit = float(0)
    fa = float(0)

    if start_h_a <= start_r_a and stop_h_a <= stop_r_a:
        hit = (stop_h_a - start_r_a) / ref_dur
        if ((start_r_a - start_h_a) / ref_dur) < 1.0:
            fa = ((start_r_a - start_h_a) / ref_dur)
        else:
            fa = float(1)

    elif start_h_a >= start_r_a and stop_h_a >= stop_r_a:

        hit = (stop_r_a - start_h_a) / ref_dur
        if ((stop_h_a - stop_r_a) / ref_dur) < 1.0:
            fa = ((stop_h_a - stop_r_a) / ref_dur)
        else:
            fa = float(1)

    elif start_h_a < start_r_a and stop_h_a > stop_r_a:

        hit = 1.0
        fa = ((stop_h_a - stop_r_a) + (start_r_a - start_h_a)) / \
             ref_dur
        if fa > 1.0:
            fa = float(1)

    else:
        hit = (stop_h_a - start_h_a) / ref_dur

    return (hit, fa)


def anyovlp(ref, hyp):

    refset = set(range(int(ref[0]), int(ref[1]) + 1))
    hypset = set(range(int(hyp[0]), int(hyp[1]) + 1))

    if len(refset.intersection(hypset)) != 0:
        return True

    return False


def ovlp_hyp_seqs(ref, hyp, rind, hind, refflag, hypflag):

    p_miss = float(0)

    p_hit, p_fa = calc_hf(ref[rind], hyp[hind])
    p_miss += float(1) - p_hit

    refflag[rind] = False
    hypflag[hind] = False

    hind += 1

    for i in range(hind, len(hyp)):

        if anyovlp(ref[rind], hyp[i]):

            hypflag[i] = False

            ovlp_hit, ovlp_fa \
                = calc_hf(ref[rind], hyp[i])

            p_hit += ovlp_hit
            p_miss -= ovlp_hit
            p_fa += ovlp_fa

        i += 1

    return p_hit, p_miss, p_fa


def ovlp_ref_seqs(ref, hyp, rind, hind, refflag, hypflag):

    p_miss = float(0)

    p_hit, p_fa = calc_hf(ref[rind], hyp[hind])
    p_miss += float(1) - p_hit

    hypflag[hind] = False
    refflag[rind] = False

    rind += 1

    for i in range(rind, len(ref)):

        if anyovlp(ref[i], hyp[hind]):

            refflag[i] = False
            p_miss += 1

        i += 1

    return p_hit, p_miss, p_fa


def compute_partial(ref, hyp, rind, hind, rflags, hflags):

    if not anyovlp(ref[rind], hyp[hind]):
        return (float(0), float(0), float(0))

    elif float(hyp[hind][1]) >= float(ref[rind][1]):

        p_hit, p_mis, p_fal = ovlp_ref_seqs(ref, hyp, rind, hind, rflags, hflags)

    elif float(ref[rind][1]) > float(hyp[hind][1]):

        p_hit, p_mis, p_fal \
            = ovlp_hyp_seqs(ref, hyp, rind, hind, rflags, hflags)

    return (p_hit, p_mis, p_fal)


def taes_get_events(start, stop, events_a, hflags):

    labels = []
    starts = []
    stops = []
    flags = []
    ind = []

    for i in range(len(events_a)):

        if (events_a[i][1] > start) and (events_a[i][0] < stop):
            starts.append(events_a[i][0])
            stops.append(events_a[i][1])
            labels.append(1)
            ind.append(i)
            flags.append(hflags[i])

    return [labels, starts, stops]


def ovlp_get_events(start, stop, events):

    labels = []
    starts = []
    stops = []

    for event in events:

        if (event[1] > start) and (event[0] < stop):
            starts.append(event[0])
            stops.append(event[1])
            labels.append(1)
    return [labels, starts, stops]


def ovlp_get_events_with_latency(start, stop, events):

    labels = []
    starts = []
    stops = []
    latencies = []
    not_detected = 0

    for event in events:

        if (event[1] > start) and (event[0] < stop):
            starts.append(event[0])
            stops.append(event[1])
            labels.append(1)

        if (event[0] >= start - 2) and (event[0] <= stop + 5):
            delayed_time = start - event[0]
            if delayed_time < 0:
                delayed_time = 0
            latencies.append(delayed_time)

        if (event[0] < start) and (event[1] > start):
            latencies.append(0)

    if len(latencies) == 0:
        latencies.append(stop - start)
        not_detected = 1
    else:
        not_detected = -1
    return [labels, starts, stops, min(latencies), not_detected]


def taes(ref_events, hyp_events):
    hit = 0
    mis = 0
    fal = 0
    refo = 0
    hypo = 0
    i = 0
    j = 0
    hflags = []
    rflags = []
    for i in range(len(hyp_events)):
        hflags.append(True)
    for i in range(len(ref_events)):
        rflags.append(True)

    for i, event in enumerate(ref_events):
        refo += 1
        labels, starts, stops = taes_get_events(event[0], event[1], hyp_events, hflags)

        if rflags[i]:

            for j in range(len(hyp_events)):

                if hflags[j]:

                    p_hit, p_miss, p_fa = compute_partial(ref_events, hyp_events, i, j, rflags, hflags)

                    hit += p_hit
                    mis += p_miss
                    fal += p_fa

                j += 1

        i += 1

    return hit, mis, fal, i, j


def ovlp(ref_events, hyp_events):
    hit = 0
    mis = 0
    fal = 0
    refo = 0
    hypo = 0
    latency_time = 0
    refo_minus_count = 0
    latency_time_of_detected = 0

    for event in ref_events:
        refo += 1
        labels, starts, stops, delayed_time, not_detected = ovlp_get_events_with_latency(event[0], event[1], hyp_events)
        if 1 in labels:
            hit += 1
        else:
            mis += 1
        latency_time += delayed_time
        if not_detected == 1:
            refo_minus_count += 1
        else:
            latency_time_of_detected += delayed_time

    for event in hyp_events:
        hypo += 1
        labels, starts, stops = ovlp_get_events(event[0], event[1], ref_events)
        if 1 not in labels:
            fal += 1

    return hit, mis, fal, refo, hypo, latency_time, latency_time_of_detected, refo_minus_count


ovlp_tprs_seeds = []
ovlp_tnrs_seeds = []
ovlp_fas24_seeds = []

taes_tprs_seeds = []
taes_tnrs_seeds = []
taes_fas24_seeds = []

latencies_seeds = []
detected_latencies_seeds = []
missed_for_latency_seeds = []
refos_for_latency_seeds = []

margin_3sec_rise_seeds = []
margin_3sec_fall_seeds = []
margin_5sec_rise_seeds = []
margin_5sec_fall_seeds = []

for seed_num in args.seed_list:
    args.seed = seed_num
    iteration = 1
    set_seeds(args)
    device = set_devices(args)
    logger = Logger(args)
    logger.loss = 0
    print("Project name is: ", args.project_name)

    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_loader, val_loader, test_loader, len_train_dir, len_val_dir, len_test_dir = get_data_preprocessed(args)
    model_class = get_detector_model(args)
    model = model_class(args, device).to(device)
    evaluator = Evaluator(args)
    name = args.project_name
    ckpt_path = "best_model.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ Checkpoint file not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt
    model.load_state_dict(state)
    model.eval()
    print('loaded model')
    print("Test type is: ", args.test_type)
    evaluator.reset()
    result_list = []
    evaluator.seizure_wise_eval_for_binary = True

    hyps = []
    hyps_list = []
    refs = []
    count = -1
    with torch.no_grad():
        for test_batch in tqdm(test_loader, total=len(test_loader),
                               bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            count += 1
            test_x, test_y, seq_lengths, _ = test_batch
            test_x = test_x.to(device)

            iter_num = math.ceil(test_x.shape[1] / 6000)
            signal_len = test_x.shape[1]
            label_len = test_y.shape[1]
            for iter_idx in range(iter_num):
                sig_start = iter_idx * 6000
                lable_start = iter_idx * 1500

                if 6000 < (signal_len - sig_start):
                    sig_end = sig_start + 6000
                    label_end = lable_start + 1500
                else:
                    sig_end = signal_len
                    label_end = label_len
                    if sig_end - sig_start < 400:
                        print(f"⚠️ Skipping segment: {sig_end - sig_start} samples")
                        continue

                test_x_sliced = test_x[:, sig_start:sig_end, :]
                test_y_sliced = test_y[:, lable_start:label_end]
                seq_lengths = [sig_end - sig_start]
                target_lengths = [label_end - lable_start]

                model, _ = sliding_window_v1(args, iteration, test_x_sliced, test_y_sliced, seq_lengths,
                                             target_lengths, model, logger, device, scheduler,
                                             optimizer, criterion, flow_type="test")

            if logger.pred_results:
                print(f"📊 logger.pred_results collected: {len(logger.pred_results)}")
                hyps.append(torch.stack(logger.pred_results).numpy()[:, 1])
            else:
                print("⚠️ No predictions were generated. Check your model, data, or pipeline.")
            refs.append(logger.ans_results)

            logger.pred_results = []
            logger.ans_results = []

        logger.test_result_only()

    hyps_list = [list(hyp) for hyp in hyps]

    print("##### margin test evaluation #####")
    target_stack = torch.tensor([item for sublist in refs for item in sublist])
    thresholds_margintest = list(logger.evaluator.thresholds_margintest)
    print("thresholds_margintest: ", thresholds_margintest)
    margin_threshold = 0
    for margin in args.margin_list:
        for threshold_idx, threshold in enumerate(thresholds_margintest):
            hyp_output = list([[int(hyp_step > threshold) for hyp_step in hyp_one] for hyp_one in hyps_list])
            pred_stack = torch.tensor(list([item for sublist in hyp_output for item in sublist]))
            margin_threshold = threshold

            pred_stack2 = pred_stack.unsqueeze(1)
            target_stack2 = target_stack.unsqueeze(1)
            rise_true, rise_pred_correct, fall_true, fall_pred_correct = binary_detector_evaluator(pred_stack2,
                                                                                                   target_stack2,
                                                                                                   margin)
            print("Margin: {}, Threshold: {}, TPR: {}, TNR: {}".format(str(margin), str(threshold),
                                                                       str(logger.evaluator.picked_tprs[threshold_idx]),
                                                                       str(logger.evaluator.picked_tnrs[
                                                                               threshold_idx])))
            print("rise_accuarcy:{}, fall_accuracy:{}".format(
                str(np.round((rise_pred_correct / float(rise_true)), decimals=4)),
                str(np.round((fall_pred_correct / float(fall_true)), decimals=4))))
            if margin == 3:
                margin_3sec_rise_seeds.append(np.round((rise_pred_correct / float(rise_true)), decimals=4))
                margin_3sec_fall_seeds.append(np.round((fall_pred_correct / float(fall_true)), decimals=4))

            if margin == 5:
                margin_5sec_rise_seeds.append(np.round((rise_pred_correct / float(rise_true)), decimals=4))
                margin_5sec_fall_seeds.append(np.round((fall_pred_correct / float(fall_true)), decimals=4))

    ref_events = []
    t_dur = 0
    for ref in refs:
        ref.insert(0, 0)
        ref.insert(len(ref), 0)
        ref_diff = np.array(ref) - np.array([ref[0]] + ref[:-1])
        starts = np.where(ref_diff == 1)[0]
        ends = np.where(ref_diff == -1)[0]

        if (len(starts) == 0) and (len(ends) == 0):
            ref_events.append(list())
        else:
            ref_events.append([(starts[idx] - 1, ends[idx] - 1) for idx in range(len(starts))])
        t_dur += len(ref)
        t_dur += 3

    hyps_list = [list(hyp) for hyp in hyps]

    threshold_num = 500

    tprs = []
    tnrs = []
    fprs = []
    fas = []
    latency_times = []
    detected_latency_times = []
    latency_0_95 = 0
    print("##### OVLP evaluation #####")
    for i in range(1, threshold_num):
        hyp_events = []
        threshold = float(round((1.0 / threshold_num) * i, 3))
        hyp_output = [[int(hyp_step > threshold) for hyp_step in hyp_one] for hyp_one in hyps_list]
        if threshold == 0.95:
            latency_0_95_threshold_idx = i - 1
        for hyp_element in hyp_output:
            hyp_element.insert(0, 0)
            hyp_element.insert(len(hyp_element), 0)
            hyp_diff = np.array(hyp_element) - np.array([hyp_element[0]] + hyp_element[:-1])
            starts = np.where(hyp_diff == 1)[0]
            ends = np.where(hyp_diff == -1)[0]

            if (len(starts) == 0) and (len(ends) == 0):
                hyp_events.append(list())
            else:
                hyp_events.append([(starts[idx] - 1, ends[idx] - 1) for idx in range(len(starts))])

        hit_t = 0
        mis_t = 0
        fal_t = 0
        refo_t = 0
        hypo_t = 0
        latency = 0
        detected_latency = 0
        refo_minus = 0
        for k in range(len(ref_events)):
            hit, mis, fal, refo, hypo, delayed_time, latency_time_of_detected, refo_minus_count = ovlp(ref_events[k],
                                                                                                       hyp_events[k])
            hit_t += hit
            mis_t += mis
            fal_t += fal
            refo_t += refo
            hypo_t += hypo
            latency += delayed_time
            detected_latency += latency_time_of_detected
            refo_minus += refo_minus_count

        if refo_t == 0:
            tprs.append(1)
        else:
            tprs.append(float(hit_t) / refo_t)
        if hypo_t == 0:
            tnrs.append(0)
        else:
            tnrs.append(1 - (float(fal_t) / hypo_t))
        if hypo_t == 0:
            fprs.append(1)
        else:
            fprs.append(float(fal_t) / hypo_t)
        fas.append(fal_t)
        latency_times.append(latency)
        detected_latency_times.append((detected_latency, refo_minus))

    best_threshold = np.argmax(np.array(tprs) + np.array(tnrs))
    fa_24_hours = (float(fas[best_threshold]) / t_dur) * (60 * 60 * 24)
    print("Best sensitivity: ", tprs[best_threshold])
    print("Best specificity: ", tnrs[best_threshold])
    print("Best FA/24hrs: ", fa_24_hours)

    ovlp_tprs_seeds.append(tprs[best_threshold])
    ovlp_tnrs_seeds.append(tnrs[best_threshold])
    ovlp_fas24_seeds.append(fa_24_hours)

    print("##### latency evaluation #####")
    print("Latency in second: ", float(latency_times[latency_0_95_threshold_idx]) / refo_t)
    if (refo_t - detected_latency_times[latency_0_95_threshold_idx][1]) != 0:
        print("Detected Latency in second: ", float(detected_latency_times[latency_0_95_threshold_idx][0]) / (
                    refo_t - detected_latency_times[latency_0_95_threshold_idx][1]))
        print("Detected Latency: {}, Missed Events: {}/{}".format(
            str(detected_latency_times[latency_0_95_threshold_idx][0]),
            str(detected_latency_times[latency_0_95_threshold_idx][1]), str(refo_t)))

    latencies_seeds.append(float(latency_times[latency_0_95_threshold_idx]) / refo_t)
    if (refo_t - detected_latency_times[latency_0_95_threshold_idx][1]) != 0:
        detected_latencies_seeds.append(float(detected_latency_times[latency_0_95_threshold_idx][0]) / (
                    refo_t - detected_latency_times[latency_0_95_threshold_idx][1]))
    else:
        detected_latencies_seeds.append(0)
    missed_for_latency_seeds.append(detected_latency_times[latency_0_95_threshold_idx][1])
    refos_for_latency_seeds.append(refo_t)

    tprs = []
    tnrs = []
    fprs = []
    fas = []
    print("##### TAES evaluation #####")
    for i in range(1, threshold_num):
        hyp_events = []
        threshold = float(round((1.0 / threshold_num) * i, 3))
        hyp_output = [[int(hyp_step > threshold) for hyp_step in hyp_one] for hyp_one in hyps_list]

        for hyp_element in hyp_output:
            hyp_element.insert(0, 0)
            hyp_element.insert(len(hyp_element), 0)
            hyp_diff = np.array(hyp_element) - np.array([hyp_element[0]] + hyp_element[:-1])
            starts = np.where(hyp_diff == 1)[0]
            ends = np.where(hyp_diff == -1)[0]

            if (len(starts) == 0) and (len(ends) == 0):
                hyp_events.append(list())
            else:
                hyp_events.append([(starts[idx] - 1, ends[idx] - 1) for idx in range(len(starts))])

        hit_t = 0
        mis_t = 0
        fal_t = 0
        refo_t = 0
        hypo_t = 0
        for k in range(len(ref_events)):
            hit, mis, fal, refo, hypo = taes(ref_events[k], hyp_events[k])
            hit_t += hit
            mis_t += mis
            fal_t += fal
            refo_t += refo
            hypo_t += hypo

        if refo_t == 0:
            tprs.append(1)
        else:
            tprs.append(float(hit_t) / refo_t)
        if hypo_t == 0:
            tnrs.append(0)
        else:
            tnrs.append(1 - (float(fal_t) / hypo_t))
        if hypo_t == 0:
            fprs.append(1)
        else:
            fprs.append(float(fal_t) / hypo_t)
        fas.append(fal_t)

    best_threshold = np.argmax(np.array(tprs) + np.array(tnrs))
    fa_24_hours = (float(fas[best_threshold]) / t_dur) * (60 * 60 * 24)
    print("Best sensitivity: ", tprs[best_threshold])
    print("Best specificity: ", tnrs[best_threshold])
    print("Best FA/24hrs: ", fa_24_hours)
    taes_tprs_seeds.append(tprs[best_threshold])
    taes_tnrs_seeds.append(tnrs[best_threshold])
    taes_fas24_seeds.append(fa_24_hours)

os.system("echo  \'#######################################\'")
os.system("echo  \'##### Final test results per seed #####\'")
os.system("echo  \'#######################################\'")
os.system("echo  \'Total average -- ovlp_tpr: {}, ovlp_tnr: {}, ovlp_fas24: {}\'".format(str(np.mean(ovlp_tprs_seeds)),
                                                                                         str(np.mean(ovlp_tnrs_seeds)),
                                                                                         str(np.mean(
                                                                                             ovlp_fas24_seeds))))
os.system("echo  \'Total std -- ovlp_tpr: {}, ovlp_tnr: {}, ovlp_fas24: {}\'".format(str(np.std(ovlp_tprs_seeds)),
                                                                                     str(np.std(ovlp_tnrs_seeds)),
                                                                                     str(np.std(ovlp_fas24_seeds))))

os.system("echo  \'Total average -- taes_tpr: {}, taes_tnr: {}, taes_fas24: {}\'".format(str(np.mean(taes_tprs_seeds)),
                                                                                         str(np.mean(taes_tnrs_seeds)),
                                                                                         str(np.mean(
                                                                                             taes_fas24_seeds))))
os.system("echo  \'Total std -- taes_tpr: {}, taes_tnr: {}, taes_fas24: {}\'".format(str(np.std(taes_tprs_seeds)),
                                                                                     str(np.std(taes_tnrs_seeds)),
                                                                                     str(np.std(taes_fas24_seeds))))

os.system(
    "echo  \'Total average -- latnecy: {}, d_latency: {}, missed: {}, refos: {}\'".format(str(np.mean(latencies_seeds)),
                                                                                          str(np.mean(
                                                                                              detected_latencies_seeds)),
                                                                                          str(np.mean(
                                                                                              missed_for_latency_seeds)),
                                                                                          str(np.mean(
                                                                                              refos_for_latency_seeds))))
os.system(
    "echo  \'Total std -- latnecy: {}, d_latency: {}, missed: {}, refos: {}\'".format(str(np.std(latencies_seeds)),
                                                                                      str(np.std(
                                                                                          detected_latencies_seeds)),
                                                                                      str(np.std(
                                                                                          missed_for_latency_seeds)),
                                                                                      str(np.std(
                                                                                          refos_for_latency_seeds))))

os.system("echo  \'Total average -- 3sec_rise: {}, 3sec_fall: {}, 5sec_rise: {}, 5sec_fall: {}\'".format(
    str(np.mean(margin_3sec_rise_seeds)), str(np.mean(margin_3sec_fall_seeds)), str(np.mean(margin_5sec_rise_seeds)),
    str(np.mean(margin_5sec_fall_seeds))))
os.system("echo  \'Total std -- 3sec_rise: {}, 3sec_fall: {}, 5sec_rise: {}, 5sec_fall: {}\'".format(
    str(np.std(margin_3sec_rise_seeds)), str(np.std(margin_3sec_fall_seeds)), str(np.std(margin_5sec_rise_seeds)),
    str(np.std(margin_5sec_fall_seeds))))
