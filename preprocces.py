from pyedflib import highlevel
from scipy import signal as sci_sig
from utils.process_util import run_multi_process
from utils.utils import search_walk
import numpy as np
import os
import argparse
import torch
import pickle
import random
import ast
from itertools import groupby
from functools import partial


GLOBAL_DATA = {
    'disease_labels': {
        'bckg': 0,
        'fnsz': 1,
        'gnsz': 2,
        'cpsz': 3,
        'absz': 4,
        'tcsz': 5,
        'mysz': 6,
        'atsz': 7
    }
}
label_dict = {}
sample_rate_dict = {}
sev_label = {}

def label_sampling_tuh(labels, feature_samplerate):
    y_target = ""
    remained = 0
    feature_intv = 1 / float(feature_samplerate)

    for i in labels:
        i = i.strip()
        if i == "" or i.lower().startswith("channel") or i.lower().startswith("start_time"):
            continue

        parts = i.split(",")
        if len(parts) < 4:
            print(f"⚠️ Skipping malformed line (too short): {i}")
            continue

        try:
            begin = float(parts[1])
            end = float(parts[2])
            label = parts[3].strip()

            intv_count, remained = divmod(end - begin + remained, feature_intv)
            y_target += int(intv_count) * str(GLOBAL_DATA['disease_labels'][label])
        except Exception as e:
            print(f"⚠️ Skipping malformed line: {i} — {e}")
            continue

    return y_target


def generate_training_data_leadwise_tuh_train(file):
    sample_rate = GLOBAL_DATA['sample_rate']
    file_name = ".".join(
        file.split(".")[:-1])
    data_file_name = file_name.split("/")[-1]
    signals, signal_headers, header = highlevel.read_edf(file)
    label_list_c = []
    for idx, signal in enumerate(signals):
        label_noref = signal_headers[idx]['label'].split("-")[0]
        label_list_c.append(label_noref)

    label_file = open(file_name + "." + GLOBAL_DATA['label_type'], 'r')
    y = label_file.readlines()
    y = list(y[2:])
    y_cleaned = []
    for i in y:
        i = i.strip()
        if i == "":
            continue
        parts = i.split()
        if len(parts) < 3:
            continue
        y_cleaned.append(parts[2])

    y_labels = list(set(y_cleaned))
    signal_sample_rate = int(signal_headers[0]['sample_rate'])
    if sample_rate > signal_sample_rate:
        return
    if not all(elem in label_list_c for elem in
               GLOBAL_DATA['label_list']):
        return

    y_sampled = label_sampling_tuh(y, GLOBAL_DATA['feature_sample_rate'])

    signal_list = []
    signal_label_list = []
    signal_final_list_raw = []

    for idx, signal in enumerate(signals):
        label = signal_headers[idx]['label'].split("-")[0]
        if label not in GLOBAL_DATA['label_list']:
            continue

        if int(signal_headers[idx]['sample_rate']) > sample_rate:
            secs = len(signal) / float(signal_sample_rate)
            samps = int(secs * sample_rate)
            x = sci_sig.resample(signal, samps)
            signal_list.append(x)
            signal_label_list.append(label)
        else:
            signal_list.append(signal)
            signal_label_list.append(label)

    if len(signal_label_list) != len(GLOBAL_DATA['label_list']):
        print("Not enough labels: ", signal_label_list)
        return

    for lead_signal in GLOBAL_DATA['label_list']:
        signal_final_list_raw.append(signal_list[signal_label_list.index(lead_signal)])

    new_length = len(signal_final_list_raw[0]) * (
                float(GLOBAL_DATA['feature_sample_rate']) / GLOBAL_DATA['sample_rate'])

    if len(y_sampled) > new_length:
        y_sampled = y_sampled[:new_length]
    elif len(y_sampled) < new_length:
        if len(y_sampled) == 0:
            print(f"⚠️ Skipping file {file_name}: empty y_sampled (likely no valid labels)")
            return
        diff = int(new_length - len(y_sampled))
        y_sampled += y_sampled[-1] * diff

    y_sampled_np = np.array(list(map(int, y_sampled)))
    new_labels = []
    new_labels_idxs = []

    y_sampled = ["0" if l not in GLOBAL_DATA['selected_diseases'] else l for l in y_sampled]

    if any(l in GLOBAL_DATA['selected_diseases'] for l in y_sampled):
        y_sampled = [str(GLOBAL_DATA['target_dictionary'][int(l)]) if l in GLOBAL_DATA['selected_diseases'] else l for l
                     in y_sampled]

    new_data = {}
    raw_data = torch.Tensor(signal_final_list_raw).permute(1, 0)

    max_seg_len_before_seiz_label = GLOBAL_DATA['max_bckg_before_slicelength'] * GLOBAL_DATA['feature_sample_rate']
    max_seg_len_before_seiz_raw = GLOBAL_DATA['max_bckg_before_slicelength'] * GLOBAL_DATA['sample_rate']
    max_seg_len_after_seiz_label = GLOBAL_DATA['max_bckg_after_seiz_length'] * GLOBAL_DATA['feature_sample_rate']
    max_seg_len_after_seiz_raw = GLOBAL_DATA['max_bckg_after_seiz_length'] * GLOBAL_DATA['sample_rate']

    min_seg_len_label = GLOBAL_DATA['min_binary_slicelength'] * GLOBAL_DATA['feature_sample_rate']
    min_seg_len_raw = GLOBAL_DATA['min_binary_slicelength'] * GLOBAL_DATA['sample_rate']
    max_seg_len_label = GLOBAL_DATA['max_binary_slicelength'] * GLOBAL_DATA['feature_sample_rate']
    max_seg_len_raw = GLOBAL_DATA['max_binary_slicelength'] * GLOBAL_DATA['sample_rate']

    label_order = [x[0] for x in groupby(y_sampled)]
    label_change_idxs = np.where(y_sampled_np[:-1] != y_sampled_np[1:])[0]

    start_raw_idx = 0
    start_label_idx = 0
    end_raw_idx = raw_data.size(0)
    end_label_idx = len(y_sampled)
    previous_bckg_len = 0

    sliced_raws = []
    sliced_labels = []
    pre_bckg_lens_label = []
    label_list_for_filename = []

    for idx, label in enumerate(label_order):
        if (len(label_order) == idx + 1) and (label == "0"):
            sliced_raw_data = raw_data[start_raw_idx:].permute(1, 0)
            sliced_y1 = torch.Tensor(list(map(int, y_sampled[start_label_idx:]))).byte()

            if sliced_y1.size(0) < min_seg_len_label:
                continue
            sliced_raws.append(sliced_raw_data)
            sliced_labels.append(sliced_y1)
            pre_bckg_lens_label.append(0)
            label_list_for_filename.append(label)

        elif (len(label_order) != idx + 1) and (label == "0"):
            end_raw_idx = (label_change_idxs[idx] + 1) * GLOBAL_DATA['fsr_sr_ratio']
            end_label_idx = label_change_idxs[idx] + 1

            sliced_raw_data = raw_data[start_raw_idx:end_raw_idx].permute(1, 0)
            sliced_y1 = torch.Tensor(list(map(int, y_sampled[start_label_idx:end_label_idx]))).byte()
            previous_bckg_len = end_label_idx - start_label_idx

            start_raw_idx = end_raw_idx
            start_label_idx = end_label_idx
            if sliced_y1.size(0) < min_seg_len_label:
                continue

            sliced_raws.append(sliced_raw_data)
            sliced_labels.append(sliced_y1)
            pre_bckg_lens_label.append(0)
            label_list_for_filename.append(label)

        elif (idx == 0) and (label != "0"):
            end_raw_idx = (label_change_idxs[idx] + 1) * GLOBAL_DATA['fsr_sr_ratio']
            end_label_idx = label_change_idxs[idx] + 1

            if len(y_sampled) - end_label_idx > max_seg_len_after_seiz_label:
                post_len_label = max_seg_len_after_seiz_label
                post_len_raw = max_seg_len_after_seiz_raw
            else:
                post_len_label = len(y_sampled) - end_label_idx
                post_len_raw = ((len(y_sampled) - end_label_idx) * GLOBAL_DATA['fsr_sr_ratio'])
            post_ictal_end_label = end_label_idx + post_len_label
            post_ictal_end_raw = end_raw_idx + post_len_raw

            start_raw_idx = end_raw_idx
            start_label_idx = end_label_idx
            if len(y_sampled) < min_seg_len_label:
                continue

            sliced_raw_data = raw_data[:post_ictal_end_raw].permute(1, 0)
            sliced_y1 = torch.Tensor(list(map(int, y_sampled[:post_ictal_end_label]))).byte()

            if sliced_y1.size(0) > max_seg_len_label:
                sliced_y2 = sliced_y1[:max_seg_len_label]
                sliced_raw_data2 = sliced_raw_data.permute(1, 0)[:max_seg_len_raw].permute(1, 0)
                sliced_raws.append(sliced_raw_data2)
                sliced_labels.append(sliced_y2)
                pre_bckg_lens_label.append(0)
                label_list_for_filename.append(label)
            elif sliced_y1.size(0) >= min_seg_len_label:
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y1)
                pre_bckg_lens_label.append(0)
                label_list_for_filename.append(label)
            else:
                sliced_y2 = torch.Tensor(list(map(int, y_sampled[:min_seg_len_label]))).byte()
                sliced_raw_data2 = raw_data[:min_seg_len_raw].permute(1, 0)
                sliced_raws.append(sliced_raw_data2)
                sliced_labels.append(sliced_y2)
                pre_bckg_lens_label.append(0)
                label_list_for_filename.append(label)

        elif label != "0":
            end_raw_idx = (label_change_idxs[idx] + 1) * GLOBAL_DATA['fsr_sr_ratio']
            end_label_idx = label_change_idxs[idx] + 1

            if len(y_sampled) - end_label_idx > max_seg_len_after_seiz_label:
                post_len_label = max_seg_len_after_seiz_label
                post_len_raw = max_seg_len_after_seiz_raw
            else:
                post_len_label = len(y_sampled) - end_label_idx
                post_len_raw = ((len(y_sampled) - end_label_idx) * GLOBAL_DATA['fsr_sr_ratio'])
            post_ictal_end_label = end_label_idx + post_len_label
            post_ictal_end_raw = end_raw_idx + post_len_raw

            if previous_bckg_len > max_seg_len_before_seiz_label:
                pre_seiz_label_len = max_seg_len_before_seiz_label
            else:
                pre_seiz_label_len = previous_bckg_len
            pre_seiz_raw_len = pre_seiz_label_len * GLOBAL_DATA['fsr_sr_ratio']

            sample_len = post_ictal_end_label - (start_label_idx - pre_seiz_label_len)
            if sample_len < min_seg_len_label:
                post_ictal_end_label = start_label_idx - pre_seiz_label_len + min_seg_len_label
                post_ictal_end_raw = start_raw_idx - pre_seiz_raw_len + min_seg_len_raw
            if len(y_sampled) < post_ictal_end_label:
                start_raw_idx = end_raw_idx
                start_label_idx = end_label_idx
                continue

            sliced_raw_data = raw_data[start_raw_idx - pre_seiz_raw_len:post_ictal_end_raw].permute(1, 0)
            sliced_y1 = torch.Tensor(
                list(map(int, y_sampled[start_label_idx - pre_seiz_label_len:post_ictal_end_label]))).byte()

            if sliced_y1.size(0) > max_seg_len_label:
                sliced_y2 = sliced_y1[:max_seg_len_label]
                sliced_raw_data2 = sliced_raw_data.permute(1, 0)[:max_seg_len_raw].permute(1, 0)
                sliced_raws.append(sliced_raw_data2)
                sliced_labels.append(sliced_y2)
                pre_bckg_lens_label.append(pre_seiz_label_len)
                label_list_for_filename.append(label)
            else:
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y1)
                pre_bckg_lens_label.append(pre_seiz_label_len)
                label_list_for_filename.append(label)
            start_raw_idx = end_raw_idx
            start_label_idx = end_label_idx

        else:
            print("Error! Impossible!")
            exit(1)

    for data_idx in range(len(sliced_raws)):
        sliced_raw = sliced_raws[data_idx]
        sliced_y = sliced_labels[data_idx]
        sliced_y_map = list(map(int, sliced_y))

        if GLOBAL_DATA['binary_target1'] is not None:
            sliced_y2 = torch.Tensor([GLOBAL_DATA['binary_target1'][i] for i in sliced_y_map]).byte()
        else:
            sliced_y2 = None

        if GLOBAL_DATA['binary_target2'] is not None:
            sliced_y3 = torch.Tensor([GLOBAL_DATA['binary_target2'][i] for i in sliced_y_map]).byte()
        else:
            sliced_y3 = None

        new_data['RAW_DATA'] = [sliced_raw]
        new_data['LABEL1'] = [sliced_y]
        new_data['LABEL2'] = [sliced_y2]
        new_data['LABEL3'] = [sliced_y3]

        prelabel_len = pre_bckg_lens_label[data_idx]
        label = label_list_for_filename[data_idx]

        with open(GLOBAL_DATA['data_file_directory'] + "/{}_c{}_pre{}_len{}_label_{}.pkl".format(data_file_name,
                                                                                                 str(data_idx),
                                                                                                 str(prelabel_len),
                                                                                                 str(len(sliced_y)),
                                                                                                 str(label)),
                  'wb') as _f:
            pickle.dump(new_data, _f)
        new_data = {}


def generate_training_data_leadwise_tuh_train_final(file, global_data):
    sample_rate = global_data['sample_rate']  # EX) 200Hz
    file_name = ".".join(
        file.split(".")[:-1])
    data_file_name = file_name.split("/")[-1]
    signals, signal_headers, header = highlevel.read_edf(file)
    label_list_c = []
    for idx, signal in enumerate(signals):
        label_noref = signal_headers[idx]['label'].split("-")[0]
        label_list_c.append(label_noref)

    label_file = open(file_name + "." + global_data['label_type'], 'r')
    y = label_file.readlines()
    y = list(y[2:])
    y_cleaned = []
    for i in y:
        i = i.strip()
        if i == "":
            continue
        parts = i.split()
        if len(parts) < 3:
            continue
        y_cleaned.append(parts[2])  # seizure label (e.g., gnsz)

    y_labels = list(set(y_cleaned))
    signal_sample_rate = int(signal_headers[0]['sample_rate'])
    if sample_rate > signal_sample_rate:
        return
    if not all(elem in label_list_c for elem in
               global_data['label_list']):
        return

    y_sampled = label_sampling_tuh(y, global_data['feature_sample_rate'])

    patient_wise_dir = "/".join(file_name.split("/")[:-2])
    patient_id = file_name.split("/")[-3]
    edf_list = search_walk({'path': patient_wise_dir, 'extension': ".csv_bi"})
    patient_bool = False
    for csv_bi_file in edf_list:
        label_file = open(csv_bi_file, 'r')
        y = label_file.readlines()
        y = list(y[2:])
        for line in y:
            if len(line) > 5:
                if line.split(" ")[2] != 'bckg':
                    patient_bool = True
                    break
        if patient_bool:
            break

    signal_list = []
    signal_label_list = []
    signal_final_list_raw = []

    for idx, signal in enumerate(signals):
        label = signal_headers[idx]['label'].split("-")[0]
        if label not in global_data['label_list']:
            continue

        if int(signal_headers[idx]['sample_rate']) > sample_rate:
            secs = len(signal) / float(signal_sample_rate)
            samps = int(secs * sample_rate)
            x = sci_sig.resample(signal, samps)
            signal_list.append(x)
            signal_label_list.append(label)
        else:
            signal_list.append(signal)
            signal_label_list.append(label)

    if len(signal_label_list) != len(global_data['label_list']):
        print("Not enough labels: ", signal_label_list)
        return

    for lead_signal in global_data['label_list']:
        signal_final_list_raw.append(signal_list[signal_label_list.index(lead_signal)])

    new_length = len(signal_final_list_raw[0]) * (
                float(global_data['feature_sample_rate']) / global_data['sample_rate'])

    if len(y_sampled) > new_length:
        y_sampled = y_sampled[:new_length]
    elif len(y_sampled) < new_length:
        diff = int(new_length - len(y_sampled))
        if len(y_sampled) > 0:
            y_sampled += y_sampled[-1] * diff
        else:
            print(f"⚠️ Skipping {file_name}: empty label sequence after sampling")
            return

    y_sampled_np = np.array(list(map(int, y_sampled)))
    new_labels = []
    new_labels_idxs = []

    y_sampled = ["0" if l not in global_data['selected_diseases'] else l for l in y_sampled]

    if any(l in global_data['selected_diseases'] for l in y_sampled):
        y_sampled = [str(global_data['target_dictionary'][int(l)]) if l in global_data['selected_diseases'] else l for l
                     in y_sampled]

    new_data = {}
    raw_data = torch.Tensor(signal_final_list_raw).permute(1, 0)
    raw_data = raw_data.type(torch.float16)

    min_seg_len_label = global_data['min_binary_slicelength'] * global_data['feature_sample_rate']
    min_seg_len_raw = global_data['min_binary_slicelength'] * global_data['sample_rate']
    min_binary_edge_seiz_label = global_data['min_binary_edge_seiz'] * global_data['feature_sample_rate']
    min_binary_edge_seiz_raw = global_data['min_binary_edge_seiz'] * global_data['sample_rate']

    label_order = [x[0] for x in groupby(y_sampled)]
    label_change_idxs = np.where(y_sampled_np[:-1] != y_sampled_np[1:])[0]
    label_change_idxs = np.append(label_change_idxs, np.array([len(y_sampled_np) - 1]))

    sliced_raws = []
    sliced_labels = []
    label_list_for_filename = []
    if len(y_sampled) < min_seg_len_label:
        return
    else:
        label_count = {}
        y_sampled_2nd = list(y_sampled)
        raw_data_2nd = raw_data
        while len(y_sampled) >= min_seg_len_label:
            is_at_middle = False
            sliced_y = y_sampled[:min_seg_len_label]
            labels = [x[0] for x in groupby(sliced_y)]

            if len(labels) == 1 and "0" in labels:
                y_sampled = y_sampled[min_seg_len_label:]
                sliced_raw_data = raw_data[:min_seg_len_raw].permute(1, 0)
                raw_data = raw_data[min_seg_len_raw:]
                if patient_bool:
                    label = "0_patT"
                else:
                    label = "0_patF"
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)
                label_list_for_filename.append(label)

            elif len(labels) != 1 and (sliced_y[0] == '0') and (sliced_y[-1] != '0'):
                temp_sliced_y = list(sliced_y)
                temp_sliced_y.reverse()
                boundary_seizlen = temp_sliced_y.index("0") + 1
                if boundary_seizlen < min_binary_edge_seiz_label:
                    if len(y_sampled) > (min_seg_len_label + min_binary_edge_seiz_label):
                        sliced_y = y_sampled[min_binary_edge_seiz_label:min_seg_len_label + min_binary_edge_seiz_label]
                        sliced_raw_data = raw_data[min_binary_edge_seiz_raw:min_seg_len_raw + min_binary_edge_seiz_raw].permute(1, 0)
                    else:
                        sliced_raw_data = raw_data[:min_seg_len_raw].permute(1, 0)
                else:
                    sliced_raw_data = raw_data[:min_seg_len_raw].permute(1, 0)

                y_sampled = y_sampled[min_seg_len_label:]
                raw_data = raw_data[min_seg_len_raw:]

                label = str(max(list(map(int, labels))))
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)

                label = label + "_beg"
                label_list_for_filename.append(label)
                is_at_middle = True

            elif (len(labels) != 1) and (sliced_y[0] != '0') and (sliced_y[-1] != '0'):
                y_sampled = y_sampled[min_seg_len_label:]
                sliced_raw_data = raw_data[:min_seg_len_raw].permute(1, 0)
                raw_data = raw_data[min_seg_len_raw:]

                label = str(max(list(map(int, labels))))
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)

                label = label + "_whole"
                label_list_for_filename.append(label)
                is_at_middle = True

            elif (len(labels) == 1) and (sliced_y[0] != '0') and (sliced_y[-1] != '0'):
                y_sampled = y_sampled[min_seg_len_label:]
                sliced_raw_data = raw_data[:min_seg_len_raw].permute(1, 0)
                raw_data = raw_data[min_seg_len_raw:]

                label = str(max(list(map(int, labels))))
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)

                label = label + "_middle"
                label_list_for_filename.append(label)
                is_at_middle = True

            elif len(labels) != 1 and (sliced_y[0] != '0') and (sliced_y[-1] == '0'):
                y_sampled = y_sampled[min_seg_len_label:]
                sliced_raw_data = raw_data[:min_seg_len_raw].permute(1, 0)
                raw_data = raw_data[min_seg_len_raw:]

                label = str(max(list(map(int, labels))))
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)

                label = label + "_end"
                label_list_for_filename.append(label)

            elif len(labels) != 1 and (sliced_y[0] == '0') and (sliced_y[-1] == '0'):
                y_sampled = y_sampled[min_seg_len_label:]
                sliced_raw_data = raw_data[:min_seg_len_raw].permute(1, 0)
                raw_data = raw_data[min_seg_len_raw:]

                label = str(max(list(map(int, labels))))
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)

                label = label + "_whole"
                label_list_for_filename.append(label)

            else:
                print("unexpected case")
                exit(1)
        if is_at_middle == True:
            sliced_y = y_sampled_2nd[-min_seg_len_label:]
            sliced_raw_data = raw_data_2nd[-min_seg_len_raw:].permute(1, 0)

            if sliced_y[-1] == '0':
                label = str(max(list(map(int, labels))))
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)

                label = label + "_end"
                label_list_for_filename.append(label)
            else:
                pass

    for data_idx in range(len(sliced_raws)):
        sliced_raw = sliced_raws[data_idx]
        sliced_y = sliced_labels[data_idx]
        sliced_y_map = list(map(int, sliced_y))
        sliced_y = torch.Tensor(sliced_y_map).byte()

        if GLOBAL_DATA['binary_target1'] is not None:
            sliced_y2 = torch.Tensor([global_data['binary_target1'][i] for i in sliced_y_map]).byte()
        else:
            sliced_y2 = None

        if global_data['binary_target2'] is not None:
            sliced_y3 = torch.Tensor([global_data['binary_target2'][i] for i in sliced_y_map]).byte()
        else:
            sliced_y3 = None

        new_data['RAW_DATA'] = [sliced_raw]
        new_data['LABEL1'] = [sliced_y]
        new_data['LABEL2'] = [sliced_y2]
        new_data['LABEL3'] = [sliced_y3]

        label = label_list_for_filename[data_idx]

        with open(global_data['data_file_directory'] + "/{}_c{}_label_{}.pkl".format(data_file_name, str(data_idx),
                                                                                     str(label)), 'wb') as _f:
            pickle.dump(new_data, _f)
        new_data = {}


def generate_training_data_leadwise_tuh_dev(file):
    sample_rate = GLOBAL_DATA['sample_rate']
    file_name = ".".join(
        file.split(".")[:-1])
    data_file_name = file_name.split("/")[-1]
    signals, signal_headers, header = highlevel.read_edf(file)
    label_list_c = []
    for idx, signal in enumerate(signals):
        label_noref = signal_headers[idx]['label'].split("-")[0]
        label_list_c.append(label_noref)

    label_file = open(file_name + "." + GLOBAL_DATA['label_type'],'r')
    y = label_file.readlines()
    y = list(y[2:])
    y_cleaned = []
    for i in y:
        i = i.strip()
        if i == "":
            continue
        parts = i.split(",")
        if len(parts) < 3:
            continue
        y_cleaned.append(parts[2])  # seizure label (e.g., gnsz)

    y_labels = list(set(y_cleaned))
    signal_sample_rate = int(signal_headers[0]['sample_rate'])
    if sample_rate > signal_sample_rate:
        return
    if not all(elem in label_list_c for elem in
               GLOBAL_DATA['label_list']):
        return

    y_sampled = label_sampling_tuh(y, GLOBAL_DATA['feature_sample_rate'])

    patient_wise_dir = "/".join(file_name.split("/")[:-2])
    edf_list = search_walk({'path': patient_wise_dir, 'extension': ".csv_bi"})
    patient_bool = False
    for csv_bi_file in edf_list:
        label_file = open(csv_bi_file, 'r')
        y = label_file.readlines()
        y = list(y[2:])
        for line in y:
            if len(line) > 5:
                if line.split(" ")[2] != 'bckg':
                    patient_bool = True
                    break
        if patient_bool:
            break

    signal_list = []
    signal_label_list = []
    signal_final_list_raw = []

    for idx, signal in enumerate(signals):
        label = signal_headers[idx]['label'].split("-")[0]
        if label not in GLOBAL_DATA['label_list']:
            continue

        if int(signal_headers[idx]['sample_rate']) > sample_rate:
            secs = len(signal) / float(signal_sample_rate)
            samps = int(secs * sample_rate)
            x = sci_sig.resample(signal, samps)
            signal_list.append(x)
            signal_label_list.append(label)
        else:
            signal_list.append(signal)
            signal_label_list.append(label)

    if len(signal_label_list) != len(GLOBAL_DATA['label_list']):
        print("Not enough labels: ", signal_label_list)
        return

    for lead_signal in GLOBAL_DATA['label_list']:
        signal_final_list_raw.append(signal_list[signal_label_list.index(lead_signal)])

    new_length = len(signal_final_list_raw[0]) * (
                float(GLOBAL_DATA['feature_sample_rate']) / GLOBAL_DATA['sample_rate'])

    if len(y_sampled) > new_length:
        y_sampled = y_sampled[:new_length]
    elif len(y_sampled) < new_length:
        diff = int(new_length - len(y_sampled))
        if len(y_sampled) > 0:
            y_sampled += y_sampled[-1] * diff
        else:
            print(f"⚠️ Skipping {file_name}: empty label sequence after sampling")
            return

    y_sampled_np = np.array(list(map(int, y_sampled)))
    new_labels = []
    new_labels_idxs = []

    y_sampled = ["0" if l not in GLOBAL_DATA['selected_diseases'] else l for l in y_sampled]

    if any(l in GLOBAL_DATA['selected_diseases'] for l in y_sampled):
        y_sampled = [str(GLOBAL_DATA['target_dictionary'][int(l)]) if l in GLOBAL_DATA['selected_diseases'] else l for l
                     in y_sampled]

    new_data = {}
    raw_data = torch.Tensor(signal_final_list_raw).permute(1, 0)
    raw_data = raw_data.type(torch.float16)

    min_seg_len_label = GLOBAL_DATA['min_binary_slicelength'] * GLOBAL_DATA['feature_sample_rate']
    min_seg_len_raw = GLOBAL_DATA['min_binary_slicelength'] * GLOBAL_DATA['sample_rate']

    sliced_raws = []
    sliced_labels = []
    label_list_for_filename = []

    if len(y_sampled) < min_seg_len_label:
        return
    else:
        label_count = {}
        while len(y_sampled) >= min_seg_len_label:
            one_left_slice = False
            sliced_y = y_sampled[:min_seg_len_label]

            if (sliced_y[-1] == '0'):
                sliced_raw_data = raw_data[:min_seg_len_raw].permute(1, 0)
                raw_data = raw_data[min_seg_len_raw:]
                y_sampled = y_sampled[min_seg_len_label:]

                labels = [x[0] for x in groupby(sliced_y)]
                if (len(labels) == 1) and (labels[0] == '0'):
                    label = "0"
                else:
                    label = ("".join(labels)).replace("0", "")[0]
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)
                label_list_for_filename.append(label)

            else:
                if '0' in y_sampled[min_seg_len_label:]:
                    end_1 = y_sampled[min_seg_len_label:].index('0')
                    temp_y_sampled = list(y_sampled[min_seg_len_label + end_1:])
                    temp_y_sampled_order = [x[0] for x in groupby(temp_y_sampled)]

                    if len(list(set(temp_y_sampled))) == 1:
                        end_2 = len(temp_y_sampled)
                        one_left_slice = True
                    else:
                        end_2 = temp_y_sampled.index(temp_y_sampled_order[1])

                    if end_2 >= min_end_margin_label:
                        temp_sec = random.randint(1, args.slice_end_margin_length)
                        temp_seg_len_label = int(min_seg_len_label + (temp_sec * args.feature_sample_rate) + end_1)
                        temp_seg_len_raw = int(
                            min_seg_len_raw + (temp_sec * args.samplerate) + (end_1 * GLOBAL_DATA['fsr_sr_ratio']))
                    else:
                        if one_left_slice:
                            temp_label = end_2
                        else:
                            temp_label = end_2 // 2

                        temp_seg_len_label = int(min_seg_len_label + temp_label + end_1)
                        temp_seg_len_raw = int(min_seg_len_raw + (temp_label * GLOBAL_DATA['fsr_sr_ratio']) + (
                                    end_1 * GLOBAL_DATA['fsr_sr_ratio']))

                    sliced_y = y_sampled[:temp_seg_len_label]
                    sliced_raw_data = raw_data[:temp_seg_len_raw].permute(1, 0)
                    raw_data = raw_data[temp_seg_len_raw:]
                    y_sampled = y_sampled[temp_seg_len_label:]

                    labels = [x[0] for x in groupby(sliced_y)]
                    if (len(labels) == 1) and (labels[0] == '0'):
                        label = "0"
                    else:
                        label = ("".join(labels)).replace("0", "")[0]
                    sliced_raws.append(sliced_raw_data)
                    sliced_labels.append(sliced_y)
                    label_list_for_filename.append(label)
                else:
                    sliced_y = y_sampled[:]
                    sliced_raw_data = raw_data[:].permute(1, 0)
                    raw_data = []
                    y_sampled = []

                    labels = [x[0] for x in groupby(sliced_y)]
                    if (len(labels) == 1) and (labels[0] == '0'):
                        label = "0"
                    else:
                        label = ("".join(labels)).replace("0", "")[0]
                    sliced_raws.append(sliced_raw_data)
                    sliced_labels.append(sliced_y)
                    label_list_for_filename.append(label)

    for data_idx in range(len(sliced_raws)):
        sliced_raw = sliced_raws[data_idx]
        sliced_y = sliced_labels[data_idx]
        sliced_y_map = list(map(int, sliced_y))

        if GLOBAL_DATA['binary_target1'] is not None:
            sliced_y2 = torch.Tensor([GLOBAL_DATA['binary_target1'][i] for i in sliced_y_map]).byte()
        else:
            sliced_y2 = None

        if GLOBAL_DATA['binary_target2'] is not None:
            sliced_y3 = torch.Tensor([GLOBAL_DATA['binary_target2'][i] for i in sliced_y_map]).byte()
        else:
            sliced_y3 = None

        new_data['RAW_DATA'] = [sliced_raw]
        new_data['LABEL1'] = [sliced_y]
        new_data['LABEL2'] = [sliced_y2]
        new_data['LABEL3'] = [sliced_y3]

        label = label_list_for_filename[data_idx]

        with open(
                GLOBAL_DATA['data_file_directory'] + "/{}_c{}_len{}_label_{}.pkl".format(data_file_name, str(data_idx),
                                                                                         str(len(sliced_y)),
                                                                                         str(label)), 'wb') as _f:
            pickle.dump(new_data, _f)
        new_data = {}


def main(args):
    save_directory = args.save_directory
    data_type = args.data_type
    dataset = args.dataset
    label_type = args.label_type
    sample_rate = args.samplerate
    cpu_num = args.cpu_num
    feature_type = args.feature_type
    feature_sample_rate = args.feature_sample_rate
    task_type = args.task_type
    data_file_directory = save_directory + "/dataset-{}_task-{}_datatype-{}_v6".format(dataset, task_type, data_type)

    labels = ['EEG FP1', 'EEG FP2', 'EEG F3', 'EEG F4', 'EEG F7', 'EEG F8',
              'EEG C3', 'EEG C4', 'EEG CZ', 'EEG T3', 'EEG T4',
              'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2', 'EEG T5', 'EEG T6', 'EEG PZ', 'EEG FZ']

    eeg_data_directory = "/Users/gresaismaili/Desktop/TUH-EEG/train"

    if label_type == "csv":
        disease_labels = {'bckg': 0, 'cpsz': 1, 'mysz': 2, 'gnsz': 3, 'fnsz': 4, 'tnsz': 5, 'tcsz': 6, 'spsz': 7, 'absz': 8}
    elif label_type == "csv_bi":
        disease_labels = {'bckg': 0, 'seiz': 1}
    disease_labels_inv = {v: k for k, v in disease_labels.items()}

    edf_list1 = search_walk({'path': eeg_data_directory, 'extension': ".edf"})
    edf_list2 = search_walk({'path': eeg_data_directory, 'extension': ".EDF"})

    edf_list1 = edf_list1 if isinstance(edf_list1, list) else []
    edf_list2 = edf_list2 if isinstance(edf_list2, list) else []

    edf_list = edf_list1 + edf_list2

    if os.path.isdir(data_file_directory):
        os.system("rm -rf {}".format(data_file_directory))
    os.system("mkdir {}".format(data_file_directory))

    GLOBAL_DATA['label_list'] = labels
    GLOBAL_DATA['disease_labels'] = disease_labels
    GLOBAL_DATA['disease_labels_inv'] = disease_labels_inv
    GLOBAL_DATA['data_file_directory'] = data_file_directory
    GLOBAL_DATA['label_type'] = label_type
    GLOBAL_DATA['feature_type'] = feature_type
    GLOBAL_DATA['feature_sample_rate'] = feature_sample_rate
    GLOBAL_DATA['sample_rate'] = sample_rate
    GLOBAL_DATA['fsr_sr_ratio'] = (sample_rate // feature_sample_rate)
    GLOBAL_DATA['min_binary_slicelength'] = args.min_binary_slicelength
    GLOBAL_DATA['min_binary_edge_seiz'] = args.min_binary_edge_seiz

    target_dictionary = {0: 0}
    selected_diseases = []
    for idx, i in enumerate(args.disease_type):
        selected_diseases.append(str(disease_labels[i]))
        target_dictionary[disease_labels[i]] = idx + 1

    GLOBAL_DATA[
        'disease_type'] = args.disease_type
    GLOBAL_DATA['target_dictionary'] = target_dictionary
    GLOBAL_DATA['selected_diseases'] = selected_diseases
    GLOBAL_DATA['binary_target1'] = args.binary_target1
    GLOBAL_DATA['binary_target2'] = args.binary_target2

    print("########## Preprocessor Setting Information ##########")
    print("Number of EDF files: ", len(edf_list))
    for i in GLOBAL_DATA:
        print("{}: {}".format(i, GLOBAL_DATA[i]))
    with open(data_file_directory + '/preprocess_info.infopkl', 'wb') as pkl:
        pickle.dump(GLOBAL_DATA, pkl, protocol=pickle.HIGHEST_PROTOCOL)
    print("################ Preprocess begins... ################\n")

    if (task_type == "binary") and (args.data_type == "train"):
        wrapped_func = partial(generate_training_data_leadwise_tuh_train_final, global_data=GLOBAL_DATA)
        run_multi_process(wrapped_func, edf_list, n_processes=cpu_num)
    elif (task_type == "binary") and (args.data_type == "dev"):
        wrapped_func = partial(generate_training_data_leadwise_tuh_train_final, global_data=GLOBAL_DATA)
        run_multi_process(wrapped_func, edf_list, n_processes=cpu_num)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-sd', type=int, default=1004,help='Random seed number')
    parser.add_argument('--samplerate', '-sr', type=int, default=200,help='Sample Rate')
    parser.add_argument('--save_directory', '-sp', type=str,help='Path to save data')
    parser.add_argument('--label_type', '-lt', type=str,default='csv',help='csv_bi = global with binary label, csv = global with various labels, cae = severance CAE seizure label.')
    parser.add_argument('--cpu_num', '-cn', type=int,default=32,help='select number of available cpus')
    parser.add_argument('--feature_type', '-ft', type=str,default=['rawsignal'])
    parser.add_argument('--feature_sample_rate', '-fsr', type=int,default=50,help='select features sample rate')
    parser.add_argument('--dataset', '-st', type=str,default='tuh',choices=['tuh'])
    parser.add_argument('--data_type', '-dt', type=str,default='train',choices=['train', 'dev'])
    parser.add_argument('--task_type', '-tt', type=str,default='binary',choices=['anomaly', 'multiclassification', 'binary'])

    parser.add_argument('--disease_type', nargs='+',
                        default=['gnsz', 'fnsz', 'spsz', 'cpsz', 'absz', 'tnsz', 'tcsz', 'mysz'],
                        choices=['gnsz', 'fnsz', 'spsz', 'cpsz', 'absz', 'tnsz', 'tcsz', 'mysz'])

    parser.add_argument('--binary_target1', type=str,
                        default="{0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}",
                        help='Dictionary as string for binary target1')

    parser.add_argument('--binary_target2', type=str,
                        default="{0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 1, 6: 3, 7: 4, 8: 5}",
                        help='Dictionary as string for binary target2')
    parser.add_argument('--min_binary_slicelength', type=int, default=30)
    parser.add_argument('--min_binary_edge_seiz', type=int, default=3)
    args = parser.parse_args()
    main(args)