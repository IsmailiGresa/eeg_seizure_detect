import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def plot_eeg_similarity_map(mma, sample, n_head):
    print("mma: ", mma)
    sample = sample.permute(1, 0)

    label = ["fp1-f7", "fp2-f8", "f7-t3", "f8-t4", "t3-t5", "t4-t6", "t5-o1", "t6-o2", "t3-c3",
             "c4-t4", "c3-cz", "cz-c4", "fp1-f3", "fp2-f4", "f3-c3", "f4-c4", "c3-p3", "c4-p4", "p3-o1", "p4-o2"]

    plt.figure()
    for idx, label_name in enumerate(label):
        plt.subplot(20, 1, idx + 1)
        plt.plot(sample[idx].detach().cpu().numpy())
        plt.legend([label_name])
    plt.show()

    for i in range(n_head):
        fig, ax = plt.subplots()
        heatmap = ax.pcolor(mma[i], cmap=plt.cm.Blues)
        ax.set_xticks(np.arange(20) + 0.5, minor=False)
        ax.set_yticks(np.arange(20) + 0.5, minor=False)
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.set_xticklabels(label, minor=False)
        ax.set_yticklabels(label, minor=False)
        plt.xticks(rotation=45)
        plt.show()
    exit(1)

def sliding_window_v1(args, iteration, train_x, train_y, seq_lengths=None, target_lengths=None,
                      model=None, logger=None, device=None, scheduler=None,
                      optimizer=None, criterion=None, signal_name_list=None,
                      flow_type="train"):

    train_x = train_x.squeeze(1).permute(0, 2, 1).contiguous()
    train_y = train_y.to(device)

    iter_loss = []
    val_loss = []

    answer_list = []
    prediction_list = []

    requirement_target1 = args.window_shift_label // 2
    requirement_target2 = args.window_size_label // 4

    if args.requirement_target is not None:
        requirement_target = int(args.requirement_target * args.feature_sample_rate)
    elif requirement_target1 >= requirement_target2:
        requirement_target = requirement_target1
    else:
        requirement_target = requirement_target2

    model.init_state(device)
    shift_start = 0
    shift_num = (train_x.size(1) - args.window_size_label) // args.window_shift_label + 1

    for i in range(shift_start, shift_num):
        x_idx = i * args.window_shift
        y_idx = i * args.window_shift_label

        slice_start = x_idx
        slice_end = x_idx + args.window_size

        seq_slice = train_x[:, slice_start:slice_end, :]

        target_temp = train_y[:, y_idx: y_idx + args.window_size_label].float().to(device)

        target, _ = torch.max(target_temp, 1)
        seiz_count = torch.count_nonzero(target_temp, dim=1)

        target[seiz_count < requirement_target] = 0
        final_target = torch.round(target).long().squeeze().to(device)
        if final_target.dim() == 0:
            final_target = final_target.unsqueeze(0)

        if final_target.numel() == 0 or final_target.shape[0] != seq_slice.shape[0]:
            continue

        if flow_type == "train":
            optimizer.zero_grad()
            logits = model(seq_slice)

            loss = criterion(logits, final_target)

            if args.loss_decision == "max_division":
                loss = torch.div(torch.sum(loss), args.batch_size)
            elif args.loss_decision == "mean":
                loss = torch.mean(loss)
            else:
                print("Error! Select Correct args.loss_decision...")
                exit(1)

            iter_loss.append(loss.item())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

        else:
            with torch.no_grad():
                logits = model(seq_slice)
                proba = nn.functional.softmax(logits, dim=1)

                if torch.isnan(proba).any() or torch.isinf(proba).any():
                    print("❌ proba contains NaN or Inf")

                logger.pred_results.append(proba.cpu())

                loss = criterion(logits, final_target)
                val_loss.append(torch.mean(loss).item())

                _, predicted = logits.max(1)
                answer_list.append(final_target.cpu().numpy())
                prediction_list.append(predicted.cpu().numpy())

                if args.calibration:
                    model.temperature_scaling.collect_logits(logits)
                    model.temperature_scaling.collect_labels(final_target)

                if args.binary_target_groups == 1:
                    final_target[final_target != 0] = 1
                    re_proba = torch.cat((proba[:, 0].unsqueeze(1), torch.sum(proba[:, 1:], 1).unsqueeze(1)), 1)
                    logger.evaluator.add_batch(np.array(final_target.cpu()), np.array(re_proba.cpu()))
                else:
                    logger.evaluator.add_batch(np.array(final_target.cpu()), np.array(proba.cpu()))

                if args.margin_test:
                    probability = proba[:, 1]
                    logger.evaluator.probability_list.append(probability)
                    logger.evaluator.final_target_list.append(final_target)

    if flow_type == "train":
        return model, iter_loss
    else:
        return model, val_loss