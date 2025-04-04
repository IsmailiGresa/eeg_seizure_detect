import torch

def binary_detector_evaluator(pred_stack, target_stack, margin):
    rise_true, rise_pred_correct, fall_true, fall_pred_correct = 0, 0, 0, 0
    target_rotated = torch.cat([target_stack[0].unsqueeze(0), target_stack[:-1]], dim=0)
    pred_rotated = torch.cat([pred_stack[0].unsqueeze(0), pred_stack[:-1]], dim=0)

    target_change = torch.subtract(target_rotated, target_stack)
    pred_change = torch.subtract(pred_rotated, pred_stack)

    for idx, sample in enumerate(target_change.permute(1, 0)):
        fall_index_list = (sample == 1).nonzero(as_tuple=True)[0]
        rise_index_list = (sample == -1).nonzero(as_tuple=True)[0]

        for fall_index in fall_index_list:
            start_margin_index = fall_index - margin
            end_margin_index = fall_index + margin
            if start_margin_index < 0:
                start_margin_index = 0
            if end_margin_index > len(sample):
                end_margin_index = len(sample)
            if 1 in pred_change[start_margin_index:end_margin_index + 1]:
                fall_pred_correct += 1
            fall_true += 1
        for rise_index in rise_index_list:
            start_margin_index = rise_index - margin
            end_margin_index = rise_index + margin
            if start_margin_index < 0:
                start_margin_index = 0
            if end_margin_index > len(sample):
                end_margin_index = len(sample)
            if -1 in pred_change[start_margin_index:end_margin_index + 1]:
                rise_pred_correct += 1
            rise_true += 1

    return rise_true, rise_pred_correct, fall_true, fall_pred_correct