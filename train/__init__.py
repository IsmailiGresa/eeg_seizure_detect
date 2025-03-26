from .trainer import *

def get_trainer(args, iteration, x, y, seq_lengths, target_lengths, model, logger, device, scheduler, optimizer, criterion, signal_name_list, flow_type="train"):
    model, iter_loss = sliding_window_v1(args, iteration, x, y, seq_lengths, target_lengths, model, logger, device, scheduler, optimizer, criterion, signal_name_list, flow_type)

    return model, iter_loss