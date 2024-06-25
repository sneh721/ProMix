import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

def debias_pl(logit,bias,tau=0.4):
    bias = bias.detach().clone()
    debiased_prob = F.softmax(logit - tau*torch.log(bias), dim=1)
    return debiased_prob


def debias_output(logit, bias, tau=0.8):
    bias = bias.detach().clone()
    debiased_opt = logit + tau*torch.log(bias)
    return debiased_opt

def bias_initial(num_class=10):
    bias = (torch.ones(num_class, dtype=torch.float)/num_class).to(set_device())
    return bias

def bias_update(input, bias, momentum, bias_mask=None):
    if bias_mask is not None:
        input_mean = input.detach()*bias_mask.detach().unsqueeze(dim=-1)
    else:
        input_mean = input.detach().mean(dim=0)
    bias = momentum * bias + (1 - momentum) * input_mean
    return bias

def set_global_seeds(i):
    random.seed(i)
    np.random.seed(i)
    torch.manual_seed(i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(i)

def set_device():
    if torch.cuda.is_available():
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")
    print(f'Current device is {_device}', flush=True)
    return _device

class CE_Soft_Label(nn.Module):
    def __init__(self):
        super().__init__()
        self.confidence = None

    def init_confidence(self, noisy_labels, num_class):
        noisy_labels = torch.Tensor(noisy_labels).long().to(set_device())
        self.confidence = F.one_hot(noisy_labels, num_class).float().clone().detach()

    def forward(self, outputs, targets=None):
        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = logsm_outputs * targets.detach()
        loss_vec = - ((final_outputs).sum(dim=1))
        average_loss = loss_vec.mean()
        return loss_vec

    @torch.no_grad()
    def confidence_update(self, temp_un_conf, batch_index, conf_ema_m):
        with torch.no_grad():
            _, prot_pred = temp_un_conf.max(dim=1)
            pseudo_label = F.one_hot(prot_pred, temp_un_conf.shape[1]).float().to(set_device()).detach()
            self.confidence[batch_index, :] = conf_ema_m * self.confidence[batch_index, :]\
                 + (1 - conf_ema_m) * pseudo_label
        return None

def linear_rampup2(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length
        
# # Adjusted scheduler to update every iteration instead of every epoch
# def adjust_learning_rate(args, optimizer, iteration, total_iterations, epoch, pretrain_iterations):
#     lr = args.lr
#     pretrain_epochs = args.pretrain_ep

#     # Warmup phase: increasing lr
#     if epoch < pretrain_epochs:
#         lr = lr * (iteration / pretrain_iterations)
#     else:  # Training phase: decreasing lr
#         iteration -= pretrain_iterations
#         total_iterations -= pretrain_iterations
#         eta_min = lr * (args.lr_decay_rate ** 3)
#         lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * iteration / total_iterations)) / 2

#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
        
        
# Adjusted scheduler to update every iteration instead of every epoch
def adjust_learning_rate(args, optimizer, curr_batch, warmup_batches, total_batches):
    lr = args.lr

    # Warmup phase: increasing lr
    if curr_batch < warmup_batches:
        lr = lr * (curr_batch / warmup_batches)
    else:  # Training phase: decreasing lr
        train_batches = total_batches - warmup_batches
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * curr_batch / train_batches)) / 2

    lr = max(lr, 0.000001)      # just so that lr is never 0
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        