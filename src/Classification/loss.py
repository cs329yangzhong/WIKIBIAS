# loss types for different type of self-paced learning method.
import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn as nn


def loss_number_(out, label, number):
    '''
    Args:
        out: output from the current model.
        label: gold labels for the batch. batch_size * 2
        number: the cut_off number for the sample in each batch. max batch_size
    Return:
        weighted_loss: losses multiply by the sample weights.
    '''

    loss = F.cross_entropy(out, torch.max(label, 1)[1], reduce=False)
    ind_sorted = torch.argsort(loss)
    loss_1_sorted = loss[ind_sorted]
    number = int(number)
    # cut off number, selecting the minimum.
    ind_update = ind_sorted[:number]
    weight_0_count = label.size()[0] - number
    loss_update = F.cross_entropy(
        out[ind_update], torch.max(label[ind_update], 1)[1])
    return loss_update, weight_0_count


def loss_number_old(out, label, old_loss, number):
    '''
    Args:
        out: output from the current model.
        label: gold labels for the batch. batch_size * 2
        number: the cut_off number for the sample in each batch. max batch_size
    Return:
        weighted_loss: losses multiply by the sample weights.
    '''

    loss = F.cross_entropy(out, torch.max(label, 1)[1], reduce=False)
    ind_sorted = torch.argsort(old_loss)
    loss_1_sorted = old_loss[ind_sorted]
    number = int(number)
    # cut off number, selecting the minimum.
    ind_update = ind_sorted[:number]
    weight_0_count = label.size()[0] - number
    loss_update = F.cross_entropy(
        out[ind_update], torch.max(label[ind_update], 1)[1])
    return loss_update, weight_0_count


def loss_threshold(out, label, old_loss, loss_cut_off, device):
    '''
    Args:
        out: output from the current model.
        label: gold labels for the batch. batch_size * 2
        number: the cut_off number for the sample in each batch. max batch_size
    Return:
        weighted_loss: losses multiply by the sample weights.
    '''
    loss = F.cross_entropy(out, torch.max(label, 1)[1], reduce=False)

    loss_np = loss.detach().cpu().numpy()
    loss_cut_off = np.percentile(loss_np, loss_cut_off)
    weights = np.ones(label.shape[0])
    weights[np.where(loss_np <= loss_cut_off)] = 1
    weights[np.where(loss_np > loss_cut_off)] = 0
    weight_0_count = np.count_nonzero(weights == 0)

    weights = torch.tensor(weights).to(device).float()
    weights.requires_grad = False
    weighted_loss = torch.mul(loss, weights)
    #weight_0_count = np.count_nonzero(weights==0)
    return weighted_loss.mean(), weight_0_count


def loss_threshold_old(out, label, old_loss, loss_cut_off, device):
    '''
    Args:
        out: output from the current model.
        label: gold labels for the batch. batch_size * 2
        number: the cut_off number for the sample in each batch. max batch_size
    Return:
        weighted_loss: losses multiply by the sample weights.
    '''
    loss = F.cross_entropy(out, torch.max(label, 1)[1], reduce=False)
    #loss_np = old_loss.cpu().numpy()
   # weights = np.ones(label.shape[0])
    #weights[np.where(loss_np <= loss_cut_off)] = 1
   # weights[np.where(loss_np > loss_cut_off)] = 0
    #weight_0_count = np.count_nonzero(weights==0)

    #weights = torch.tensor(weights).to(device).float()
    #weights.requires_grad = False
    weights = torch.ones(label.shape[0]).to(device)
    weights[torch.where(old_loss <= loss_cut_off)] = 1
    weights[torch.where(old_loss > loss_cut_off)] = 0
    weight_0_count = (weights == 0).sum().item()

    weights = weights.to(device).float()
    weights.require_grad = False
    weighted_loss = torch.mul(loss, weights)
    #weight_0_count = np.count_nonzero(weights==0)
    return weighted_loss.mean(), weight_0_count


class LossDropper(nn.Module):
    def __init__(
            self,
            dropc=0.1,
            min_count=10000,
            recompute=10000,
            verbose=True
    ):
        super().__init__()
        self.keepc = 1. - dropc
        self.count = 0
        self.min_count = min_count

        self.recompute = recompute
        self.last_computed = 0
        self.percentile_val = 1000000000.
        self.cur_idx = 0

        self.verbose = verbose
        self.vals = np.zeros(self.recompute, dtype=np.float32)

    def forward(self, loss):
        if loss is None:
            return loss

        self.last_computed += loss.numel()
        self.count += loss.numel()

        # Use all samples while not seen the first epoch for recompute num.

        if self.count < len(self.vals):
            self.vals[self.count - loss.numel()                      :self.count] = loss.detach().cpu().numpy().flatten()
            self.cur_idx += loss.numel()
            return (loss < np.inf).type(loss.dtype)
        else:
            for idx, item in enumerate(loss):
                self.vals[self.cur_idx] = item
                self.cur_idx += 1
                if self.cur_idx >= len(self.vals):
                    self.cur_idx = 0
        if self.count < self.min_count:
            return (loss < np.inf).type(loss.dtype)

        if self.last_computed > self.recompute:
            self.percentile_val = np.percentile(self.vals, self.keepc * 100)
            if self.verbose:
                print('Using cutoff', self.percentile_val)
                print(loss)
            self.last_computed = 0

        mask = (loss < self.percentile_val).type(loss.dtype)
        return mask


class MovingLossDropper(nn.Module):
    def __init__(
            self,
            dropc=0.1,
            min_count=10000,
            recompute=10000,
            verbose=True
    ):
        super().__init__()
        self.keepc = 1. - dropc
        self.count = 0
        self.min_count = min_count

        self.recompute = recompute
        self.last_computed = 0

        self.cur_idx = 0
        self.percentile_val = None
        self.verbose = verbose
        self.vals = np.zeros(self.recompute, dtype=np.float32)

    def forward(self, loss):
        if loss is None:
            return loss

        self.last_computed += loss.numel()
        self.count += loss.numel()

        # Use all samples while not seen the first epoch for recompute num.
        if self.count < len(self.vals):
            self.vals[self.count - loss.numel()                      :self.count] = loss.detach().cpu().numpy().flatten()
            self.cur_idx += loss.numel()
            return (loss < np.inf).type(loss.dtype)
        else:
            for idx, item in enumerate(loss):
                self.vals[self.cur_idx] = item
                self.cur_idx += 1
                if self.cur_idx >= len(self.vals):
                    self.cur_idx = 0
        if self.count < self.min_count:
            return (loss < np.inf).type(loss.dtype)

        if self.last_computed > self.recompute:
            if self.percentile_val == None:
                self.percentile_val = np.percentile(
                    self.vals, self.keepc * 100)
            else:
                new_percentile = np.percentile(self.vals, self.keepc * 100)
                self.percentile_val = 0.5 * self.percentile_val + 0.5 * new_percentile

            if self.verbose:
                print('Using cutoff', self.percentile_val)
                print(loss)
            self.last_computed = 0

        mask = (loss < self.percentile_val).type(loss.dtype)
        return mask
