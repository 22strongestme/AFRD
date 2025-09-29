import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import Module


class CoverModel(torch.nn.Module):
    def __init__(self):
        super(CoverModel, self).__init__()

    def get_model(self, **kwargs)->torch.nn.Module:
        raise NotImplementedError

    # def get_optimizer(self, lr, **kwargs)->torch.optim.Optimizer:
    #     raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def forward(self, x, **kwargs):
        raise NotImplementedError

    def cal_loss(self, **kwargs)->torch.Tensor:
        raise NotImplementedError

    def cal_am(self, **kwargs):
        raise NotImplementedError

    def train_mode(self):
        raise NotImplementedError

    def eval_mode(self):
        raise NotImplementedError