import os
import torch
import torch.nn as nn



class GraphGather(nn.Module):


    def forward(self, x, feature_size_list):

        x_gataher = torch.zeros((0, x.shape[1]), dtype=torch.float).cuda()
        for i, feature_size in enumerate(feature_size_list):
            start = sum(feature_size_list[:i])
            end = start + feature_size
            x_gataher = torch.cat([x_gataher, torch.mean(x[start:end, :], dim=0).view(1, -1)])
        return x_gataher


class TanhExp(torch.autograd.Function):


    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.tanh(torch.exp(x))
    
    @staticmethod
    def backward(ctx, dout):
        x, = ctx.saved_tensors
        exp_x = torch.exp(x)
        inf_mask = torch.isinf(exp_x)
        dx_over_dout_temp = torch.zeros(x.shape).cuda()
        dx_over_dout_temp[~inf_mask] = (x * (exp_x * (torch.tanh(exp_x)**2 - 1)))[~inf_mask]
        dx_over_dout_temp[inf_mask] = 0
        dx_over_dout = torch.tanh(exp_x) - dx_over_dout_temp
        dx = dout * dx_over_dout
        return dx
    

class Swish(nn.Module):

    
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * torch.sigmoid(x)