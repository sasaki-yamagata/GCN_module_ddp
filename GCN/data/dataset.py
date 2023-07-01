import os
import torch
from rdkit import Chem
from torch.utils import data
from torch_geometric.data import Data
from GCN.settings import config



class MoleculeDataset(data.Dataset):
    def __init__(self, x_data, y):
        self.x_data = x_data
        self.y = y


    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, index):
        return self.x_data[index], self.y[index, :]
    

def gcn_collate_fn(batch):
    x_data, y = zip(*batch)
    feature_batch = torch.zeros((0, x_data[0].x.shape[1])).cuda()
    edge_index_batch = torch.zeros((2, 0), dtype=torch.long).cuda()


    feature_size_list = []
    for i, x_d in enumerate(x_data):
        x_d.x = x_d.x.cuda()
        x_d.edge_index = x_d.edge_index.cuda()
        x_d.feature_size = x_d.feature_size.cuda()
        
        feature_batch = torch.cat([feature_batch, x_d.x], dim=0)
        feature_size_list.append(x_d.feature_size)
        edge_index_batch = torch.cat([edge_index_batch, x_d.edge_index+sum(feature_size_list[:i])], dim=1)

    # 目的変数をバッチ化
    y_num = y[0].shape[0]
    y_batch = torch.zeros((0, y_num), dtype=torch.float).cuda()
    for target in y:
        target = target.cuda()
        target = target.view(1, y_num)
        y_batch = torch.cat([y_batch, target], dim=0)

    return Data(x=feature_batch, edge_index=edge_index_batch, feature_size_list=feature_size_list), y_batch

# DP用にfeature, edge_sizeの変形をおこなわないcollate_fn
# def gcn_collate_fn(batch):

        
#     x_data, y = zip(*batch)
#     # print(x_data, y)
#     x_data_batch = [x for x in x_data]
#     y_num = y[0].shape[0]
#     y_batch = torch.zeros((0, y_num), dtype=torch.float)
#     for target in y:
#         target = target.view(1, y_num)
#         y_batch = torch.cat([y_batch, target], dim=0)
    
#     return x_data_batch, y_batch


