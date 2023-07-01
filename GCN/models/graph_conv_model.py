import os
import torch
from datetime import datetime

from torch import nn
from torch_geometric.nn import GCNConv, Sequential

from ..layers.graph_conv_layer import GraphGather, TanhExp
# logging.basicConfig(level=logging.INFO, filename=f"logs/{datetime.now().strftime('%Y-%m-%d-%H%M%S')}outputs.log", format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
class GraphConvModel(nn.Module):
    def __init__(self, n_input, n_output, gc_hidden_size_list, affine_hidden_size_list):
        super().__init__()
        activation = TanhExp.apply
        conv_layers = []
        for i in range(len(gc_hidden_size_list)):
            if i == 0:
                conv_layers.append((GCNConv(n_input, gc_hidden_size_list[i]), "x, edge_index -> x"))
                conv_layers.append(activation)
            else:
                conv_layers.append((GCNConv(gc_hidden_size_list[i-1], gc_hidden_size_list[i]), "x, edge_index -> x"))
                conv_layers.append(activation)
                # setattr(self, f"conv{num}", GCNConv(gc_hidden_size_list[i-1], gc_hidden_size_list[i]))
        liner_layers = []
        for i in range(len(affine_hidden_size_list)):
            if i == 0:
                liner_layers.append((nn.Linear(gc_hidden_size_list[-1], affine_hidden_size_list[i]), "x -> x"))
                liner_layers.append(activation)
                # setattr(self, f"l{num}", nn.Linear(gc_hidden_size_list[-1], affine_hidden_size_list[i]))

            elif i == len(affine_hidden_size_list) - 1:
                liner_layers.append((nn.Linear(affine_hidden_size_list[i-1], affine_hidden_size_list[i]), "x -> x"))
                liner_layers.append(activation)
                liner_layers.append((nn.Linear(affine_hidden_size_list[i], n_output), "x -> x"))

                # setattr(self, f"l{num}", nn.Linear(affine_hidden_size_list[i-1], affine_hidden_size_list[i]))
                # setattr(self, f"l{num}", nn.Linear(affine_hidden_size_list[i], n_output))
            else:
                liner_layers.append((nn.Linear(affine_hidden_size_list[i-1], affine_hidden_size_list[i]), "x -> x"))
                liner_layers.append(activation)
                # setattr(self, f"l{num}", nn.Linear(affine_hidden_size_list[i-1], affine_hidden_size_list[i]))
        
        
        self.conv = Sequential("x, edge_index", conv_layers)
        self.gather = GraphGather()
        self.linear = Sequential("x", liner_layers)
        # self.conv = nn.Sequential(
        #     self.conv1,
        #     self.tanhexp, 
        #     self.conv2,
        #     self.tanhexp,
        #     self.conv3,
        #     self.tanhexp
        # )

        # self.linear = nn.Sequential(
        #     self.l1,
        #     self.tanhexp,
        #     self.l2,
        #     self.tanhexp,
        #     self.l3,
        # )



    def forward(self, x, edge_index, feature_size_list):

        x = self.conv(x, edge_index)
        x = self.gather(x, feature_size_list)
        x = self.linear(x)
        
        return x
    
    # def __transform_data(self, x_data):
        
    #     # if config["device"] == "cuda":
    #     #     device_index = torch.cuda.current_device()
    #     #     device = f"cuda:{device_index}"
    #     # else:
    #     #     device = "cpu"

    #     feature_batch = torch.zeros((0, x_data[0].x.shape[1]))
    #     edge_index_batch = torch.zeros((2, 0), dtype=torch.long)
    #     feature_size_list = []
    #     for i, x_d in enumerate(x_data):
    #         x_d.x = x_d.x
    #         x_d.edge_index = x_d.edge_index
    #         # print(f"feature_batch_device: {feature_batch.device.type}")
    #         # print(f"x_d.x: {x_d.x.device.type}")
    #         feature_batch = torch.cat([feature_batch, x_d.x], dim=0)
    #         feature_size_list.append(x_d.feature_size)
    #         edge_index_batch = torch.cat([edge_index_batch, x_d.edge_index+sum(feature_size_list[:i])], dim=1)
    #     return feature_batch, edge_index_batch, feature_size_list

    