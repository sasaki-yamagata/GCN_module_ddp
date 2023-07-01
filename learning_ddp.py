import numpy as np
import os
import pandas as pd
import pytz
import time
import torch
from datetime import datetime
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import torch.multiprocessing as mp
from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from GCN.data.dataloader import DataFrameLoader
from GCN.data.dataset import MoleculeDataset, gcn_collate_fn
from GCN.models.graph_conv_model import GraphConvModel
from GCN.common.method import fit, predict, accuracy ,torch_seed, init_gpu
from GCN.settings import config

## GPUのrankを取得
def main(df_path, target_props):
    torch_seed()
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    init_gpu(local_rank=LOCAL_RANK, is_torchrun=True)
    
    is_master = LOCAL_RANK == 0
    
    
    
    # データロード
    loader = DataFrameLoader(df_path=df_path, target_props=target_props, start=config["start"], end=config["end"])

    x_train, x_test, y_train, y_test = train_test_split(loader.x_data,  
                                                        loader.y,
                                                        test_size=0.25, 
                                                        random_state=1)
    del loader
    dataset_train = MoleculeDataset(x_train, y_train)
    dataset_test = MoleculeDataset(x_test, y_test)

    n_input = dataset_train.x_data[0].x.shape[1]
    n_output = dataset_train.y.shape[1]

    datasets = {"train": dataset_train, "test": dataset_test}
    del x_train, y_train, x_test, y_test, dataset_train, dataset_test

    

    dataloaders = {
        x: DataLoader(
            datasets[x],
            batch_size=config["batch_size"],
            collate_fn=gcn_collate_fn, 
            sampler=DistributedSampler(datasets[x], rank=LOCAL_RANK)
        ) for x in ["train", "test"]
    }

    # data_loader_train = DataLoader(molecule_dataset_train, 
    #                          batch_size=config["batch_size"],
    #                          collate_fn=gcn_collate_fn, 
    #                          sampler=DistributedSampler(molecule_dataset_train, rank=LOCAL_RANK))
    
   
    
    # data_loader_test = DataLoader(molecule_dataset_test, 
    #                               batch_size=config["batch_size"], 
    #                               collate_fn=gcn_collate_fn, 
    #                               sampler=DistributedSampler(molecule_dataset_test, rank=LOCAL_RANK))
    
    gc_hidden_size_list = [
        2**config["node_num"] if i % 2 == 0 else 2**config["node_num"]*2 for i in range(config["gc_layer_num"])]
    affine_hidden_size_list = [
        2**config["node_num"]*2 if i % 2 == 0 else 2**config["node_num"] for i in range(config["affine_layer_num"])] + [config["last_layer_node_num"]]
    
    model = GraphConvModel(n_input, n_output, gc_hidden_size_list, affine_hidden_size_list).cuda()


    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    model = DDP(model, device_ids=[LOCAL_RANK])

    # 学習
    start = time.time()
    model, history = fit(model, optimizer, criterion, config["n_epoch"], dataloaders)
    print(f"elapsed_time:{time.time() - start}")
    del dataloaders

    if is_master:
    # predictはシングルgpuで行う
        
        dataloaders = {
            x: DataLoader(
                datasets[x],
                batch_size=config["batch_size"],
                collate_fn=gcn_collate_fn, 
            ) for x in ["train", "test"]
        }
       
        # 予測
        result_train, result_test = predict(model, dataloaders)

        # 保存
        save_folder = f"result/{datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y-%m-%d-%H%M%S')}"
        os.makedirs(save_folder, exist_ok=True)
        target_props_str = "_".join(target_props)
        np.save(f"{save_folder}/config.npy", config)
        torch.save(model, f'{save_folder}/model_{target_props_str}.pth')
        torch.save(model.state_dict(), f"{save_folder}/model_weight_{target_props_str}.pth")
        df_history = pd.DataFrame(history)
        df_history.to_csv(f"{save_folder}/history_{target_props_str}.csv")
        train_columns = [f"train_obs_{target_prop}" for target_prop in target_props] \
                            + [f"train_pre_{target_prop}" for target_prop in target_props ]
        test_columns = [f"test_obs_{target_prop}" for target_prop in target_props] \
                            + [f"test_pre_{target_prop}" for target_prop in target_props ] 
        df_result_train = pd.DataFrame(result_train.detach().to("cpu"), columns=train_columns)
        df_result_test = pd.DataFrame(result_test.detach().to("cpu"), columns=test_columns)
        df_result_train.to_csv(f"{save_folder}/result_train_{target_props_str}.csv")
        df_result_test.to_csv(f"{save_folder}/result_test_{target_props_str}.csv")

        # 精度
        df_accuracy = accuracy(target_props, result_train, result_test)
        df_accuracy.to_csv(f"{save_folder}/accuracy_{target_props_str}.csv")


if __name__ == "__main__":
    abs_path = os.path.dirname(os.path.abspath(__file__))
    main(df_path=f"{abs_path}/datasets/homolumo_matrix.pkl", 
         target_props=["HOMO", "LUMO"])


# /workspace/230327_GCN_homo_lumo_pytorch/1_exec/datasets
# /home/usr9/n70419a/sasaki/230613_GCN_homo_lumo_pytorch_ddp/datasets
# /home1/t2/t231385m/Maildir/workspace/230613_GCN_homo_lumo_pytorch_ddp/datasets