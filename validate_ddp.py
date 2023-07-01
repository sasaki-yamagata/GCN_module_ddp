from typing import Any
import numpy as np
import time
import tracemalloc
import torch.distributed as dist
import optuna
import os
import pandas as pd
import pickle
import pytz
import torch
import torch.multiprocessing as mp
from glob import glob
from datetime import datetime
from torch import nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split, KFold

from GCN.data.dataloader import DataFrameLoader
from GCN.data.dataset import MoleculeDataset, gcn_collate_fn
from GCN.models.graph_conv_model import GraphConvModel
from GCN.common.method import fit, predict, accuracy ,torch_seed, init_gpu
from GCN.settings import config
ABS_PATH = os.path.dirname(os.path.abspath(__file__))


def main(df_path, target_props, timeout, study_path=None):
    torch_seed()
    

    if study_path is None:
        study = optuna.create_study(direction="minimize")
    else:
        with open(study_path, "rb") as f:
            study = pickle.load(f)

    objective = Objective(df_path, target_props)
    study.optimize(objective, catch=(ValueError,), timeout=timeout)
    # study.optimize(lambda trial: objective(trial, df_path, target_props), catch=(ValueError,), timeout=timeout)

    print(study.best_params)

    # 保存
    save_folder = f"valid_result/{datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y-%m-%d-%H%M%S')}"
    os.makedirs(save_folder, exist_ok=True)
    target_props_str = "_".join(target_props)
    study.trials_dataframe().to_csv(f"{save_folder}/validate_{target_props_str}.csv")
    with open(f"{save_folder}/study_{target_props_str}.pkl", "wb") as f:
        pickle.dump(study, f)
    # with open(f"{save_folder}/molcode_train_{target_props_str}.pkl", "wb") as f:
    #     pickle.dump(molcode_train, f)
    
class Objective:

    def __init__(self, df_path, target_props):
        self.df_path = df_path
        self.target_props = target_props


    def __call__(self, trial):
        

        bayes_params = {
            "gc_layer_num" : trial.suggest_int("gc_layer_num", 1, 9),
            "affine_layer_num" : trial.suggest_int("affine_layer_num", 1, 9),
            "node_num" : trial.suggest_int("node_num", 5, 7),
            "last_layer_node_num" : trial.suggest_int("last_layer_node_num", 4, 10),
            "lr" : trial.suggest_float("lr", 1e-4, 1e-3)
        }
        self.bayes_params = bayes_params
       
        mp.spawn(self._leaning,
                args=(),
                nprocs=torch.cuda.device_count())

        with open(f"{ABS_PATH}/temp/valid_value.txt", "r") as f:
            valid_value = float(f.readline())
        print(f"valid_value: {valid_value}")
        os.remove(f"{ABS_PATH}/temp/valid_value.txt")

        return valid_value
    
    def _leaning(self, local_rank):
        init_gpu(local_rank)
        is_master = local_rank == 0
        config.update(self.bayes_params)
        loader = DataFrameLoader(df_path=self.df_path, target_props=self.target_props, start=config["start"], end=config["end"])

        n_input = loader.x_data[0].x.shape[1]
        n_output = len(self.target_props)

        x, _, y, _, _, _ = train_test_split(loader.x_data,  
                                                        loader.y,
                                                        loader.molcode,
                                                        test_size=0.25, 
                                                        random_state=1)
        del loader, _
        metrix = "rmse" # ベイズ最適化の評価指標を指定
        kf = KFold(n_splits=config["n_splits"])
        for i, (train, test) in enumerate(kf.split(x)):
            x_train = [x[tr] for tr in train]
            x_test = [x[ts] for ts in test]
            y_train = y[train]
            y_test = y[test]
            datasets = {"train": MoleculeDataset(x_train, y_train), "test": MoleculeDataset(x_test, y_test)}
            del x_train, y_train, x_test, y_test

            self.datasets = datasets
            del datasets
            gc_hidden_size_list = [
                2**config["node_num"] if i % 2 == 0 else 2**config["node_num"]*2 for i in range(config["gc_layer_num"])]
            affine_hidden_size_list = [
                2**config["node_num"]*2 if i % 2 == 0 else 2**config["node_num"] for i in range(config["affine_layer_num"])] + [config["last_layer_node_num"]]
            
            dataloaders = {
                x: DataLoader(
                    self.datasets[x],
                    batch_size=config["batch_size"],
                    collate_fn=gcn_collate_fn, 
                    sampler=DistributedSampler(self.datasets[x], rank=local_rank)
                ) for x in ["train", "test"]
            }
            print(f"x.device: {next(iter(dataloaders['train']))[0].x.device}, local_rank: {local_rank}")

            model = GraphConvModel(n_input, n_output, gc_hidden_size_list, affine_hidden_size_list).cuda()

            model = DDP(model, device_ids=[local_rank])
            
            criterion = nn.MSELoss()
            
            optimizer = optim.Adam(model.parameters(), lr=config["lr"])
            model, _ = fit(model, optimizer, criterion, config["n_epoch"], dataloaders, is_detect_anomaly=False)

            del dataloaders
            if is_master:
                dataloaders = {
                    x: DataLoader(
                        self.datasets[x],
                        batch_size=config["batch_size"],
                        collate_fn=gcn_collate_fn, 
                    ) for x in ["train", "test"]
                }
                del self.datasets
                result_train, result_test = predict(model, dataloaders)
                df_accuracy_temp = accuracy(self.target_props, result_train, result_test)

                if i == 0:
                    df_accuracy_sum = df_accuracy_temp
                else:
                    df_accuracy_sum = df_accuracy_sum.add(df_accuracy_temp)
                
        if is_master:
            df_accuracy = df_accuracy_sum / config["n_splits"]
            print(df_accuracy)
            df_test_accuracy = df_accuracy.loc[[i for i in df_accuracy.index if "test" in i], metrix]
            valid_value = df_test_accuracy.mean()
            os.makedirs(f"{ABS_PATH}/temp", exist_ok=True)
            with open(f"{ABS_PATH}/temp/valid_value.txt", "w") as f:
                f.write(str(valid_value))


        dist.destroy_process_group()
    
# def objective(trial, df_path, target_props):
#     bayes_params = {
#             "gc_layer_num" : trial.suggest_int("gc_layer_num", 1, 9),
#             "affine_layer_num" : trial.suggest_int("affine_layer_num", 1, 9),
#             "node_num" : trial.suggest_int("node_num", 5, 7),
#             "last_layer_node_num" : trial.suggest_int("last_layer_node_num", 4, 10),
#             "lr" : trial.suggest_float("lr", 1e-4, 1e-3)
#         }
#     bayes_params = bayes_params
#     queue = mp.Queue()
#     mp.spawn(leaning,
#             args=(df_path, target_props, bayes_params, queue),
#             nprocs=torch.cuda.device_count())
#     valid_value = queue.get()
#     return valid_value

# def leaning(local_rank, df_path, target_props, bayes_params, queue):
#     init_gpu(local_rank)
#     is_master = local_rank == 0
#     config.update(bayes_params)
#     loader = DataFrameLoader(df_path=df_path, target_props=target_props, start=config["start"], end=config["end"])

#     n_input = loader.x_data[0].x.shape[1]
#     n_output = len(target_props)

#     x, _, y, _, _, _ = train_test_split(loader.x_data,  
#                                                     loader.y,
#                                                     loader.molcode,
#                                                     test_size=0.25, 
#                                                     random_state=1)
#     del loader, _
#     metrix = "rmse" # ベイズ最適化の評価指標を指定
#     kf = KFold(n_splits=config["n_splits"])
#     for i, (train, test) in enumerate(kf.split(x)):
#         x_train = [x[tr] for tr in train]
#         x_test = [x[ts] for ts in test]
#         y_train = y[train]
#         y_test = y[test]
#         datasets = {"train": MoleculeDataset(x_train, y_train), "test": MoleculeDataset(x_test, y_test)}
#         del x_train, y_train, x_test, y_test

#         gc_hidden_size_list = [
#             2**config["node_num"] if i % 2 == 0 else 2**config["node_num"]*2 for i in range(config["gc_layer_num"])]
#         affine_hidden_size_list = [
#             2**config["node_num"]*2 if i % 2 == 0 else 2**config["node_num"] for i in range(config["affine_layer_num"])] + [config["last_layer_node_num"]]
        
#         dataloaders = {
#             x: DataLoader(
#                 datasets[x],
#                 batch_size=config["batch_size"],
#                 collate_fn=gcn_collate_fn, 
#                 sampler=DistributedSampler(datasets[x], rank=local_rank)
#             ) for x in ["train", "test"]
#         }
#         print(f"x.device: {next(iter(dataloaders['train']))[0].x.device}, local_rank: {local_rank}")

#         model = GraphConvModel(n_input, n_output, gc_hidden_size_list, affine_hidden_size_list).cuda()

#         model = DDP(model, device_ids=[local_rank])
        
#         criterion = nn.MSELoss()
        
#         optimizer = optim.Adam(model.parameters(), lr=config["lr"])
#         model, _ = fit(model, optimizer, criterion, config["n_epoch"], dataloaders, is_detect_anomaly=False)

#         del dataloaders
#         if is_master:
#             dataloaders = {
#                 x: DataLoader(
#                     datasets[x],
#                     batch_size=config["batch_size"],
#                     collate_fn=gcn_collate_fn, 
#                 ) for x in ["train", "test"]
#             }
#             del datasets
#             result_train, result_test = predict(model, dataloaders)
#             df_accuracy_temp = accuracy(target_props, result_train, result_test)

#             if i == 0:
#                 df_accuracy_sum = df_accuracy_temp
#             else:
#                 df_accuracy_sum = df_accuracy_sum.add(df_accuracy_temp)
            
#     if is_master:
#         df_accuracy = df_accuracy_sum / config["n_splits"]
#         print(df_accuracy)
#         df_test_accuracy = df_accuracy.loc[[i for i in df_accuracy.index if "test" in i], metrix]
#         valid_value = df_test_accuracy.mean()
#         queue.put(valid_value)

# def objective(trial, x, y, n_input, n_output, target_props):

#     bayes_params = {
#         "gc_layer_num" : trial.suggest_int("gc_layer_num", 1, 9),
#         "affine_layer_num" : trial.suggest_int("affine_layer_num", 1, 9),
#         "node_num" : trial.suggest_int("node_num", 5, 7),
#         "last_layer_node_num" : trial.suggest_int("last_layer_node_num", 4, 10),
#         "lr" : trial.suggest_float("lr", 1e-4, 1e-3)
#     }
    

#     metrix = "rmse" # ベイズ最適化の評価指標を指定
#     kf = KFold(n_splits=config["n_splits"])
#     for i, (train, test) in enumerate(kf.split(x)):
#         x_train = [x[tr] for tr in train]
#         x_test = [x[ts] for ts in test]
#         y_train = y[train]
#         y_test = y[test]
#         dataset_train = MoleculeDataset(x_train, y_train)
#         dataset_test = MoleculeDataset(x_test, y_test)
#         datasets = {"train": dataset_train, "test": dataset_test}
#         del x_train, y_train, x_test, y_test, dataset_train, dataset_test

#         mp.spawn(leaning,
#                 args=(datasets, n_input, n_output, target_props, bayes_params),
#                 nprocs=torch.cuda.device_count())

#         file = glob(f"{ABS_PATH}/accuracy_temp/*")[0]
#         df_accuracy_temp = pd.read_csv(file, index_col=0)
#         os.remove(file)
        
#         if i == 0:
#             df_accuracy_sum = df_accuracy_temp
#         else:
#             df_accuracy_sum = df_accuracy_sum.add(df_accuracy_temp)
        

#     df_accuracy = df_accuracy_sum / config["n_splits"]
#     print(df_accuracy)
#     df_test_accuracy = df_accuracy.loc[[i for i in df_accuracy.index if "test" in i], metrix]
#     valid_value = df_test_accuracy.mean()
#     return valid_value
    


# def leaning(local_rank, datasets, n_input, n_output, target_props, bayes_params):
#     init_gpu(local_rank)
#     config.update(bayes_params)

#     is_master = local_rank == 0

#     gc_hidden_size_list = [
#         2**config["node_num"] if i % 2 == 0 else 2**config["node_num"]*2 for i in range(config["gc_layer_num"])]
#     affine_hidden_size_list = [
#         2**config["node_num"]*2 if i % 2 == 0 else 2**config["node_num"] for i in range(config["affine_layer_num"])] + [config["last_layer_node_num"]]
    
#     dataloaders = {
#         x: DataLoader(
#             datasets[x],
#             batch_size=config["batch_size"],
#             collate_fn=gcn_collate_fn, 
#             sampler=DistributedSampler(datasets[x], rank=local_rank)
#         ) for x in ["train", "test"]
#     }

#     # print(f"x.device: {next(iter(dataloaders['train']))[0].x.device}, local_rank: {local_rank}")

#     model = GraphConvModel(n_input, n_output, gc_hidden_size_list, affine_hidden_size_list).cuda()

#     model = DDP(model, device_ids=[local_rank])
    
#     criterion = nn.MSELoss()
    
#     optimizer = optim.Adam(model.parameters(), lr=config["lr"])
#     model, _ = fit(model, optimizer, criterion, config["n_epoch"], dataloaders, is_detect_anomaly=False)

#     if is_master:
#         dataloaders = {
#             x: DataLoader(
#                 datasets[x],
#                 batch_size=config["batch_size"],
#                 collate_fn=gcn_collate_fn, 
#             ) for x in ["train", "test"]
#         }

#         result_train, result_test = predict(model, dataloaders)
#         df_accuracy = accuracy(target_props, result_train, result_test)

#         os.makedirs(f"{ABS_PATH}/accuracy_temp", exist_ok=True)
#         df_accuracy.to_csv(f"{ABS_PATH}/accuracy_temp/accuracy_temp.csv")
#         dist.destroy_process_group()

if __name__ == "__main__":
    main(df_path=f"{ABS_PATH}/datasets/homolumo_matrix.pkl", 
         target_props=["HOMO", "LUMO"], 
         timeout= 20
         )