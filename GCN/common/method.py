import itertools
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import subprocess
import torch
import torch.distributed as dist


from datetime import datetime
from pprint import pprint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.autograd import set_detect_anomaly

def fit(model, optimizer, criterion, n_epochs, dataloaders, is_detect_anomaly=False): 
    # ログファイルの作成
    # os.makedirs("logs", exist_ok=True)
    # filename = f"logs/{datetime.now().strftime('%Y-%m-%d-%H%M%S')}.log"
    # logging.basicConfig(level=logging.INFO, filename=filename, format="%(asctime)s %(levelname)s:%(name)s:%(message)s") 
    local_rank = torch.cuda.current_device()
    is_master = local_rank == 0

    batch_size = dataloaders["train"].batch_size
    history = {"epoch_num" : [],
               "train_loss" : [],
               "test_loss" : []}
    
    for epoch in range(n_epochs):  
        
        for phase in ["train", "test"]:
            dist.barrier() # 他のGPUプロセスが終わるまで処理を止める

            if phase == "train":
                model.train()
            else:
                model.eval()

            # 1エポックあたりの累積損失(平均化前)
            loss_accum = 0

            for x_data, y in dataloaders[phase]:     

                optimizer.zero_grad()
                
                # foward 
                # 自動微分をtrainのときのみ行う
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(x_data.x, x_data.edge_index, x_data.feature_size_list)

                    loss = criterion(outputs, y)

                    if phase == "train":

                        # backward中に発生したエラーを検出する
                        with set_detect_anomaly(is_detect_anomaly):
                            loss.backward()
                        
                        optimizer.step()

                    # lossは平均計算が行われているので平均前の損失に戻して加算
                    loss_accum += loss * batch_size
                    del loss

            # 各GPUから損失を集約
            dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM)
            torch.cuda.empty_cache()

            # ホストGPUときのみ損失を出力
            if is_master:
                loss_avg = loss_accum.item() / len(dataloaders[phase].dataset.x_data)
                print(f"epoch: {epoch}, {phase} loss: {loss_avg:.4f}")

                if phase == "train":
                    history["epoch_num"].append(epoch)
                    history["train_loss"].append(loss_avg)
                else:
                    history["test_loss"].append(loss_avg)
  
    return model, history


def predict(model, dataloaders):
    
    model.eval()
    result_dic = {}
    with torch.no_grad():
        for phase in ["train", "test"]:
            result = torch.zeros((0, dataloaders[phase].dataset.y.shape[1]*2)).cuda()
            for x_data, y in dataloaders[phase]:
                pred = model(x_data.x, x_data.edge_index, x_data.feature_size_list)
                result = torch.cat([result, torch.cat([y, pred], dim=1)], dim=0)
            result_dic[phase] = result

        return result_dic["train"], result_dic["test"]
        # result_train = torch.zeros((0, loader_train.dataset.y.shape[1]*2)).to(config["device"])
        # for x_data_train, y_train in loader_train:
        #     x_data_train = x_data_train.to(config["device"])
        #     y_train = y_train.to(config["device"])
        #     pre_train = model(x_data_train.x, x_data_train.edge_index, x_data_train.feature_size_list)
        #     result_train = torch.cat([result_train, torch.cat([y_train, pre_train], dim=1)], dim=0)

        # result_test = torch.zeros((0, loader_test.dataset.y.shape[1]*2)).to(config["device"])
        # for x_data_test, y_test in loader_test:
        #     x_data_test = x_data_test.to(config["device"])
        #     y_test = y_test.to(config["device"])
        #     pre_test = model(x_data_test.x, x_data_test.edge_index, x_data_test.feature_size_list)
        #     result_test = torch.cat([result_test, torch.cat([y_test, pre_test], dim=1)], dim=0)



def accuracy(target_props, result_train, result_test):
    metrix_dic = {"mse":mean_squared_error, "rmse":mean_squared_error, "mae":mean_absolute_error, "r2":r2_score}
    n_output = len(target_props)
    result_train = result_train.detach().to("cpu")
    result_test = result_test.detach().to("cpu")

    accuracy = {}
    # accuracy["phase"] = [f'{phase}_accuracy_{target_prop}' for target_prop, phase in itertools.product(target_props, ['train', 'test'])]
    for metrix_name, metrix_method in metrix_dic.items():
        accuracy[metrix_name] = []
        for i in range(n_output):
            if metrix_name == "rmse":
                train_accuracy = metrix_method(result_train[:, i], result_train[:, i+n_output], squared=False)
                test_accuracy = metrix_method(result_test[:, i], result_test[:, i+n_output], squared=False)
            else:
                train_accuracy = metrix_method(result_train[:, i], result_train[:, i+n_output])
                test_accuracy = metrix_method(result_test[:, i], result_test[:, i+n_output])

            accuracy[metrix_name].append(train_accuracy)
            accuracy[metrix_name].append(test_accuracy) 
    df_accuracy = pd.DataFrame(accuracy, index=[f'{phase}_accuracy_{target_prop}' for target_prop, phase in itertools.product(target_props, ['train', 'test'])])
    return df_accuracy 

def make_accuracy_scatter(x, y, padding=0, step=2, title=None, max_value=None, min_value=None, is_lim=False, figpath=False, cycler=0):
    '''
    Info:
        実測値と予測値のプロットを作る

    Arg:
        padding {int} -- 目盛りの範囲を決める際に、最大値と最小値にプラスする値
        step {int} -- 目盛りの間隔を決める
        title {str} -- 図のタイトルを決める
        max_value {int} -- 目盛りの最大値を決める、決めない場合は、自動で設定。
        min_value {int} -- 目盛りの最大値を決める、決めない場合は、自動で設定。
        is_lim {bool} -- 図のx, yの範囲を決定
        figpath {str} -- 図の作成したいパスを指定、指定しない場合は、作成されない。
        cycler {int} -- プロットのカラーを設定。デフォルトのカラーマップに従う。
        
    '''
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # フォントの種類の設定
    plt.rcParams['font.family'] = 'Times New Roman'
    # フォントのサイズの設定
    plt.rcParams["font.size"] = 15
    # 軸を内側にする
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.figure(figsize=(10/3, 10/3), dpi=300)
    plt.scatter(x, y, alpha=0.3, s=10, c=colors[cycler])
    plt.xlabel('DFT Calculation [eV]')
    plt.ylabel('Machine Learning [eV]')


    # xとyの目盛りの範囲をそろえる
    if max_value is None:
        max_value = round(max(x.max(), y.max()))
        max_value += padding
        
        # 偶数にする
        if not max_value % 2 == 0:
            max_value += 1

        
    if min_value is None:
         min_value = round(min(x.min(), y.min()))
         min_value -= padding

         if not min_value % 2 == 0:
            min_value -= 1

    if title:
        plt.title(title)
    

    # 目盛りの範囲を指定
    tick_range = np.arange(min_value, max_value+1, step)

    # 精度線を引く
    line_plot = np.linspace(min_value, max_value)
    plt.plot(line_plot, line_plot, c="black", lw=0.8)

    plt.xticks(tick_range)
    plt.yticks(tick_range)

    # 図の範囲を指定
    if is_lim:
        plt.xlim(min_value, max_value)  
        plt.ylim(min_value, max_value)

    # 図をファイルにする
    if figpath:
        plt.savefig(figpath)

def torch_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def init_gpu(local_rank, is_torchrun=False, world_size=torch.cuda.device_count()):
    torch.cuda.set_device(local_rank)
    if not is_torchrun:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
    # init
    dist.init_process_group(backend='nccl', init_method='env://', rank=local_rank, world_size=world_size)

    # GPUの数、ノード上のGPU数を取得し表示
    pid = os.getpid()
    world_size = dist.get_world_size() 
    n_gpus = torch.cuda.device_count() 
    print(f'[{pid}] Node info: GPUs: {n_gpus}, local_rank: {local_rank}, world_size: {world_size}')


def get_gpu_info(phase, nvidia_smi_path='nvidia-smi', no_units=False):
    print(f"--------------------- {phase} ----------------------")
    keys = (
    'timestamp',
    'memory.total',
    'memory.free',
    'memory.used',
    'utilization.gpu',
    'utilization.memory'
    )
    nu_opt = '' if not no_units else ',nounits'
    cmd = f'{nvidia_smi_path} --query-gpu={",".join(keys)} --format=csv,noheader{nu_opt}'
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]
    pprint([ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ])

    # return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]



