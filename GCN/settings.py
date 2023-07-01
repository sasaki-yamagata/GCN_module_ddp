import os
import torch
from argparse import ArgumentParser
import torch.distributed as dist


def get_config(debug=False):
    """
    GCNのハイパーパラメータの設定を取得

    Parameters
    ----------
    debug : boolean
        Trueにするとデバックモードにできる
    
    Returns
    ----------
    config : dict
        GCNのハイパーパラメータ

    Notes
    ----------
    configについて
        start : integer
            抽出するデータの最初を指定
        end : integer
            抽出するデータの最後を指定
        device : string
            cpuとgpuのどちらで計算するかを指定
        n_epoch : integer
            エポック数を指定
        lr : float
            学習率を指定
        batch_size : integer
            バッチサイズを指定
        n_splits : integer
            交差検証する際の分割数を指定
        gc_layer_num : integer
            GCNの層数を指定
        affine_layer_num : integer
            線形回帰の層数を指定
        node_num : integer
            ノード数を指定 
        last_layer_node_num : integer
            最後の層のノード数を指定
    """
    
    if debug:
        config = {
            "start": None,
            "end": 1000,
            "device": 'cuda:0' if torch.cuda.is_available() else 'cpu',
            "n_epoch": 5,
            "lr": 0.001,
            "batch_size" : 12,
            "n_splits": 2, # 交差検証のときのみ使用  
            "gc_layer_num" : 3,
            "affine_layer_num" : 2,
            "node_num" : 4,
            "last_layer_node_num" : 6
        }
        
    else:
        config = {
            "start": None,
            "end": None,
            "device": 'cuda:0' if torch.cuda.is_available() else 'cpu',
            "n_epoch": 50,
            "lr": 0.001	,
            "batch_size" : 50,
            "n_splits": 3, # 交差検証のときのみ使用 
            "gc_layer_num" : 3,
            "affine_layer_num" : 1,
            "node_num" : 6	,
            "last_layer_node_num" : 9 
        }
        # [50, 0.000875, 50, 3, 5, 8, 6, 10] 5epochでnanが検出されたハイパーパラメータ
    # if multi_gpu:
    #     local_rank = init_gpu()
    #     config["device"] = f'cuda:{local_rank}'

    

    # if local_rank == 0:       
    # print(f"-------------  Device in this enviroment is {config['device']} -------------")
    return config

config = get_config(debug=True)



