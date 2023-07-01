import numpy as np
import os
import random
import torch
import pandas as pd
from glob import glob
from rdkit import Chem
from tqdm import tqdm
from torch_geometric.data import Data
from GCN.settings import config

class SMILESDataLoader:

    def __init__(self, pickle_path, target_props, max_atom=None, smiles_col="smiles", start=0, end=None):
        if end is None:
            df = pd.read_pickle(pickle_path)
        else:
            df = pd.read_pickle(pickle_path)[start:end]
            
        targets = torch.zeros((0, len(target_props)), dtype=torch.float)
        mols = []

        for smiles, target in zip(df[smiles_col].values, df[target_props].values):

            mol = Chem.MolFromSmiles(smiles)

            target = torch.tensor(target, dtype=torch.float).view(1, -1)
            if mol is not None:
                if max_atom is not None:
                    atom_num = len(mol.GetAtoms())
                    if atom_num > max_atom:
                        print(f"atoms num is more than {max_atom}")
                        continue

                mols.append(mol)
                targets = torch.cat([targets, target], dim=0)
            else:
                continue
        del df      
        self.mols = mols
        self.targets = targets

class DataFrameLoader:
    """
    DataFrameLoaderクラスは、DataFrameからデータをロードするためのクラスです。
    """

    def __init__(self, df_path, target_props, end=None, start=None):
        """
        DataFrameLoaderクラスのコンストラクタです。

        Parameters
        ----------
        df_path : str
            DataFrameのパス
        target_props : list
            目的変数のカラム名のリスト
        end : int, optional
            DataFrameの終了インデックス, by default None
        start : int, optional
            DataFrameの開始インデックス, by default None
        """
        # DataFrameの読み込み
        if end is not None and start is not None:
            df_data = pd.read_pickle(df_path)[start:end]
        elif end is not None:
            df_data = pd.read_pickle(df_path)[:end]
        elif start is not None:
            df_data = pd.read_pickle(df_path)[start:]
        else:
            df_data = pd.read_pickle(df_path)
        
        # 目的変数の準備
        y = torch.tensor(df_data[target_props].values.astype(np.float32))

        # 特徴データの準備
        x_data = [
            Data(
                x=feature,
                edge_index=edge_index,
                feature_size=torch.tensor(feature_size)
            )
            
            for feature, edge_index, feature_size in zip(
                df_data["feature_matrix"].values,
                df_data["edge_index"].values,
                df_data["feature_size"].values
            )
        ]

        # モルコードの準備
        self.molcode = df_data["molcode"].values

        # データのセットアップ
        self.x_data = x_data
        self.y = y

        # 不要なデータの削除
        del df_data

class MolOriginal:
    '''
    mol_originalを扱いやすくするためのクラス
    '''

    def __init__(self, filepath, max_atom=None, end=None, start=None, add_hs=False, is_random=False):
        '''
        Info:
            mol_originalのmolファイルから直接読み込む。そのため時間がかかる。
        Args:
            filepath {str} -- mol_originalのmolファイルのパス指定
            add_hs {bool} -- 水素の有無
            is_random {bool} -- ランダムで読み込める
        '''
        if end is not None and start is not None:
            molfiles = glob(fR"{filepath}/*.mol")[start:end]
        elif end is not None:
            molfiles = glob(fR"{filepath}/*.mol")[:end]
        elif start is not None:
            molfiles = glob(fR"{filepath}/*.mol")[start:]
        else:
            molfiles = glob(fR"{filepath}/*.mol")
        
        if is_random:
            random.shuffle(molfiles)
        

        total = len(molfiles)
        self.molcodes = []
        self.mols = []
        self.smiles = []
        
        for molfile in tqdm(molfiles, total=total):
           
            mol = Chem.MolFromMolFile(molfile)
            # rdkitで読み込む際にNoneになるものをはじく
            if mol is None:
                continue

            if max_atom is not None:
                atom_num = len(mol.GetAtoms())
                if atom_num > max_atom:
                    print(f"molcode{molcode} is more than {max_atom}")
                    continue

            if add_hs:
                try:
                    mol_addhs = Chem.AddHs(mol)     
                except:
                    continue
                self.mols.append(mol_addhs)
            else:
                self.mols.append(mol)
                           
            filename = os.path.split(molfile)[-1]
            molcode = filename.split('_')[0]
            self.molcodes.append(molcode)
            self.smiles.append(Chem.MolToSmiles(mol))






