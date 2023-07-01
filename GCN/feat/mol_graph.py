import torch
from rdkit import Chem
from torch_geometric.data import Data
class MolGraph:

    def __init__(self, mols):
        self.mols = mols
        self.symbol_list, self.hybridization_list = self.__get_symbol_hybridization_types(mols)
        feature_list = []
        edge_index_list = []
        feature_size_list = []
        for mol in mols:
            feature = self.__get_feature_matrix(mol)
            adjacency_matrix = self.__get_adjacency_matrix(mol)
            edge_index = self.__get_edge_index(adjacency_matrix)
            feature_size_list.append(feature.shape[0])
            feature_list.append(feature)
            edge_index_list.append(edge_index)

        # self.x_data = [Data(x=feature, edge_index=edge_index, feature_size=feature_size) for feature, edge_index, feature_size, in zip(feature_list, edge_index_list, feature_size_list)]
        self.feature_list = feature_list
        self.edge_index_list = edge_index_list
        self.feature_size_list = feature_size_list
  
    def __get_feature_matrix(self, mol):
        atoms = mol.GetAtoms()
        feature = torch.zeros((len(atoms), len(self.symbol_list)+len(self.hybridization_list)+2))
        for i, atom in enumerate(atoms):
            feature[i, self.symbol_list.index(atom.GetSymbol())]
            feature[i, self.symbol_list.index(atom.GetSymbol())] += 1
            feature[i, len(self.symbol_list)] += atom.GetTotalValence()
            feature[i, len(self.symbol_list)+1] += atom.GetFormalCharge()
            feature[i, self.hybridization_list.index(atom.GetHybridization())+len(self.symbol_list)+2] += 1
        return feature

    def __get_adjacency_matrix(self, mol):
        return Chem.rdmolops.GetAdjacencyMatrix(mol)

    def __get_symbol_hybridization_types(self, mols):
        # 原子の種類と混成軌道の種類をsetを使用して把握する。
        symbol_set = set()
        hybridization_set = set()
        for mol in mols:
            atoms = mol.GetAtoms()
            for atom in atoms:
                symbol_set.add(atom.GetSymbol())
                hybridization_set.add(atom.GetHybridization())
        symbol_list = sorted(list(symbol_set))
        hybridization_list = sorted(list(hybridization_set))

        return symbol_list, hybridization_list



    def __get_edge_index(self, adjacency_matrix):
        adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.long)
        edge_index = adjacency_matrix.nonzero().t().contiguous()
        return edge_index
