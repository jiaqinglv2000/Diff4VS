import os
import os.path as osp
import pathlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.utils import subgraph

import src.utils as utils
from src.datasets.abstract_dataset import MolecularDataModule, AbstractDatasetInfos
from src.analysis.rdkit_functions import  mol2smiles, build_molecule_with_partial_charges
from src.analysis.rdkit_functions import compute_molecular_metrics

class RemoveYTransform:
    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data


class SelectHIVTransform:
    def __call__(self, data):
        return data



def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class HIVDataset(InMemoryDataset):
    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv')

    def __init__(self, stage, root, transform=None, pre_transform=None, pre_filter=None):
        """ stage: train, val, test
            root: data directory
            remove_h: remove hydrogens
            target_prop: property to predict (for guidance only).
        """
        self.stage = stage
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2

        # Atom_decoder should be consistent with the pre-trained model MOSES
        self.atom_decoder = ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H']

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        return ['HIV.csv']

    @property
    def split_file_name(self):
        return ['train.csv', 'val.csv', 'test.csv']

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        return ['proc_tr_h.pt', 'proc_val_h.pt', 'proc_test_h.pt']

    def download(self):
        """
        Download raw HIV files.
        """
        import rdkit  # noqa
        download_url(self.raw_url, self.raw_dir)

        if files_exist(self.split_paths):
            return

        data = pd.read_csv(self.raw_paths[0])

        # Remove molecules whose atoms exceed the pre-trained dataset range
        keep_rows = []
        for index, row in data.iterrows():
            smi = row['smiles']
            mol = Chem.MolFromSmiles(smi)

            if mol and 250 <= Descriptors.MolWt(mol) <= 350 and all(atom.GetSymbol() in self.atom_decoder for atom in mol.GetAtoms()):
                keep_rows.append(True)
            else:
                keep_rows.append(False)

        filtered_data = data[keep_rows]
        filtered_data = filtered_data.reset_index(drop=True)
        n_samples = len(filtered_data)

        n_train = 12000
        n_test = 1000
        n_val = n_samples - (n_train + n_test)

        filtered_data = filtered_data.sample(frac=1, random_state=42)

        pd_train = filtered_data.iloc[0:n_train]
        pd_train.reset_index(drop=True, inplace=True)
        pd_test = filtered_data.iloc[n_train:n_train+n_test]
        pd_test.reset_index(drop=True, inplace=True)
        pd_val = filtered_data.iloc[n_train+n_test:n_train+n_test+n_val]
        pd_val.reset_index(drop=True, inplace=True)


        train_active_count = len(pd_train[pd_train['HIV_active'] == 1])
        train_inactive_count = len(pd_train[pd_train['HIV_active'] == 0])

        r = train_inactive_count // train_active_count

        mask = pd_train['HIV_active'] == 1
        train_active_copy = pd_train[mask].copy()

        for _ in range(r - 1):
            pd_train = pd.concat([pd_train, train_active_copy])

        pd_train.reset_index(drop=True, inplace=True)

        pd_train = pd_train.sample(frac=1, random_state=42)
        pd_train.reset_index(drop=True, inplace=True)



        pd_train.to_csv(os.path.join(self.raw_dir, 'train.csv'))
        pd_val.to_csv(os.path.join(self.raw_dir, 'val.csv'))
        pd_test.to_csv(os.path.join(self.raw_dir, 'test.csv'))

    def process(self):
        RDLogger.DisableLog('rdApp.*')

        types = {atom: i for i, atom in enumerate(self.atom_decoder)}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        hiv_df = pd.read_csv(self.split_paths[self.file_idx], index_col=0)

        data_list = []
        for i, rcd in hiv_df.iterrows():
            smiles = rcd['smiles']
            label = rcd['HIV_active']
            mol = Chem.MolFromSmiles(smiles)

            N = mol.GetNumAtoms()

            type_idx = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()] + 1]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

            x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()

            #hiv active or not
            y = torch.tensor([label], dtype=torch.float32)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])



class HIVDataModule(MolecularDataModule):
    def __init__(self, cfg, regressor: bool = False):
        self.datadir = cfg.dataset.datadir
        self.regressor = regressor
        target = cfg.general.guidance_target
        if self.regressor and target == 'HIV':
            transform = SelectHIVTransform()
            print('case1')
        else:
            transform = RemoveYTransform()
            print('case2')

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': HIVDataset(stage='train', root=root_path,
                                        transform=transform if self.regressor else RemoveYTransform()),
                    'val': HIVDataset(stage='val', root=root_path,
                                      transform=transform if self.regressor else RemoveYTransform()),
                    'test': HIVDataset(stage='test', root=root_path,
                                       transform=transform)}
        super().__init__(cfg,datasets)
        self.remove_h = cfg.dataset.remove_h



class HIVinfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        self.need_to_strip = False        # to indicate whether we need to ignore one output from the model

        self.name = 'HIV'
        self.input_dims = None
        self.output_dims = None
        self.remove_h = False

        self.atom_decoder = ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H']
        self.atom_encoder = {atom: i for i, atom in enumerate(self.atom_decoder)}
        self.atom_weights = {0: 12, 1: 14, 2: 32, 3: 16, 4: 19, 5: 35.4, 6: 79.9, 7: 1}
        self.valencies = [4, 3, 4, 2, 1, 1, 1, 1]
        self.num_atom_types = len(self.atom_decoder)
        self.max_weight = 350

        meta_files = dict(n_nodes=f'{self.name}_n_counts.txt',
                          node_types=f'{self.name}_atom_types.txt',
                          edge_types=f'{self.name}_edge_types.txt',
                          valency_distribution=f'{self.name}_valencies.txt')

        self.n_nodes = torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.097634362347889692e-06,
                                     1.858580617408733815e-05, 5.007842264603823423e-05, 5.678996240021660924e-05,
                                     1.244216400664299726e-04, 4.486406978685408831e-04, 2.253012731671333313e-03,
                                     3.231865121051669121e-03, 6.709992419928312302e-03, 2.289564721286296844e-02,
                                     5.411050841212272644e-02, 1.099515631794929504e-01, 1.223291903734207153e-01,
                                     1.280680745840072632e-01, 1.445975750684738159e-01, 1.505961418151855469e-01,
                                     1.436946094036102295e-01, 9.265746921300888062e-02, 1.820066757500171661e-02,
                                     2.065089574898593128e-06])
        self.max_n_nodes = len(self.n_nodes) - 1 if self.n_nodes is not None else None
        self.node_types = torch.Tensor([0.722338, 0.13661, 0.163655, 0.103549, 0.1421803, 0.005411, 0.00150, 0.0])
        self.edge_types = torch.Tensor([0.89740, 0.0472947, 0.062670, 0.0003524, 0.0486])
        self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
        self.valency_distribution[:7] = torch.Tensor([0.0, 0.1055, 0.2728, 0.3613, 0.2499, 0.00544, 0.00485])
        super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)


def get_train_smiles(cfg, datamodule, dataset_infos, evaluate_dataset=False):
    base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
    smiles_path = os.path.join(base_path, cfg.dataset.datadir)

    train_smiles = None
    if os.path.exists(smiles_path):
        print("Dataset smiles were found.")
        train_smiles = np.array(open(smiles_path).readlines())

    if evaluate_dataset:
        train_dataloader = datamodule.dataloaders['train']
        all_molecules = []
        for i, data in enumerate(tqdm(train_dataloader)):
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E

            for k in range(X.size(0)):
                n = int(torch.sum((X != -1)[k, :]))
                atom_types = X[k, :n].cpu()
                edge_types = E[k, :n, :n].cpu()
                all_molecules.append([atom_types, edge_types])

        print("Evaluating the dataset -- number of molecules to evaluate", len(all_molecules))
        metrics = compute_molecular_metrics(molecule_list=all_molecules, train_smiles=train_smiles,
                                            dataset_info=dataset_infos)
        print(metrics[0])

    return train_smiles



