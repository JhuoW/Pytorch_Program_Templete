from .dataset_helper import dataset
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

class DatasetLocal(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    mask = None
    feat_transform = None
    recache = False

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)


    def get_data_mask(self, idx = None):
        if idx is None:
            return self.data
        else:
            mask = idx
            data = self.data.clone()
            data.train_mask = self.data.train_mask[:,mask]
            data.val_mask = self.data.val_mask[:,mask]
            data.test_mask = self.data.test_mask[:, mask]
            return data


    def load(self,config):
        if config['feat_norm']:
            self.feat_transform = T.NormalizeFeatures()

        if self.dataset_name in ['Cora']:
            dataset = Planetoid(root=self.dataset_source_folder_path, name = self.dataset_name, transform=self.feat_transform)
        data = dataset[0]
        self.config_data(data, dataset)

    def config_data(self, data, dataset):
        self.data = data
        self.features = data.x
        self.nfeat = dataset.num_features
        self.edge_index = data.edge_index
        self.N = int(data.num_nodes)
        self.num_classes = int(dataset.num_classes)