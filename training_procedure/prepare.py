import torch
import torch.nn as nn
from importlib import import_module
import os.path as osp
from DataHelper.DatasetLocal import DatasetLocal

def prepare_train(self, model):
    config = self.config
    optimizer = getattr(torch.optim, config['optimizer'])(  params          = model.parameters(), 
                                                            lr              = config['lr'] ,
                                                            weight_decay    = config.get('weight_decay', 0) )
    if config.get('lr_scheduler', False):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['step_size'],gamma=config['gamma'])
    loss_func = nn.CrossEntropyLoss()
    return optimizer, loss_func

def prepare_model(self, dataset):
    config = self.config
    model_name = config['model_name']
    if model_name in ["GCN"]:
        Model_Class = getattr(import_module("model.GCN"), model_name)
        model = Model_Class(config, config['hidden_dim'], dataset.nfeat, dataset.num_classes).cuda()
    if model_name in ["GAT"]:
        Model_Class = getattr(import_module("model.GAT"), model_name)
        model = Model_Class(config, config['hidden_dim'], dataset.nfeat, dataset.num_classes).cuda()
    return model

def load_data(self, idx):
    args = self.args
    path = osp.join("datasets/")
    dataset = DatasetLocal(args.dataset, "")
    dataset.dataset_source_folder_path = path
    dataset.mask = idx
    data = dataset.load()
    return data, dataset

def init(self, dataset):
    config = self.config
    model = self.prepare_model(dataset)
    optimizer, loss_func = self.prepare_train(model)
    
    return model, optimizer, loss_func