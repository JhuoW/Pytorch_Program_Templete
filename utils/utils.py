import yaml
import numpy as np
from sklearn import metrics
from torch_geometric.data import Data
import torch
from torch_sparse import SparseTensor
from torch_scatter import scatter_mean
import os.path as osp
from DataHelper.DatasetLocal import DatasetLocal

def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config


def load_data(args):
    path = osp.join(args.data_dir)
    dataset = DatasetLocal(args.dataset, "")
    dataset.dataset_source_folder_path = path
    dataset.recache = args.recache
    return dataset
    


def homophily(edge_index, y, method: str = 'edge'):
    assert method in ['edge', 'node']
    y = y.squeeze(-1) if y.dim() > 1 else y

    if isinstance(edge_index, SparseTensor):
        col, row, _ = edge_index.coo()
    else:
        row, col = edge_index

    if method == 'edge':
        return int((y[row] == y[col]).sum()) / row.size(0)  # out neigh 的同质率
    else:
        out = torch.zeros_like(row, dtype=float)
        out[y[row] == y[col]] = 1.
        out = scatter_mean(out, col, 0, dim_size=y.size(0))
        return float(out.mean())


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def getneighborslst(data: Data):
    ei = data.edge_index.numpy()
    lst = {i: set(ei[1][ei[0] == i]) for i in range(data.num_nodes)} 
    return lst

def get_device(cuda_id: int):
    device = torch.device('cuda' if cuda_id < 0 else 'cuda:%d' % cuda_id)
    return device


def Evaluation(output, labels):
    preds = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    '''
    binary_pred = preds
    binary_pred[binary_pred > 0.0] = 1
    binary_pred[binary_pred <= 0.0] = 0
    '''
    num_correct = 0
    binary_pred = np.zeros(preds.shape).astype('int')
    for i in range(preds.shape[0]):
        k = labels[i].sum().astype('int')
        topk_idx = preds[i].argsort()[-k:]
        binary_pred[i][topk_idx] = 1
        for pos in list(labels[i].nonzero()[0]):
            if labels[i][pos] and labels[i][pos] == binary_pred[i][pos]:
                num_correct += 1

    # print('total number of correct is: {}'.format(num_correct))
    #print('preds max is: {0} and min is: {1}'.format(preds.max(),preds.min()))
    #'''
    return metrics.f1_score(labels, binary_pred, average="micro"), metrics.f1_score(labels, binary_pred, average="macro")

def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def edge_index_to_torch_sparse_tensor(edge_index, edge_weight = None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),)).cuda()
    
    n_node = edge_index.max().item() + 1

    return torch.cuda.sparse.FloatTensor(edge_index, edge_weight, torch.Size((n_node, n_node)))