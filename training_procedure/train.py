
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm

def train(self, graph, model, loss_func, optimizer, dataset = None):
    model.train(True)
    config = self.config 
    edge_index = graph.edge_index.cuda()
    x = graph.x.cuda()
    y = graph.y.cuda()
    graph = graph.to(x.device)
    optimizer.zero_grad()
    if config['model_name'] in [ 'GCN' ,'GAT']:
        out = model(x, edge_index)

    if self.config['multilabel']:
        loss = nn.BCEWithLogitsLoss()(out[graph.train_mask], y[graph.train_mask])
    else:
        loss = loss_func(out[graph.train_mask], y[graph.train_mask])
    loss.backward()
    optimizer.step()
    
    return model, float(loss)