import torch
from sklearn.metrics import f1_score
import torch.nn as nn
from utils.utils import Evaluation

@torch.no_grad()
def get_eval_result(self, labels, pred_l, loss):

    if self.config['multilabel']:
        micro , macro = Evaluation(pred_l , labels)
    else:
        micro = f1_score(labels.cpu(), pred_l.cpu(), average = "micro")
        macro = 0

    return {
        "micro": round(micro * 100 , 2) , # to percentage
        "macro": round(macro * 100 , 2)
    }


@torch.no_grad()
def evaluate(self, graph, model, loss_func, dataset = None):
    model.eval()
    edge_index = graph.edge_index.cuda()
    x = graph.x.cuda()
    y = graph.y.cuda()

    if self.config['model_name'] in ['GCN','GAT']:
        out = model(x, edge_index)

    results = []
    val_test_loss = []
    for prefix in ["val", 'test']:
        mask = f'{prefix}_mask'

        if self.config['multilabel']:
            loss = nn.BCEWithLogitsLoss()(out[graph[mask]], y[graph[mask]])
            pred_l = out[graph[mask]]
        else:
            loss = loss_func(out[graph[mask]], y[graph[mask]])
            pred_l = out[graph[mask]].argmax(-1)
        results.append(get_eval_result(self, y[graph[mask]], pred_l, loss.item()))
        val_test_loss.append(loss)
    return results, val_test_loss