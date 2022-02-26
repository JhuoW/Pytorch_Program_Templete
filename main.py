import os.path as osp
from argparse import ArgumentParser
import sys
from tqdm import tqdm
import torch
import random
from utils.logger import Logger
from utils.utils import *
from utils.random_seeder import set_random_seed
from training_procedure import Trainer
from torch_geometric.transforms import ToUndirected


def main(args, config, logger: Logger, run_id: int, dataset):
    T = Trainer(config=config, args= args, logger= logger)
    data = dataset.get_data_mask()
    if config.get('to_undirected', False):
        data = ToUndirected()(data)
    model, optimizer, loss_func = T.init(dataset)   # model of current split

    pbar = tqdm(range(config['epochs']), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    patience_cnt 		= 0
    maj_metric 			= "micro"   # or macro
    best_metric 	  	= 0
    best_metric_epoch 	= -1 # best number on dev set
    report_dev_res 		= 0
    report_tes_res 		= 0
    best_dev_loss       = 100000.0
    val_loss_history = []
    for epoch in pbar:
        model, loss = T.train(data, model, loss_func, optimizer, dataset)

        (dev_result, test_result), (dev_loss, test_loss) = T.evaluation(data, model, loss_func, dataset)  # return 2 list, 
        now_metric = dev_result[maj_metric]    # acc of val set
        if config['monitor'] == 'val_acc':
            if args.no_dev or best_metric  <= now_metric: 
                best_metric         = now_metric
                best_metric_epoch   = epoch
                report_dev_res      = dev_result
                report_tes_res      = test_result

                patience_cnt        = 0    
            else:
                patience_cnt        += 1
        elif config['monitor'] == 'val_loss' or config['monitor'] == 'val_loss_avg':
            if args.no_dev or dev_loss <= best_dev_loss:
                best_dev_loss           = dev_loss
                best_metric_epoch       = epoch
                report_dev_res          = dev_result  
                report_tes_res          = test_result
                patience_cnt            = 0
            else:
                patience_cnt            += 1

            if config['monitor'] == 'val_loss_avg':
                val_loss_history.append(dev_loss)
                if config.get('patience', -1) > 0 and epoch > config['epochs'] // 2:
                    # from digcn
                    tmp = torch.tensor(val_loss_history[-(config['patience'] + 1):-1])
                    if dev_loss > tmp.mean().item():
                        break
        if config['patience'] > 0 and patience_cnt >= config['patience'] and config['monitor'] in ['val_acc', 'val_loss']:
            break
        postfix_str = "<Epoch %d> [Train Loss] %.4f [Curr Dev Acc] %.2f <Best Epoch %d> [Best Dev Acc] %.2f [Test] %.2f ([Report Test] %.2f) " % ( 
                        epoch ,      loss,         dev_result[maj_metric], best_metric_epoch ,report_dev_res[maj_metric], test_result[maj_metric], report_tes_res[maj_metric])

        pbar.set_postfix_str(postfix_str)
    logger.log("best epoch is %d" % best_metric_epoch)
    logger.log("Best Epoch Valid Acc is %.2f" % (report_dev_res[maj_metric]))
    logger.log("Best Epoch Test  Acc is %.2f" % (report_tes_res[maj_metric]))
    return model,  report_dev_res, report_tes_res, loss

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default= 'Cora') 
    parser.add_argument('--num_workers', default=8, type=int, choices=[0,8])
    parser.add_argument('--seed', default=1234, type=int, choices=[0, 1, 1234])
    parser.add_argument('--data_dir', type= str, default="datasets/") 
    parser.add_argument('--hyper_file', type=str, default= 'config/')
    parser.add_argument('--recache', action="store_true", help="clean up the old adj data", default=True)   
    parser.add_argument('--no_dev', action = "store_true" , default = False)
    parser.add_argument('--patience', type = int  , default = -1)
    parser.add_argument('--gpu_id', type = int  , default = 0)
    parser.add_argument('--model', type = str, default='GCN')  # GCN, GAT or other
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)

    logger = Logger(mode = [print])  
    logger.add_line = lambda : logger.log("-" * 50)
    logger.log(" ".join(sys.argv))
    logger.add_line()
    logger.log()

    config_path = osp.join(args.hyper_file, args.dataset + '.yml')
    config = get_config(config_path)
    model_name = args.model
    config = config[model_name] 
    config['model_name'] = model_name
    dev_ress = []
    tes_ress = []
    tra_ress = []
    if config.get('seed',-1) > 0:
        set_random_seed(config['seed'])
        logger.log ("Seed set. %d" % (config['seed']))
    seeds = [random.randint(0,233333333) for _ in range(config['multirun'])]
    dataset = load_data(args)
    dataset.load(config)  # config dataset
    print_config(config)
    all_org_wei = []
    all_gen_wei = []

    for run_id in range(config['multirun']):   # one mask
        logger.add_line()
        logger.log ("\t\t%d th Run" % run_id)
        logger.add_line()
        # set_random_seed(seeds[run_id])
        # logger.log ("Seed set to %d." % seeds[run_id])

        model,  report_dev_res, report_tes_res, loss = main(args, config, logger, run_id, dataset)