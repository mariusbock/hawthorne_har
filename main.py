# ------------------------------------------------------------------------
# Main script to commence baseline experiments
# ------------------------------------------------------------------------
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
# ------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import argparse
import datetime
import json
import os
from pprint import pprint
import sys
import time

import pandas as pd
import numpy as np
import neptune
from neptune.types import File
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from model.train import run_inertial_network
from utils.torch_utils import fix_random_seed
from utils.os_utils import Logger, load_config
import matplotlib.pyplot as plt


def main(args):
    if args.neptune:
        run = neptune.init_run(
        project="",
        api_token="",
    )
    else:
        run = None

    config = load_config(args.config)
    config['init_rand_seed'] = args.seed
    config['devices'] = [args.gpu]

    ts = datetime.datetime.fromtimestamp(int(time.time()))
    log_dir = os.path.join('logs', config['name'], str(ts))
    sys.stdout = Logger(os.path.join(log_dir, 'log.txt'))

    # save the current cfg
    with open(os.path.join(log_dir, 'cfg.txt'), 'w') as fid:
        pprint(config, stream=fid)
        fid.flush()
    
    if args.neptune:
        run['eval_type'] = args.eval_type
        run['config_name'] = args.config
        run['seed'] = args.seed
        run['config'].upload(os.path.join(log_dir, 'cfg.txt'))
        run['dataset'] = config['dataset_name']
        run['network'] = config['name']

    rng_generator = fix_random_seed(config['init_rand_seed'], include_cuda=True)    

    all_t_pred = np.array([])
    all_t_gt = np.array([])
    all_v_pred = np.array([])
    all_v_gt = np.array([])
        
    for i, anno_split in enumerate(config['anno_json']):
        with open(anno_split) as f:
            file = json.load(f)
        anno_file = file['database']
        config['labels'] = list(file['label_dict'])
        config['label_dict'] = dict(zip(config['labels'], list(range(len(config['labels'])))))
        train_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Training']
        val_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Validation']
        
        print('Split {} / {}'.format(i + 1, len(config['anno_json'])))
        if args.eval_type == 'split':
            name = 'split_' + str(i)
        elif args.eval_type == 'loso':
            name = 'sbj_' + str(i)
        config['dataset']['json_anno'] = anno_split

        t_losses, t_preds, t_gt, v_losses, v_preds, v_gt = run_inertial_network(train_sbjs, val_sbjs, config, log_dir, args.ckpt_freq, args.resume, rng_generator, run)

        # unprocessed results
        t_conf_mat = confusion_matrix(t_gt, t_preds, normalize='true', labels=range(len(config['labels'])))
        t_acc = t_conf_mat.diagonal()/t_conf_mat.sum(axis=1)
        t_prec = precision_score(t_gt, t_preds, average=None, zero_division=1, labels=range(len(config['labels'])))
        t_rec = recall_score(t_gt, t_preds, average=None, zero_division=1, labels=range(len(config['labels'])))
        t_f1 = f1_score(t_gt, t_preds, average=None, zero_division=1, labels=range(len(config['labels'])))
        
        # unprocessed results
        conf_mat = confusion_matrix(v_gt, v_preds, normalize='true', labels=range(len(config['labels'])))
        v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
        v_prec = precision_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(config['labels'])))
        v_rec = recall_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(config['labels'])))
        v_f1 = f1_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(config['labels'])))

        # print to terminal
        if args.eval_type == 'split':
            block1 = '\nFINAL RESULTS SPLIT {}'.format(i + 1)
        elif args.eval_type == 'loso':
            block1 = '\nFINAL RESULTS SUBJECT {}'.format(i)
        block2 = 'TRAINING:\tavg. loss {:.2f}'.format(np.nanmean(t_losses))
        block3 = ''
        block3  += '\t\tAcc {:>4.2f} (%)'.format(np.nanmean(t_acc) * 100)
        block3  += ' Prec {:>4.2f} (%)'.format(np.nanmean(t_prec) * 100)
        block3  += ' Rec {:>4.2f} (%)'.format(np.nanmean(t_rec) * 100)
        block3  += ' F1 {:>4.2f} (%)\n'.format(np.nanmean(t_f1) * 100)
        block4 = 'VALIDATION:\tavg. loss {:.2f}'.format(np.nanmean(v_losses))
        block5 = ''
        block5  += '\t\tAcc {:>4.2f} (%)'.format(np.nanmean(v_acc) * 100)
        block5  += ' Prec {:>4.2f} (%)'.format(np.nanmean(v_prec) * 100)
        block5  += ' Rec {:>4.2f} (%)'.format(np.nanmean(v_rec) * 100)
        block5  += ' F1 {:>4.2f} (%)\n'.format(np.nanmean(v_f1) * 100)

        print('\n'.join([block1, block2, block3, block4, block5]))
                                
        all_t_gt = np.append(all_t_gt, t_gt)
        all_t_pred = np.append(all_t_pred, t_preds)
        all_v_gt = np.append(all_v_gt, v_gt)
        all_v_pred = np.append(all_v_pred, v_preds)

        # save unprocessed confusion matrix
        _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
        conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=config['labels'])
        conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
        ax.set_title('Confusion Matrix ' + name + ' (unprocessed)')
        plt.savefig(os.path.join(log_dir, name + '_unprocessed.png'))
        if run is not None:
            run['conf_matrices'].append(_, name=name + '_unprocessed')
        plt.close()

    # final unprocessed results across all splits
    t_conf_mat = confusion_matrix(all_t_gt, all_t_pred, normalize='true', labels=range(len(config['labels'])))
    t_acc = t_conf_mat.diagonal()/t_conf_mat.sum(axis=1)
    t_prec = precision_score(all_t_gt, all_t_pred, average=None, zero_division=1, labels=range(len(config['labels'])))
    t_rec = recall_score(all_t_gt, all_t_pred, average=None, zero_division=1, labels=range(len(config['labels'])))
    t_f1 = f1_score(all_t_gt, all_t_pred, average=None, zero_division=1, labels=range(len(config['labels'])))

    # final unprocessed results across all splits
    conf_mat = confusion_matrix(all_v_gt, all_v_pred, normalize='true', labels=range(len(config['labels'])))
    v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
    v_prec = precision_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))
    v_rec = recall_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))
    v_f1 = f1_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))

    # print final results to terminal
    block1 = '\nFINAL AVERAGED RESULTS:'
    block2 = 'TRAINING'
    block2  += '\t\tAcc {:>4.2f} (%)'.format(np.nanmean(t_acc) * 100)
    block2  += ' Prec {:>4.2f} (%)'.format(np.nanmean(t_prec) * 100)
    block2  += ' Rec {:>4.2f} (%)'.format(np.nanmean(t_rec) * 100)
    block2  += ' F1 {:>4.2f} (%)'.format(np.nanmean(t_f1) * 100)
    block3 = 'VALIDATION'
    block3  += '\t\tAcc {:>4.2f} (%)'.format(np.nanmean(v_acc) * 100)
    block3  += ' Prec {:>4.2f} (%)'.format(np.nanmean(v_prec) * 100)
    block3  += ' Rec {:>4.2f} (%)'.format(np.nanmean(v_rec) * 100)
    block3  += ' F1 {:>4.2f} (%)'.format(np.nanmean(v_f1) * 100)
    
    print('\n'.join([block1, block2, block3]))

    # save final unprocessed confusion matrix
    _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
    ax.set_title('Confusion Matrix Total (unprocessed)')
    conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=config['labels']) 
    conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
    plt.savefig(os.path.join(log_dir, 'all_unprocessed.png'))
    if run is not None:
        run['conf_matrices'].append(File(os.path.join(log_dir, 'all_unprocessed.png')), name='all')
    plt.close()
    
    # submit final values to neptune 
    if run is not None:
        run['final_accuracy'] = np.nanmean(v_acc)
        run['final_precision'] = (np.nanmean(v_prec))
        run['final_recall'] = (np.nanmean(v_rec))
        run['final_f1'] = (np.nanmean(v_f1))

    print("ALL FINISHED")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/deepconvlstm_0_vs_1_split.yaml')
    parser.add_argument('--eval_type', default='loso')
    parser.add_argument('--neptune', default=False, action='store_true',) 
    parser.add_argument('--seed', default=42, type=int)       
    parser.add_argument('--ckpt-freq', default=-1, type=int)
    parser.add_argument('--post-freq', default=5, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--gpu', default='cuda:0', type=str)
    args = parser.parse_args()
    main(args)  

