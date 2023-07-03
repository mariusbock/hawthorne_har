# ------------------------------------------------------------------------
# Methods used for training inertial-based models
# ------------------------------------------------------------------------
# Author: Marius Bock
# E-mail: marius.bock(at)uni-siegen.de
# ------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils import compute_class_weight
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from utils.data_utils import unwindow_inertial_data
from utils.torch_utils import init_weights, save_checkpoint, worker_init_reset_seed, InertialDataset
from utils.os_utils import mkdir_if_missing
from model.AttendAndDiscriminate import AttendAndDiscriminate
from model.DeepConvLSTM import DeepConvLSTM


def run_inertial_network(train_sbjs, val_sbjs, cfg, ckpt_folder, ckpt_freq, resume, rng_generator, run):
    split_name = '_'.join(cfg['dataset']['json_anno'].split('/')[-1].split('.')[0].split('_')[-2:])
    # load train and val inertial data
    train_data, val_data = np.empty((0, cfg['dataset']['input_dim'] + 2)), np.empty((0, cfg['dataset']['input_dim'] + 2))
    for t_sbj in train_sbjs:
        t_data = pd.read_csv(os.path.join(cfg['dataset']['sens_folder'], t_sbj + '.csv'), index_col=False, dtype = {'acc_x': float, 'acc_y': float, 'acc_z': float, 'label': str}).replace({"label": cfg['label_dict']}).fillna(0).to_numpy()
        train_data = np.append(train_data, t_data, axis=0)
    for v_sbj in val_sbjs:
        v_data = pd.read_csv(os.path.join(cfg['dataset']['sens_folder'], v_sbj + '.csv'), index_col=False, dtype = {'acc_x': float, 'acc_y': float, 'acc_z': float, 'label': str}).replace({"label": cfg['label_dict']}).fillna(0).to_numpy()
        val_data = np.append(val_data, v_data, axis=0)
    
    # define inertial datasets
    train_dataset = InertialDataset(train_data, cfg['dataset']['window_size'], cfg['dataset']['window_overlap'], cfg['dataset']['include_null'], cfg['dataset']['has_null'])
    test_dataset = InertialDataset(val_data, cfg['dataset']['window_size'], cfg['dataset']['window_overlap'], cfg['dataset']['include_null'], cfg['dataset']['has_null'])

    # define dataloaders
    train_loader = DataLoader(train_dataset, cfg['loader']['batch_size'], shuffle=True, num_workers=4, worker_init_fn=worker_init_reset_seed, generator=rng_generator, persistent_workers=True)
    test_loader = DataLoader(test_dataset, cfg['loader']['batch_size'], shuffle=False, num_workers=4, worker_init_fn=worker_init_reset_seed, generator=rng_generator, persistent_workers=True)
    
    # define network
    if cfg['name'] == 'deepconvlstm':
        net = DeepConvLSTM(
            train_dataset.channels, train_dataset.classes, train_dataset.window_size,
            cfg['model']['conv_kernels'], cfg['model']['conv_kernel_size'], 
            cfg['model']['lstm_units'], cfg['model']['lstm_layers'], cfg['model']['dropout']
            )

    # define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adadelta(net.parameters(), lr=cfg['train_cfg']['lr'], weight_decay=cfg['train_cfg']['weight_decay'])

    # use lr schedule if selected
    if cfg['train_cfg']['lr_step'] > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg['train_cfg']['lr_step'], gamma=cfg['train_cfg']['lr_decay'])
    
    # use weighted loss if selected
    if cfg['train_cfg']['weighted_loss']:
        class_weights = compute_class_weight('balanced', classes=np.unique(train_dataset.labels) + 1, y=train_dataset.labels + 1)
        criterion.weight = torch.tensor(class_weights).float().to(cfg['devices'][0])

    if resume:
        if os.path.isfile(resume):
            checkpoint = torch.load(resume, map_location = lambda storage, loc: storage.cuda(cfg['devices'][0]))
            start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            opt.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            return
    else:
        net = init_weights(net, cfg['train_cfg']['weight_init'])
        start_epoch = 0

    net.to(cfg['devices'][0])
    for epoch in range(start_epoch, cfg['train_cfg']['epochs']):
        # training
        net, t_losses, t_preds, t_gt = train_one_epoch(train_loader, net, opt, criterion, cfg['devices'][0], resume)

        # save ckpt once in a while
        if (((epoch + 1) == cfg['train_cfg']['epochs']) or ((ckpt_freq > 0) and ((epoch + 1) % ckpt_freq == 0))):
            save_states = { 
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': opt.state_dict(),
            }

            file_name = 'epoch_{:03d}_{}.pth.tar'.format(epoch + 1, split_name)
            save_checkpoint(save_states, False, file_folder=os.path.join(ckpt_folder, 'ckpts'), file_name=file_name)

        # validation
        v_losses, v_preds, v_gt = validate_one_epoch(test_loader, net, criterion, cfg['devices'][0])

        if cfg['train_cfg']['lr_step'] > 0:
            scheduler.step()
        
        # undwindow inertial data (sample-wise structure instead of windowed) 
        #v_preds, v_gt = unwindow_inertial_data(val_data, test_dataset.ids, v_preds, cfg['dataset']['window_size'], cfg['dataset']['window_overlap'])
        """t_preds, t_gt = unwindow_inertial_data(train_data, train_dataset.ids, t_preds, cfg['dataset']['window_size'], cfg['dataset']['window_overlap'])"""

        if epoch == (start_epoch + cfg['train_cfg']['epochs']) - 1:
            # save raw results (for later postprocessing)
            mkdir_if_missing(os.path.join(ckpt_folder, 'unprocessed_results'))
            np.save(os.path.join(ckpt_folder, 'unprocessed_results', 'v_preds_' + split_name), v_preds)
            np.save(os.path.join(ckpt_folder, 'unprocessed_results', 'v_gt_' + split_name), v_gt)
        
        # calculate training metrics
        t_conf_mat = confusion_matrix(t_gt, t_preds, normalize='true')
        t_acc = t_conf_mat.diagonal()/t_conf_mat.sum(axis=1)
        t_prec = precision_score(t_gt, t_preds, average=None, zero_division=1)
        t_rec = recall_score(t_gt, t_preds, average=None, zero_division=1)
        t_f1 = f1_score(t_gt, t_preds, average=None, zero_division=1)
        
        # calculate validation metrics
        conf_mat = confusion_matrix(v_gt, v_preds, normalize='true')
        v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
        v_prec = precision_score(v_gt, v_preds, average=None, zero_division=1)
        v_rec = recall_score(v_gt, v_preds, average=None, zero_division=1)
        v_f1 = f1_score(v_gt, v_preds, average=None, zero_division=1)

        # print results to terminal
        block1 = 'Epoch: [{:03d}/{:03d}]'.format(epoch, cfg['train_cfg']['epochs'])
        block2 = 'TRAINING:\tavg. loss {:.2f}'.format(np.nanmean(t_losses))
        block3 = ''
        block3  += '\t\tAcc {:>4.2f} (%)'.format(np.nanmean(t_acc) * 100)
        block3  += ' Prec {:>4.2f} (%)'.format(np.nanmean(t_prec) * 100)
        block3  += ' Rec {:>4.2f} (%)'.format(np.nanmean(t_rec) * 100)
        block3  += ' F1 {:>4.2f} (%)'.format(np.nanmean(t_f1) * 100)
        block4 = 'VALIDATION:\tavg. loss {:.2f}'.format(np.nanmean(v_losses))
        block5 = ''
        block5  += '\t\tAcc {:>4.2f} (%)'.format(np.nanmean(v_acc) * 100)
        block5  += ' Prec {:>4.2f} (%)'.format(np.nanmean(v_prec) * 100)
        block5  += ' Rec {:>4.2f} (%)'.format(np.nanmean(v_rec) * 100)
        block5  += ' F1 {:>4.2f} (%)'.format(np.nanmean(v_f1) * 100)

        print('\n'.join([block1, block2, block3, block4, block5]))

        if run is not None:
            run[split_name].append({"train_loss": np.nanmean(t_losses), "val_loss": np.nanmean(v_losses), "accuracy": np.nanmean(v_acc), "precision": np.nanmean(v_prec), "recall": np.nanmean(v_rec), 'f1': np.nanmean(v_f1)}, step=epoch)

    return t_losses, t_preds, t_gt, v_losses, v_preds, v_gt


def train_one_epoch(loader, network, opt, criterion, gpu=None, resume=False):
    losses, preds, gt = [], [], []

    network.train()
    for i, (inputs, targets) in enumerate(loader):
        if gpu is not None:
            inputs, targets = inputs.to(gpu), targets.to(gpu)
        
        output = network(inputs)
        batch_loss = criterion(output, targets)

        opt.zero_grad()
        batch_loss.backward()
        opt.step()
        # append train loss to list
        losses.append(batch_loss.item())

        # create predictions and append them to final list
        batch_preds = np.argmax(output.cpu().detach().numpy(), axis=-1)
        batch_gt = targets.cpu().numpy().flatten()
        preds = np.concatenate((preds, batch_preds))
        gt = np.concatenate((gt, batch_gt))
    
    return network, losses, preds, gt


def validate_one_epoch(loader, network, criterion, gpu=None):
    losses, preds, gt = [], [], []

    network.eval()
    with torch.no_grad():
        # iterate over validation dataset
        for i, (inputs, targets) in enumerate(loader):
            # send x and y to GPU
            if gpu is not None:
                inputs, targets = inputs.to(gpu), targets.to(gpu)

            # send inputs through network to get predictions, loss and calculate softmax probabilities
            output = network(inputs)
            batch_loss = criterion(output, targets.long())
            losses.append(batch_loss.item())

            # create predictions and append them to final list
            batch_preds = np.argmax(output.cpu().detach().numpy(), axis=-1)
            batch_gt = targets.cpu().numpy().flatten()
            preds = np.concatenate((preds, batch_preds))
            gt = np.concatenate((gt, batch_gt))
    return losses, preds, gt

