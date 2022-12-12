import os
import torch
import numpy as np
from tqdm import tqdm
from torch import no_grad
from data.uieb import UIEBTrain, UIEBValid
from torch.utils.data import DataLoader
from torch.optim import AdamW
from optim.lr_scheduler.cosine import CosineScheduler
from torch.cuda.amp import autocast, GradScaler
from util.common_utils import (MetricRecorder, Logger,
                               set_all_seed, make_all_dirs,
                               save_pics, print_epoch_result)
from util.train_utils import save_all, load_all
from kornia.metrics import ssim, psnr
from loss.charbonnier import CharbonnierL1Loss
from loss.perceptual import PerceptualLoss
from loss.ssim import SSIMLoss


def configuration_optimizer(model, hparams):
    optimizer = AdamW(
        params=model.parameters(),
        lr=hparams['optim']['lr_init'],
        weight_decay=hparams['optim']['weight_decay']
    )
    scheduler = CosineScheduler(
        optimizer=optimizer,
        param_name='lr',
        t_max=hparams['teacher_train']['max_epochs'] + 10,
        value_min=hparams['optim']['lr_min'],
        warmup_t=hparams['optim']['warmup_epochs'],
        const_t=0
    )
    return optimizer, scheduler


def configuration_dataloader(hparams):
    train_dataset = UIEBTrain(
        folder=hparams['data']['train_path'],
        size=hparams['data']['train_img_size']
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=hparams['data']['train_batch_size'],
        shuffle=True,
        num_workers=hparams['data']['num_workers'],
        pin_memory=hparams['data']['pin_memory']
    )
    valid_dataset = UIEBValid(
        folder=hparams['data']['valid_path'],
        size=hparams['data']['valid_img_size']
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        num_workers=hparams['data']['num_workers'],
        pin_memory=hparams['data']['pin_memory']
    )
    return train_loader, valid_loader


def train_one_epoch(train_loader, model, optimizer):
    model.train()
    # ssim_loss = SSIMLoss().cuda()
    loss_recorder = MetricRecorder()
    for batch in tqdm(train_loader, ncols=120):
        optimizer.zero_grad()
        l2_loss = CharbonnierL1Loss()
        l1_loss = PerceptualLoss()
        source_img, target_img = batch
        source_img = source_img.cuda()
        target_img = target_img.cuda()
        output_img = model(source_img)
        loss =0.11 * l1_loss(output_img,target_img) + 0.9 * l2_loss(output_img,target_img)
        loss.backward()
        optimizer.step()
        loss_recorder.update(loss.item())
    return {'train_loss': loss_recorder.avg,
            'lr': optimizer.param_groups[0]['lr']}


def valid_one_epoch(valid_loader, model, hparams):
    model.eval()
    l1_loss = PerceptualLoss().cuda()
    # ssim_loss = SSIMLoss().cuda()
    loss_recorder = MetricRecorder()
    ssim_recorder = MetricRecorder()
    psnr_recorder = MetricRecorder()
    for i, batch in enumerate(valid_loader):
        source_img, target_img = batch
        source_img = source_img.cuda()
        target_img = target_img.cuda()
        with no_grad():
            output_img = model(source_img)
            output_img = output_img.clamp(0, 1)
        loss = l1_loss(output_img, target_img)
        loss_recorder.update(loss.item())
        ssim_recorder.update(ssim(output_img, target_img, 5).mean().item())
        psnr_recorder.update(psnr(output_img, target_img, 1).item())
        if i % 10 == 0:
            save_pics(hparams, source_img, target_img, output_img)
    return {'valid_loss': loss_recorder.avg, 'ssim': ssim_recorder.avg,
            'psnr': psnr_recorder.avg}


def teacher_train(model, hparams):
    set_all_seed(hparams['teacher_train']['seed'])
    make_all_dirs(hparams)
    train_loader, valid_loader = configuration_dataloader(hparams)
    optimizer, scheduler = configuration_optimizer(model, hparams)
    best_metric = {'ssim': {'value': .0, 'epoch': 0},
                   'psnr': {'value': .0, 'epoch': 0}}
    logger = Logger(os.path.join(hparams['teacher_train']['save_dir'],
                                 hparams['teacher_train']['model_name'],
                                 hparams['teacher_train']['task_name'],
                                 'tensorboard'))

    if input('Whether to resume to training: ') == 'yes':
        print('==========>Start Resume<==========')
        start_epoch = load_all(hparams, hparams['teacher_train']['ckpt_name'], model,
                               optimizer, scheduler, best_metric) + 1
    else:
        print('==========>Start Training<==========')
        start_epoch = 1

    print('Since from {}th Epoch'.format(start_epoch))
    for current_epoch in range(start_epoch, hparams['teacher_train']['max_epochs'] + 1):
        # train
        train_return = train_one_epoch(train_loader, model, optimizer)
        logger.log_multi_scaler(train_return, current_epoch)
        scheduler.step(current_epoch)
        # valid
        valid_result = None
        if current_epoch % hparams['teacher_train']['valid_frequency'] == 0:
            valid_result = valid_one_epoch(valid_loader, model, hparams)
            logger.log_multi_scaler(valid_result, current_epoch)
            if valid_result['ssim'] > best_metric['ssim']['value']:
                best_metric['ssim']['value'] = valid_result['ssim']
                best_metric['ssim']['epoch'] = current_epoch
                save_all(current_epoch, model, optimizer, scheduler,
                         hparams, best_metric, 'best_ssim')
            if valid_result['psnr'] > best_metric['psnr']['value']:
                best_metric['psnr']['value'] = valid_result['psnr']
                best_metric['psnr']['epoch'] = current_epoch
                save_all(current_epoch, model, optimizer, scheduler,
                         hparams, best_metric, 'best_psnr')
        save_all(current_epoch, model, optimizer, scheduler,
                 hparams, best_metric, 'last')
        print_epoch_result(train_return, valid_result, current_epoch)
        print('best ssim: ', best_metric['ssim']['value'], '  best epoch: ', best_metric['ssim']['epoch'])
        print('best psnr: ', best_metric['psnr']['value'], '  best epoch: ', best_metric['psnr']['epoch'])


# def distill_train(model, hparams):
#     set_all_seed(hparams['distill_train']['seed'])
#     make_all_dirs(hparams)
#     train_loader, valid_loader = configuration_dataloader(hparams)
#     optimizer, scheduler = configuration_optimizer(model, hparams)
#     best_metric = {'ssim': {'value': .0, 'epoch': 0},
#                    'psnr': {'value': .0, 'epoch': 0}}
#     logger = Logger(os.path.join(hparams['distill_train']['save_dir'],
#                                  hparams['distill_train']['model_name'],
#                                  hparams['distill_train']['task_name'],
#                                  'tensorboard'))
#
#     if input('Whether to resume to training: ') == 'yes':
#         print('==========>Start Resume<==========')
#         start_epoch = load_all(hparams, hparams['distill_train']['ckpt_name'], model,
#                                optimizer, scheduler, best_metric) + 1
#     else:
#         print('==========>Start Training<==========')
#         start_epoch = 1
#
#     print('Since from {}th Epoch'.format(start_epoch))
#     for current_epoch in range(start_epoch, hparams['distill_train']['max_epochs'] + 1):
#         # train
#         train_return = train_one_epoch(train_loader, model, optimizer)
#         logger.log_multi_scaler(train_return, current_epoch)
#         scheduler.step(current_epoch)
#         # valid
#         valid_result = None
#         if current_epoch % hparams['distill_train']['valid_frequency'] == 0:
#             valid_result = valid_one_epoch(valid_loader, model, hparams)
#             logger.log_multi_scaler(valid_result, current_epoch)
#             if valid_result['ssim'] > best_metric['ssim']['value']:
#                 best_metric['ssim']['value'] = valid_result['ssim']
#                 best_metric['ssim']['epoch'] = current_epoch
#                 save_all(current_epoch, model, optimizer, scheduler,
#                          hparams, best_metric, 'best_ssim')
#             if valid_result['psnr'] > best_metric['psnr']['value']:
#                 best_metric['psnr']['value'] = valid_result['psnr']
#                 best_metric['psnr']['epoch'] = current_epoch
#                 save_all(current_epoch, model, optimizer, scheduler,
#                          hparams, best_metric, 'best_psnr')
#         save_all(current_epoch, model, optimizer, scheduler,
#                  hparams, best_metric, 'last')
#         print_epoch_result(train_return, valid_result, current_epoch)
#         print('best ssim: ', best_metric['ssim']['value'], '  best epoch: ', best_metric['ssim']['epoch'])
#         print('best psnr: ', best_metric['psnr']['value'], '  best epoch: ', best_metric['psnr']['epoch'])