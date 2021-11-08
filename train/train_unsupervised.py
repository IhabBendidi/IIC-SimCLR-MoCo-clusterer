"""
TUNIT: Truly Unsupervised Image-to-Image Translation
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
"""
R-TUNIT: Really Truly Unsupervised Image-to-Image Translation
Ihab Bendidi - Ecole Normale Superieure.
Apache license 2.0
"""
from tqdm import trange
from torch.nn import functional as F
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tools.utils import *
from clustering.gap import get_gap_statistics
from tools.ops import compute_grad_gp, update_average, copy_norm_params, calc_iic_loss, \
    queue_data, dequeue_data, average_gradients, calc_adv_loss, calc_contrastive_loss, calc_recon_loss
import torchvision.utils as vutils
import os
import numpy as np
import pandas as pd
from neptune.new.types import File
#from pyclustering.cluster.xmeans import xmeans
#from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import cv2
from tqdm import trange
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score,precision_score,recall_score
from clustering.utils import closest_to_centroid

######################
# Fully unsupervised #
######################
def trainGAN_UNSUP(data_loader, networks, opts, epoch, args, additional):
    # avg meter

    c_losses = AverageMeter()
    moco_losses = AverageMeter()
    iic_losses = AverageMeter()

    moco_losses_original = AverageMeter()
    iic_losses_original = AverageMeter()

    moco_losses_new = AverageMeter()
    iic_losses_new = AverageMeter()

    heads = ["clustering","overclustering"]


    C = networks['C'] if not args.distributed else networks['C'].module
    C_EMA = networks['C_EMA'] if not args.distributed else networks['C_EMA'].module
    # set opts

    c_opt = opts['C']
    # switch to train mode
    C.train()
    C_EMA.train()

    logger = additional['logger']
    queue = additional['queue']
    run = additional['Neptune']



    t_train = trange(0, args.iters, initial=0, total=args.iters)
    styles = []
    raw_images = []
    original_pseudo_labels = []
    transformed_pseudo_labels = []
    for head in heads :
        train_it = iter(data_loader)
        for i in t_train:
            try:
                imgs, _ = next(train_it)
            except:
                train_it = iter(data_loader)
                imgs, _ = next(train_it)



            x_org = imgs[0]
            x_tf = imgs[1]
            x_tf_2 = imgs[2]




            x_ref_idx = torch.randperm(x_org.size(0))

            x_org = x_org.cuda(args.gpu)
            x_tf = x_tf.cuda(args.gpu)
            x_tf_2 = x_tf_2.cuda(args.gpu)
            x_ref_idx = x_ref_idx.cuda(args.gpu)

            x_ref = x_org.clone()
            x_ref = x_ref[x_ref_idx]

            #################
            # BEGIN Train C #
            #################
            training_mode = 'ONLYCLS'
            q_cont = C.moco(x_org)
            k_cont = C_EMA.moco(x_tf)
            k_cont = k_cont.detach()

            k2_cont = C_EMA.moco(x_tf_2)
            k2_cont = k2_cont.detach()

            if head == "clustering" :

                q_disc = C.iic(x_org)
                k_disc = C.iic(x_tf)
                k2_disc = C.iic(x_tf_2)

                q_disc = F.softmax(q_disc, 1)
                k_disc = F.softmax(k_disc, 1)
                k2_disc = F.softmax(k2_disc, 1)



            elif head == "overclustering" :
                q_disc = C.over_iic(x_org)
                k_disc = C.over_iic(x_tf)
                k2_disc = C.over_iic(x_tf_2)


                q_disc = F.softmax(q_disc, 1)
                k_disc = F.softmax(k_disc, 1)
                k2_disc = F.softmax(k2_disc, 1)




            original = q_disc.detach().cpu().tolist()
            transformed = k_disc.detach().cpu().tolist()
            transformed_2 = k2_disc.detach().cpu().tolist()

            original = [x.index(max(x)) for x in original]
            transformed = [x.index(max(x)) for x in transformed]
            transformed_2 = [x.index(max(x)) for x in transformed_2]


            iic_loss = calc_iic_loss(q_disc, k_disc)


            iic_loss_2 = calc_iic_loss(q_disc, k2_disc)
            original_iic = iic_loss.detach().clone()
            #iic_loss *= 1
            iic_loss += iic_loss_2
            moco_loss = calc_contrastive_loss(args, q_cont, k_cont, queue)
            moco_loss_2 = calc_contrastive_loss(args, q_cont, k2_cont, queue)
            original_moco = moco_loss.detach().clone()
            moco_loss += moco_loss_2

            c_loss = moco_loss + 5.0 * iic_loss
            q_f = q_cont.detach().cpu().tolist()
            for x in range(len(q_f)) :
                original_pseudo_labels.append(original[x])
                transformed_pseudo_labels.append(transformed[x])
                if len(styles)<args.max_data:
                    styles.append(q_f[x])
                    raw_images.append(x_org[x])
            if epoch >= args.separated:
                c_loss = 0.1 * c_loss

            c_opt.zero_grad()
            #rkd_loss = c_loss.clone()
            c_loss.clone().backward()
            #c_loss = rkd_loss
            if args.distributed:
                average_gradients(C)
            c_opt.step()
            ###############
            # END Train C #
            ###############



            queue = queue_data(queue, k_cont)
            queue = dequeue_data(queue)

            update_average(C_EMA, C)

            torch.cuda.synchronize()

            with torch.no_grad():

                c_losses.update(c_loss.item(), x_org.size(0))
                moco_losses.update(moco_loss.item(), x_org.size(0))
                iic_losses.update(iic_loss.item(), x_org.size(0))

                moco_losses_original.update(original_moco.item(), x_org.size(0))
                iic_losses_original.update(original_iic.item(), x_org.size(0))

                moco_losses_new.update(moco_loss_2.item(), x_org.size(0))
                iic_losses_new.update(iic_loss_2.item(), x_org.size(0))

                if (i + 1) % args.log_step == 0 and (args.gpu == 0 or args.gpu == '0'):
                    summary_step = epoch * args.iters + i

                    if args.nept :
                        run["train/E/E_Loss"].log(c_losses.avg)
                        run["train/E/IID"].log(iic_losses.avg)
                        run["train/E/MOCO"].log(moco_losses.avg)

                        run["train/E/IID_original"].log(iic_losses_original.avg)
                        run["train/E/MOCO_original"].log(moco_losses_original.avg)

                        run["train/E/IID_new"].log(iic_losses_new.avg)
                        run["train/E/MOCO_new"].log(moco_losses_new.avg)
                    add_logs(args, logger, 'C/LOSS', c_losses.avg, summary_step)
                    add_logs(args, logger, 'C/IID', iic_losses.avg, summary_step)
                    add_logs(args, logger, 'C/MOCO', moco_losses.avg, summary_step)

                    add_logs(args, logger, 'C/IID_original', iic_losses_original.avg, summary_step)
                    add_logs(args, logger, 'C/MOCO_original', moco_losses_original.avg, summary_step)

                    add_logs(args, logger, 'C/IID_new', iic_losses_new.avg, summary_step)
                    add_logs(args, logger, 'C/MOCO_new', moco_losses_new.avg, summary_step)

                    print('Epoch: [{}/{}] [{}/{}] MODE[{}] Avg Loss:  C[{c_losses.avg:.2f}]'
                          .format(epoch + 1, args.epochs, i+1, args.iters, training_mode,
                                 c_losses=c_losses))

    # Computing f1 score for our transformations vs originals

    f1 = f1_score(original_pseudo_labels,transformed_pseudo_labels, average='micro')
    precision = precision_score(original_pseudo_labels,transformed_pseudo_labels, average='micro')
    recall = recall_score(original_pseudo_labels,transformed_pseudo_labels, average='micro')
    print("F1 score for this epoch is : " + str(f1))
    if args.nept :
        run['train/f1_score'].log(f1)
        run['train/precision'].log(precision)
        run['train/recall'].log(recall)
    print("length of existing training styles "+ str(len(styles)))





    copy_norm_params(C_EMA, C)
