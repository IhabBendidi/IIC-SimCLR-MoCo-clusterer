"""
TUNIT: Truly Unsupervised Image-to-Image Translation
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils
import os
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from sklearn import metrics

import matplotlib.pyplot as plt
import cv2
from tqdm import trange
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import shutil
import torchvision.transforms as transforms
from sklearn.cluster import KMeans

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from tools.utils import *
from neptune.new.types import File
#from pyclustering.cluster.xmeans import xmeans
#from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from tqdm import trange

def plot_tSNE(data_loader, networks, epoch, args, additional=None):
    # set nets
    C = networks['C'] if not args.distributed else networks['C'].module
    C_EMA = networks['C_EMA'] if not args.distributed else networks['C_EMA'].module
    # switch to train mode

    C.eval()
    C_EMA.eval()
    # data loader
    val_loader = data_loader['VAL']
    logger = additional['logger']
    run = additional['Neptune']

    if args.labels or args.sources :
        df = pd.read_csv(os.path.join(args.data_dir,args.data_type,'dataset_labels.csv'))
        labels_list = df['0'].tolist()




    def tensor2np(tensor, resize_to=None):
        '''
        Convert an image tensor to a numpy image array and resize

        Args:
            tensor (torch.Tensor): The input tensor that should be converted
            resize_to (tuple(int, int)): The desired output size of the array

        Returns:
            (np.ndarray): The input tensor converted to a channel last resized array
        '''

        out_array = tensor.detach().cpu().numpy()
        out_array = np.moveaxis(out_array, 0, 2) # (CHW) -> (HWC)

        if resize_to is not None:
            out_array = cv2.resize(out_array, dsize=resize_to, interpolation=cv2.INTER_CUBIC)

        return(out_array)

    with torch.no_grad():
        data = []
        targets = []

        val_iter = iter(val_loader)

        # entropy stuff
        cluster_grid = [[] for _ in range(args.output_k)]
        cluster_composition = [[] for _ in range(args.output_k)]
        cluster_features = [[] for _ in range(args.output_k)]
        plaque_composition = [[] for _ in range(args.output_k)]
        pit_composition = [[] for _ in range(args.output_k)]


        # kmeans stuff




        gt = [[] for _ in range(args.output_k)]
        gtlist = []
        phenotypes = []
        imgs = []
        plaques = []
        pits = []

        for i in tqdm(range(len(val_loader))):
            x, y = next(val_iter)
            x = x[0]
            if args.labels:
                phenotype = y[1]
            elif args.sources :
                phenotype = y[1]
                plate = y[2]
                pit = y[3]
            y = y[0]

            # x = x.view(1, *x.shape)
            x = x.cuda(args.gpu)
            outs = C_EMA(x)
            # outs = C(x)
            feat = outs['cont']
            logit = outs['disc']

            target = torch.argmax(logit, 1)

            for idx in range(len(feat.cpu().data.numpy())):
                data.append(feat.cpu().data.numpy()[idx])
                imgs.append(x.cpu()[idx])
                targets.append(int(target[idx].item()))
                gtlist.append(int(y[idx].item()))
                if args.labels:
                    phenotypes.append(phenotype[idx].item())
                    cluster_composition[int(target[idx].item())].append(phenotype[idx].item())
                elif args.sources :
                    phenotypes.append(phenotype[idx].item())
                    gt[int(target[idx].item())].append(y[idx].item())
                    plaques.append(plate[idx])
                    pits.append(pit[idx])
                    cluster_composition[int(target[idx].item())].append(phenotype[idx].item())
                    plaque_composition[int(target[idx].item())].append(plate[idx])
                    pit_composition[int(target[idx].item())].append(pit[idx])
                cluster_grid[int(target[idx].item())].append(x[idx].view(1, *x[idx].shape))
                cluster_features[int(target[idx].item())].append(feat.cpu().data.numpy()[idx])

        ret = TSNE(n_components=2, random_state=0,learning_rate=100.0).fit_transform(data)

        # Xmeans
        print("length of existing validation styles "+ str(len(data)))

        cluster_map = {}

        for i in range(args.output_k):
            numlist = [0 for _ in range(args.output_k)]
            for g in gt[i]:
                numlist[g] += 1
            cluster_map[i] = np.argmax(numlist)


        cluster_centroids = [np.mean(x, axis=0) for x in cluster_features]


        print(cluster_map)
        cluster_map_list = sorted(cluster_map, key=cluster_map.get)
        print(cluster_map_list)

    def show(data_iter, targets, t_sne_ret,phenotypes,epoch,logger):
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'violet', 'orange', 'purple','darkgoldenrod','darkkhaki','lightgreen','salmon','darkgrey','deepskyblue','magenta','silver','lawngreen','steelblue','orchid','crimson']
        colors = colors[:args.output_k]

        # entropy pseudo labels
        if args.graphs :
            plt.figure(figsize=(18, 18),dpi=60)
            print()
            for label in set(targets):
                idx = np.where(np.array(targets) == label)[0]
                plt.scatter(t_sne_ret[idx, 0], t_sne_ret[idx, 1], c=colors[label], label=label,s=15)
            plt.legend()
            plt.ylim([-90, 90])
            plt.xlim([-90, 90])
            plt.savefig([args.plot_dir + '/pseudo_{}_{}.png'][0].format(epoch,args.model_name))
            if args.nept :
                run["clustering/clusters/"+str(epoch)].log(File([args.plot_dir + '/pseudo_{}_{}.png'][0].format(epoch,args.model_name)))


            plt.close()
        num_phenotypes = len(set(phenotypes))

        # ground truth labels
        if args.labels or args.sources:
            if args.graphs :
                plt.figure(figsize=(18, 18),dpi=60)
                print()
                r = 0
                for label in set(phenotypes):
                    idx = np.where(np.array(phenotypes) == label)[0]
                    plt.scatter(t_sne_ret[idx, 0], t_sne_ret[idx, 1], c=colors[r], label=labels_list[label],s=15)
                    r += 1

                plt.legend()
                plt.ylim([-90, 90])
                plt.xlim([-90, 90])

                plt.savefig([args.plot_dir + '/gt_{}_{}.png'][0].format(epoch,args.model_name))
                if args.nept :
                    run["clustering/gt/"+str(epoch)].log(File([args.plot_dir + '/gt_{}_{}.png'][0].format(epoch,args.model_name)))
                plt.close()
            #computing AMI
            ami = metrics.adjusted_mutual_info_score(phenotypes, targets)
            cluster_accuracy = cluster_acc(np.array(phenotypes), np.array(targets))


            print("AMI score (entropy) for this epoch is : " + str(ami))
            print("Clustering accuracy score (ACC) for this epoch is : " + str(cluster_accuracy))
            logger.add_scalar('C/AMI', ami, epoch+1)
            logger.add_scalar('C/ACC', cluster_accuracy, epoch+1)
        if args.nept and (args.labels or args.sources):
            run["val/AMI"].log(ami)
            run["val/ACC"].log(cluster_accuracy)

        print("Length of test data is :" + str(len(t_sne_ret)))
        if args.labels or args.sources:
            best_AMI = additional['best_ami']
            best_clustering_epoch = additional['best_clustering_epoch']

            if ami > best_AMI :
                best_AMI = ami
                best_clustering_epoch = epoch
            if args.nept :
                run["val/best_AMI"].log(best_AMI)
                run["val/best_clustering_epoch"].log(best_clustering_epoch)
            best_kmeans_AMI = additional['best_kmeans_ami']
            best_kmeans_epoch = additional['best_kmeans_epoch']

            if cluster_accuracy > best_kmeans_AMI :
                best_kmeans_AMI = cluster_accuracy
                best_kmeans_epoch = epoch
            if args.nept :
                run["val/best_ACC"].log(best_kmeans_AMI)
                run["val/best_ACC_epoch"].log(best_kmeans_epoch)

            return [best_AMI,best_kmeans_AMI],[best_clustering_epoch,best_kmeans_epoch]
        else :
            return 0,0





    val_iter = iter(val_loader)

    best_AMI = show(val_iter, targets, ret,phenotypes,epoch,logger)
    return best_AMI
