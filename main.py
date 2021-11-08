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
import argparse
import warnings
from datetime import datetime
from glob import glob
from shutil import copyfile
from collections import OrderedDict

import torch.nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from models.generator import Generator as Generator
from models.discriminator import Discriminator as Discriminator
from models.guidingNet import GuidingNet
from models.inception import InceptionV3

from train.train_unsupervised import trainGAN_UNSUP

from validation.validation import validateUN, calcFIDBatch
from validation.plot_tsne import plot_tSNE

from tools.utils import *
from datasets.datasetgetter import get_dataset
from tools.ops import initialize_queue




from tensorboardX import SummaryWriter

# Configuration
parser = argparse.ArgumentParser(description='PyTorch GAN Training')

parser.add_argument('--data_path', type=str, default='./data',
                    help='Dataset directory. Please refer Dataset in README.md')

parser.add_argument('--data_type', type=str, default='BBBC021_128',
                    help='Dataset directory. Please refer Dataset in README.md')

parser.add_argument('--data_folder', type=str, default='N2A',
                    help='exact Dataset folder. Please refer Dataset in README.md')

parser.add_argument('--augmentation', default='original', help='augmentation to use',
                    choices=['mnist','original','improved_v0','improved','improved_v2','improved_v2.2','improved_v3','improved_v3.1','improved_v4','improved_v4.1','improved_v5'])
parser.add_argument('--workers', default=4, type=int, help='the number of workers of data loader')

parser.add_argument('--model_name', type=str, default='IIC',
                    help='Prefix of logs and results folders. '
                         'ex) --model_name=ABC generates ABC_20191230-131145 in logs and results')

parser.add_argument('--epochs', default=1000, type=int, help='Total number of epochs to run. Not actual epoch.')
parser.add_argument('--iters', default=1000, type=int, help='Total number of iterations per epoch')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--val_batch', default=10, type=int,
                    help='Batch size for validation. '
                         'The result images are stored in the form of (val_batch, val_batch) grid.')
parser.add_argument('--log_step', default=100, type=int)

parser.add_argument('--sty_dim', default=128, type=int, help='The size of style vector')
parser.add_argument('--output_k', default=10, type=int, help='Total number of classes to use')
parser.add_argument('--img_size', default=70, type=int, help='Input image size')


parser.add_argument('--neptune', dest='nept', action='store_true',
                    help='Call for neptune  mode')
parser.add_argument('--labels', dest='labels', action='store_true',
                    help='Call for labels  mode')
parser.add_argument('--sources', dest='sources', action='store_true',
                    help='Call for cell sources  mode, shouldnt be done at the same time as label mode')
parser.add_argument('--debug', dest='debug', action='store_true',
                    help='Call for debug  mode')

parser.add_argument('--load_model', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: None)'
                         'ex) --load_model GAN_20190101_101010'
                         'It loads the latest .ckpt file specified in checkpoint.txt in GAN_20190101_101010')
parser.add_argument('--validation', dest='validation', action='store_true',
                    help='Call for valiation only mode')

parser.add_argument('--cexploration', dest='cexploration', action='store_true',
                    help='Call for cluster exploration mode')

parser.add_argument('--offline', dest='offline', action='store_true',
                    help='Call for neptune offline only mode')

parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--gpu', default='0', type=str,
                    help='GPU id to use.')
parser.add_argument('--ddp', dest='ddp', action='store_true', help='Call if using DDP')
parser.add_argument('--port', default='8989', type=str)

parser.add_argument('--iid_mode', default='iid+', type=str, choices=['iid', 'iid+'])

parser.add_argument('--w_gp', default=0.01, type=float, help='Coefficient of GP of D')
parser.add_argument('--w_rec', default=0.1, type=float, help='Coefficient of Rec. loss of G')
parser.add_argument('--w_adv', default=0.1, type=float, help='Coefficient of Adv. loss of G')
parser.add_argument('--w_vec', default=1.0, type=float, help='Coefficient of Style vector rec. loss of G')
parser.add_argument('--w_cen', default=1.0, type=float, help='Coefficient of Center crop coefficient in iic')
# --w_gp 0.01 --w_rec 0.01 --w_adv 0.1 --w_vec 1
if __name__ == '__main__':
    args = parser.parse_args()
    if args.nept:
        import neptune.new as neptune
        print("You are using Neptune and Tensorboard for logging metrics")
        f = open("neptune_token.txt", "r")
        token = f.readline().split('\n')[0]
        f.close()
        if args.offline :
            CONNECTION_MODE = "offline"
            run = neptune.init(project='ihab/rtunit',# your project
                           api_token=token, # your api token
                           mode=CONNECTION_MODE,
                           )
        else :
            run = neptune.init(project='ihab/rtunit',# your project
                       api_token=token, # your api token
                       )
    else :
        print("You are using only Tensorboard for logging metrics")
        run = None





def main():
    ####################
    # Default settings #
    ####################
    args = parser.parse_args()


    print("PYTORCH VERSION", torch.__version__)
    args.data_dir = args.data_path
    args.start_epoch = 0

    args.start_cexploration = 50

    if args.data_type != 'BBBC021_128' and args.data_type != 'BBBC021_196' :
        args.w_gp = 10.0
        args.w_rec = 0.1
        args.w_adv = 1.0
        args.w_vec = 0.01



    # Some checks to see if args are good
    args.graphs = True
    if args.output_k > 18 :
        args.graphs = False
    if args.labels and args.sources :
        args.labels = False





    args.train_mode = 'GAN_UNSUP'

    if args.debug :
        args.iters = 10



    # unsup_start : train networks with supervised data only before unsup_start
    args.unsup_start = 0

    if args.debug :
        args.unsup_start = 0

    args.unsup_start = args.unsup_start

    # Cuda Set-up
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.multiprocessing_distributed = False

    if len(args.gpu) > 1:
        args.multiprocessing_distributed = True
    print(args.multiprocessing_distributed)
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    print(args.distributed)

    ngpus_per_node = torch.cuda.device_count()
    args.ngpus_per_node = ngpus_per_node

    print("MULTIPROCESSING DISTRIBUTED : ", args.multiprocessing_distributed)

    # Logs / Results
    if args.load_model is None:
        if args.nept :
            run["Experiment"] = args.model_name
            run["Run-Moment"] = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.model_name = '{}_{}_'.format(args.model_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        args.model_name = args.load_model + "_VAL"
        if args.nept :
            run["Experiment"] = args.model_name
            run["Run-Moment"] = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.log_dir_or = os.path.join('./logs', args.load_model)
        args.event_dir_or = os.path.join(args.log_dir_or, 'events')

    makedirs('./logs')
    makedirs('./results')








    args.log_dir = os.path.join('./logs', args.model_name)
    args.event_dir = os.path.join(args.log_dir, 'events')
    args.res_dir = os.path.join('./results', args.model_name)
    args.plot_dir = os.path.join(args.res_dir, 'clusters')
    args.npy_dir = os.path.join(args.res_dir, 'data')
    args.imgs_folder = os.path.join(args.npy_dir, 'imgs')
    args.viz_dir = os.path.join(args.res_dir, 'viz')


    if args.debug :
        makedirs('./debug')
        makedirs('./debug/results')
        makedirs('./debug/logs')
        args.log_dir = os.path.join('./debug/logs', args.model_name)
        args.event_dir = os.path.join(args.log_dir, 'events')
        args.res_dir = os.path.join('./debug/results', args.model_name)
        args.plot_dir = os.path.join(args.res_dir, 'clusters')
        args.npy_dir = os.path.join(args.res_dir, 'data')
        args.imgs_folder = os.path.join(args.npy_dir, 'imgs')
        args.viz_dir = os.path.join(args.res_dir, 'viz')

    makedirs(args.log_dir)
    makedirs(args.plot_dir)
    makedirs(args.npy_dir)
    makedirs(args.viz_dir)
    makedirs(args.imgs_folder)

    dirs_to_make = next(os.walk('./'))[1]
    not_dirs = ['.idea','.neptune', 'jz_logs','.git', 'logs', 'results','debug', '.gitignore', '.nsmlignore', 'resrc','script.slurm','neptune_token.txt']
    makedirs(os.path.join(args.log_dir, 'codes'))
    for to_make in dirs_to_make:
        if to_make in not_dirs:
            continue
        makedirs(os.path.join(args.log_dir, 'codes', to_make))
    makedirs(args.res_dir)

    if args.load_model is None:
        pyfiles = glob("./*.py")
        for py in pyfiles:
            copyfile(py, os.path.join(args.log_dir, 'codes') + "/" + py)

        for to_make in dirs_to_make:
            if to_make in not_dirs:
                continue
            tmp_files = glob(os.path.join('./', to_make, "*.py"))
            for py in tmp_files:
                copyfile(py, os.path.join(args.log_dir, 'codes', py[2:]))
    # Logging

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args,run))
    else:
        main_worker(args.gpu, ngpus_per_node, args,run)


def main_worker(gpu, ngpus_per_node, args,run):
    if len(args.gpu) == 1:
        args.gpu = 0
    else:
        args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:'+args.port,
                                world_size=args.world_size, rank=args.rank)



    # # of GT-classes
    args.num_cls = args.output_k


    args.att_to_use = [0,]


    # IIC statistics
    args.epoch_acc = []
    args.epoch_avg_subhead_acc = []
    args.epoch_stats = []



    # Logging
    logger = SummaryWriter(args.event_dir + "_" + str(args.gpu))

    # build model - return dict
    networks, opts = build_model(args)

    # load model if args.load_model is specified
    load_model(args, networks, opts)
    cudnn.benchmark = True

    # get dataset and data loader
    train_dataset, val_dataset = get_dataset(args)
    train_loader, val_loader, train_sampler = get_loader(args, {'train': train_dataset, 'val': val_dataset})
    args.iters = (args.min_data // args.batch_size) + 1
    if args.debug :
        args.iters = 10

    # map the functions to execute - un / sup / semi-
    trainFunc, validationFunc = map_exec_func(args)

    queue_loader =  train_loader

    queue = initialize_queue(networks['C_EMA'], args.gpu, queue_loader, feat_size=args.sty_dim)

    # print all the argument
    print_args(args)
    best_AMI = 0
    best_clustering_epoch = 0

    best_kmeans_AMI = 0
    best_kmeans_epoch = 0

    # All the test is done in the training - do not need to call
    if args.validation:
        validationFunc(val_loader, networks, 999, args, {'logger': logger, 'queue': queue,'Neptune':run})
        best_AMI,best_clustering_epoch = plot_tSNE(val_loader, networks,999, args,{'logger': logger, 'queue': queue,'Neptune':run,'best_ami':best_AMI,"best_clustering_epoch":best_clustering_epoch,'best_kmeans_ami':best_kmeans_AMI,"best_kmeans_epoch":best_kmeans_epoch})
        return

    # For saving the model
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        record_txt = open(os.path.join(args.log_dir, "record.txt"), "a+")
        for arg in vars(args):
            record_txt.write('{:35}{:20}\n'.format(arg, str(getattr(args, arg))))
        record_txt.close()

    # Run
    validationFunc(val_loader, networks, 0, args, {'logger': logger, 'queue': queue,'Neptune':run})
    #plot_tSNE(val_loader, networks,0, args,{'logger': logger, 'queue': queue})
    if args.nept :
        run["parameters"] =  vars(args)

    fid_best_ema = 200.0

    for epoch in range(args.start_epoch, args.epochs):
        print("START EPOCH[{}]".format(epoch+1))
        if args.nept :
            run["epoch"].log(epoch)
        if (epoch + 1) % (args.epochs // 10) == 0:
            save_model(args, epoch, networks, opts)

        if args.distributed:
            train_sampler.set_epoch(epoch)

        trainFunc(train_loader, networks, opts, epoch, args, {'logger': logger, 'queue': queue,'Neptune':run})

        validationFunc(val_loader, networks, epoch, args, {'logger': logger, 'queue': queue,'Neptune':run})
        best_AMI,best_clustering_epoch = plot_tSNE(val_loader, networks,epoch, args,{'logger': logger, 'queue': queue,'Neptune':run,'best_ami':best_AMI,"best_clustering_epoch":best_clustering_epoch,'best_kmeans_ami':best_kmeans_AMI,"best_kmeans_epoch":best_kmeans_epoch})
        if args.labels :
            best_kmeans_AMI = best_AMI[1]
            best_kmeans_epoch = best_clustering_epoch[1]
            best_AMI = best_AMI[0]
            best_clustering_epoch = best_clustering_epoch[0]



        # Write logs
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0):
            if (epoch + 1) % 10 == 0:
                save_model(args, epoch, networks, opts)
            if len(args.epoch_acc) > 0:
                add_logs(args, logger, 'STATC/Acc', float(args.epoch_acc[-1]), epoch + 1)


#################
# Sub functions #
#################
def print_args(args):
    for arg in vars(args):
        print('{:35}{:20}\n'.format(arg, str(getattr(args, arg))))


def build_model(args):
    args.to_train = 'CDGI'

    networks = {}
    opts = {}
    is_semi = False
    if 'C' in args.to_train:
        networks['C'] = GuidingNet(args.img_size, {'cont': args.sty_dim, 'disc': args.output_k})
        networks['C_EMA'] = GuidingNet(args.img_size, {'cont': args.sty_dim, 'disc': args.output_k})


    if args.distributed:
        if args.gpu is not None:
            print('Distributed to', args.gpu)
            torch.cuda.set_device(args.gpu)
            args.batch_size = int(args.batch_size / args.ngpus_per_node)
            args.workers = int(args.workers / args.ngpus_per_node)
            for name, net in networks.items():
                net_tmp = net.cuda(args.gpu)
                networks[name] = torch.nn.parallel.DistributedDataParallel(net_tmp, device_ids=[args.gpu], output_device=args.gpu)
        else:
            for name, net in networks.items():
                net_tmp = net.cuda()
                networks[name] = torch.nn.parallel.DistributedDataParallel(net_tmp)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        for name, net in networks.items():
            networks[name] = net.cuda(args.gpu)
    else:
        for name, net in networks.items():
            networks[name] = torch.nn.DataParallel(net).cuda()

    if 'C' in args.to_train:
        opts['C'] = torch.optim.Adam(
            networks['C'].module.parameters() if args.distributed else networks['C'].parameters(),
            1e-4, weight_decay=0.001)
        if args.distributed:
            networks['C_EMA'].module.load_state_dict(networks['C'].module.state_dict())
        else:
            networks['C_EMA'].load_state_dict(networks['C'].state_dict())


    return networks, opts


def load_model(args, networks, opts):
    if args.load_model is not None:
        check_load = open(os.path.join(args.log_dir_or, "checkpoint.txt"), 'r')
        if args.validation :
            to_restore = "model_4568.ckpt"
        else :
            to_restore = check_load.readlines()[-1].strip()
        load_file = os.path.join(args.log_dir_or, to_restore)
        print(load_file)
        if os.path.isfile(load_file):
            print("=> loading checkpoint '{}'".format(load_file))
            checkpoint = torch.load(load_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            if not args.multiprocessing_distributed:
                for name, net in networks.items():
                    tmp_keys = next(iter(checkpoint[name + '_state_dict'].keys()))
                    if 'module' in tmp_keys:
                        tmp_new_dict = OrderedDict()
                        for key, val in checkpoint[name + '_state_dict'].items():
                            tmp_new_dict[key[7:]] = val
                        net.load_state_dict(tmp_new_dict)
                        networks[name] = net
                    else:
                        net.load_state_dict(checkpoint[name + '_state_dict'])
                        networks[name] = net

            for name, opt in opts.items():
                opt.load_state_dict(checkpoint[name.lower() + '_optimizer'])
                opts[name] = opt
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(load_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.log_dir_or))


def get_loader(args, dataset):
    train_dataset = dataset['train']
    val_dataset = dataset['val']['VAL']

    print(len(val_dataset))


    train_dataset_ = train_dataset['TRAIN']
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset_)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset_, batch_size=args.batch_size,
                                                   shuffle=(train_sampler is None), num_workers=args.workers,
                                                   pin_memory=True, sampler=train_sampler, drop_last=False)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch, shuffle=True,
                                             num_workers=0, pin_memory=True, drop_last=False)

    val_loader = {'VAL': val_loader,'VALSET':  dataset['val']['FULL'],'TRAINSET': train_dataset['FULL']}
    val_loader['IDX'] = train_dataset['IDX']

    return train_loader, val_loader, train_sampler


def map_exec_func(args):
    trainFunc = trainGAN_UNSUP
    validationFunc = validateUN
    return trainFunc, validationFunc


def save_model(args, epoch, networks, opts):
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0):
        check_list = open(os.path.join(args.log_dir, "checkpoint.txt"), "a+")
        # if (epoch + 1) % (args.epochs//10) == 0:
        with torch.no_grad():
            save_dict = {}
            save_dict['epoch'] = epoch + 1
            for name, net in networks.items():
                save_dict[name+'_state_dict'] = net.state_dict()
                if name in ['C_EMA']:
                    continue
                save_dict[name.lower()+'_optimizer'] = opts[name].state_dict()
            print("SAVE CHECKPOINT[{}] DONE".format(epoch+1))
            save_checkpoint(save_dict, check_list, args.log_dir, epoch + 1)
        check_list.close()


if __name__ == '__main__':
    main()
