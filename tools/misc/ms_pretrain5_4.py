"""  
same as ms_pretrain5_3.py
reduce the projector and predictor dimension
"""

import argparse
import os
import sys
import random
import shutil
import time
import traceback
import warnings
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.cuda.amp import autocast, GradScaler
import torchvision.models as models

import albumentations as albu
from albumentations.pytorch import ToTensorV2

from simsiam.loader import *
from simsiam.builder import *
from simsiam.resnet import *
from dataset.hubmap import *
from dataset.camelyon import *
from dataset.bcss import *
from dataset.paip import *
from utils.utils import increment_path
from utils.logger import setup_logger


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch SLF-WSI Pre-Training")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet18",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=8,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 8)",
)
parser.add_argument(
    "--epochs", default=100, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=512,
    type=int,
    metavar="N",
    help="mini-batch size (default: 512), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=3e-4,
    type=float,
    metavar="LR",
    help="initial (base) learning rate",
    dest="lr",
)
parser.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
parser.add_argument(
    "--dist-url",
    default="tcp://127.0.0.1:{}".format(port),
    type=str,
    help="initialization URL for pytorch distributed backend. See "
    "https://pytorch.org/docs/stable/distributed.html for details.",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)

# simsiam specific configs:
parser.add_argument(
    "--dim", default=2048, type=int, help="feature dimension (default: 2048)"
)
parser.add_argument(
    "--pred-dim",
    default=512,
    type=int,
    help="hidden dimension of the predictor (default: 512)",
)
parser.add_argument(
    "--fix-pred-lr", action="store_true", help="Fix learning rate for the predictor"
)

# Data settings
parser.add_argument("--data-name", type=str)
parser.add_argument("--data", metavar="DIR", help="path to dataset")
parser.add_argument("--mean", nargs=3, type=float, default=[0.485, 0.456, 0.406])
parser.add_argument("--std", nargs=3, type=float, default=[0.229, 0.224, 0.225])
parser.add_argument("-i", "--img-sz", type=int, default=224)

# Log setting
parser.add_argument("--log-dir", default="./logs/temp", type=str)
parser.add_argument("--wandb", action="store_true", help="use wandb as log tool.")
parser.add_argument("--run-tag", nargs="*", default=None, type=str)
parser.add_argument("--run-name", default=None, type=str)
parser.add_argument("--run-notes", default="PyTorch SLF-WSI training", type=str)

# slf-wsi specific configs:
parser.add_argument(
    "--save-freq",
    default=50,
    type=int,
    metavar="N",
    help="save frequency (default: 100)",
)
parser.add_argument("--amp", action="store_true")

#########################
## dcv2 specific params #
#########################
parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")
parser.add_argument(
    "--nmb_prototypes",
    default=[3000, 3000, 3000],
    type=int,
    nargs="+",
    help="number of prototypes - it can be multihead",
)
parser.add_argument(
    "--crops_for_assign",
    type=int,
    nargs="+",
    default=[0, 1],
    help="list of crops id used for computing assignments",
)


parser.add_argument("--use_clr", action="store_true")
parser.add_argument("--ms_lr", action="store_true")
parser.add_argument("--scale", type=int, default=4)


def main():
    args = parser.parse_args()

    args.log_dir = str(increment_path(args.log_dir, sep="_", mkdir=True))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    with open(os.path.join(args.log_dir, "configs.txt"), "w") as file:
        for arg in vars(args):
            file.write(str(arg) + ": " + str(getattr(args, arg)) + "\n")

    ngpus_per_node = torch.cuda.device_count()

    try:
        if args.multiprocessing_distributed:
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            args.world_size = ngpus_per_node * args.world_size
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # main_worker process function
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        else:
            # Simply call main_worker function
            main_worker(args.gpu, ngpus_per_node, args)
    except Exception as e:
        # logger.critical(e, exc_info=True)
        print(e, "\n")

        # print origin trace info
        with open(args.log_dir + "/error.txt", "a") as myfile:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info, file=myfile)
            myfile.write("\n")
            del exc_info


def main_worker(gpu, ngpus_per_node, arg):
    global args
    args = arg

    global logger
    logger = logging.getLogger()

    args.gpu = gpu

    # only log if master(rank0)
    if args.multiprocessing_distributed and args.gpu == 0:
        logger = setup_logger(args.log_dir)
        logger.info(" ".join([sys.executable, *sys.argv]))
        logger.info("=> initialise python logger successfully!")

        if args.wandb:
            import wandb

            wandb.init(
                project="SLF-WSI",
                notes=args.run_notes,
                tags=args.run_tag,
                name=args.run_name,
                job_type="train",
                dir=args.log_dir,
                config=args,
            )
            logger.info("=> initialise wandb logger successfully!")

    if args.gpu is not None:
        logger.info("=> use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        logger.info(f"=> use rank of {args.rank}")
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        torch.distributed.barrier()

    # create model
    logger.info("=> creating model '{}'".format(args.arch))
    model = MSModel5_4(
        models.__dict__[args.arch], args.scale, args.dim, args.pred_dim
    )  # TODO: hardcoded
    logger.info(model)

    # infer learning rate before changing batch size
    # init_lr = args.lr * args.batch_size / 256
    init_lr = args.lr

    if args.distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
        logger.info(f"=> use batch size of {args.batch_size}")
        logger.info(f"=> use workers of {args.workers}")
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    # logger.info(model) # print model after SyncBatchNorm

    if args.fix_pred_lr:
        optim_params = [
            {"params": model.module.encoder.parameters(), "fix_lr": False},
            {"params": model.module.predictor.parameters(), "fix_lr": True},
        ]
    else:
        optim_params = model.parameters()

    if args.ms_lr:
        context_params = [
            i[1]
            for i in filter(
                lambda kv: kv[0].startswith("context_"), model.module.named_parameters()
            )
        ]
        other_params = [
            i[1]
            for i in filter(
                lambda kv: not kv[0].startswith("context_"),
                model.module.named_parameters(),
            )
        ]
        optim_params = [
            {"params": other_params},
            {"params": context_params, "lr": init_lr / 10},  # TODO: hardcoded
        ]
    else:
        optim_params = model.parameters()

    # optimizer = torch.optim.SGD(optim_params, init_lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(optim_params, lr=init_lr)
    logger.info("=> use custom Adam optimiser!")

    if args.amp:
        logger.info(f"=> enable automatic mix precision training!")
    scaler = GradScaler(enabled=args.amp)

    cudnn.benchmark = True

    # Data loading code
    context_aug = [
        albu.RandomResizedCrop(224, 224, scale=(0.5, 1.0), p=1),
        albu.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
        albu.ToGray(p=0.2),
        albu.OneOf(
            [albu.GaussianBlur(sigma_limit=[0.1, 2.0], p=0.5), albu.Sharpen(p=0.5)],
            p=0.5,
        ),
        albu.HorizontalFlip(p=0.5),
        albu.Normalize(
            mean=args.mean,
            std=args.std,
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(),
    ]
    # TODO: may need finetune
    target_aug = [
        albu.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
        albu.ToGray(p=0.2),
        albu.OneOf(
            [albu.GaussianBlur(sigma_limit=[0.1, 2.0], p=0.5), albu.Sharpen(p=0.5)],
            p=0.5,
        ),
    ]
    misc_aug = [
        # albu.RandomCrop(224, 224, p=1),
        albu.RandomResizedCrop(224, 224, scale=(0.5, 1.0), p=1),
        albu.HorizontalFlip(p=0.5),
        albu.Normalize(
            mean=args.mean,
            std=args.std,
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(),
    ]

    logger.info(f"=> Context augmentation pipeline: {context_aug}")
    logger.info(f"=> Target augmentation pipeline: {target_aug}")

    if args.data_name == "hubmap":
        train_dataset = HubmapDataset3_albu(
            args.data,
            (
                albu.Compose(context_aug),
                albu.Compose(target_aug),
                albu.Compose(misc_aug),
            ),
            all_pos=args.all_pos,
        )

    elif args.data_name == "bcss":
        train_dataset = BcssPretrainDatasetMS5_2(
            args.data,
            (
                albu.Compose(context_aug),
                albu.Compose(target_aug),
                albu.Compose(misc_aug),
            ),
            return_index=False,
        )

    elif args.data_name == "paip":
        train_dataset = PaipPretrainDatasetMS5_2(
            args.data,
            (
                albu.Compose(context_aug),
                albu.Compose(target_aug),
                albu.Compose(misc_aug),
            ),
            return_index=False,
        )

    else:
        logger.error("Unsupported dataset!")
        sys.exit(1)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    logger.info(
        f"=> Size of data: {len(train_dataset)}, size of epochs: {len(train_loader)}"
    )
    best_loss = 255

    for epoch in range(args.epochs):
        start = time.time()

        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, init_lr, epoch, args) #TODO: to change

        # train for one epoch
        loss = train(
            train_loader,
            model,
            optimizer,
            epoch,
            scaler,
        )

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):

            is_best = loss <= best_loss
            best_loss = min(loss, best_loss)

            # Wandb log
            if args.wandb:
                wandb.log({"train_loss": loss})
                wandb.run.summary["train_loss"] = best_loss
                logger.info(f"=> Wandb summary saved!")

            if (epoch + 1) % args.save_freq == 0:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": args.arch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                    },
                    is_best=False,
                    filename="{}/checkpoint_{:04d}.pth.tar".format(args.log_dir, epoch),
                )
                logger.info(f"=> Model saved at epoch {epoch}!")

            elapsed_time = (time.time() - start) / 60
            logger.info(
                f"======= TIME: {elapsed_time:.2f} mins, BEST LOSS: {loss:.4f}/{best_loss:.4f} ======="
            )

    # Wandb log
    if (
        args.multiprocessing_distributed
        and args.rank % ngpus_per_node == 0
        and args.wandb
    ):
        shutil.copyfile(
            os.path.join(args.log_dir, "log.txt"),
            os.path.join(wandb.run.dir, "train_output.log"),
        )
        logger.info(f"=> Log is copied into Wandb folder!")
        wandb.finish()


def train(train_loader, model, optimizer, epoch, scaler):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch),
    )
    logger = logging.getLogger("SLF-WSI")
    logger.info(f"=> begin epoch {epoch}")

    # switch to train mode
    model.train()
    contrast_loss = nn.CosineSimilarity(dim=1).cuda(args.gpu)
    if args.use_clr:
        cls_loss = nn.CrossEntropyLoss().cuda(args.gpu)

    end = time.time()
    for it, (context_img, target_img, jigsaw_idx) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        bs = context_img[0].size(0)

        if args.gpu is not None:
            context_img[0] = context_img[0].cuda(args.gpu, non_blocking=True)
            context_img[1] = context_img[1].cuda(args.gpu, non_blocking=True)
            target_img[0] = (
                target_img[0].flatten(0, 1).cuda(args.gpu, non_blocking=True)
            )
            target_img[1] = (
                target_img[1].flatten(0, 1).cuda(args.gpu, non_blocking=True)
            )

        # compute output and loss
        with autocast(enabled=args.amp):
            outputs = model(
                (context_img[0], target_img[0]),
                (context_img[1], target_img[1]),
                jigsaw_idx,
            )
            context_loss = (
                -(
                    contrast_loss(outputs[0][0], outputs[0][3]).mean()
                    + contrast_loss(outputs[0][1], outputs[0][2]).mean()
                )
                * 0.5
            )
            target_loss = (
                -(
                    contrast_loss(outputs[1][0], outputs[1][3]).mean()
                    + contrast_loss(outputs[1][1], outputs[1][2]).mean()
                )
                * 0.5
            )
            fuser_loss = (
                -(
                    contrast_loss(outputs[2][0], outputs[2][3]).mean()
                    + contrast_loss(outputs[2][1], outputs[2][2]).mean()
                )
                * 0.5
            )
            loss = context_loss + target_loss + fuser_loss

        losses.update(loss.item(), bs)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if it % args.print_freq == 0:
            logger.info(progress.display(it))

    return losses.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return str("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


if __name__ == "__main__":
    main()
