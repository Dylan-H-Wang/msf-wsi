"""  
SSL pre-training script for MSF-WSI
"""

import os
import sys
import math
import time
import random
import shutil
import logging
import argparse
import traceback
from pprint import pformat

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

import albumentations as albu
from albumentations.pytorch import ToTensorV2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.split(SCRIPT_DIR)[0]
sys.path.insert(0, ROOT_PATH)

from src.models import resnet
from src.models.backbone import MSFWSI
from src.utils.data.bcss import BcssPretrainDataset
from src.utils.data.paip import PaipPretrainDataset
from src.utils.data.camelyon import Camelyon16PretrainDataset
from src.utils.utils import increment_path
from src.utils.logger import setup_logger


def main(args):
    args.log_dir = str(increment_path(args.log_dir, sep="_", mkdir=True))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

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

    cudnn.benchmark = True
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("=> use tf32 for training!")
    if args.amp:
        logger.info(f"=> enable automatic mix precision training!")
        args.amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
        logger.info(f"=> use {args.amp_dtype} for training!")
    scaler = GradScaler(enabled=args.amp)
    args.gpu = gpu
    args.rank = gpu

    # only log if master(rank0)
    if args.rank == 0:
        logger = setup_logger(args.log_dir, name=args.logger_name)
        logger.info(" ".join([sys.executable, *sys.argv]))
        logger.info("=> initialise python logger successfully!")

        if args.tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            tb_path = str(
                increment_path(f"{args.log_dir}/tb_log/exp", sep="_", mkdir=True)
            )
            tb_writer = SummaryWriter(tb_path)
            logger.info("Initialise tensorboard logger successfully!")

        if args.wandb:
            import wandb

            wandb.init(
                project="MSF-WSI Experiments",
                notes=args.run_notes,
                tags=args.run_tag,
                group=args.run_group,
                name=args.run_name,
                job_type="pretrain",
                dir=args.log_dir,
                config=args,
            )
            logger.info("=> initialise wandb logger successfully!")

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
    model = MSFWSI(
        resnet.__dict__[args.arch],
        args.scale,
        args.dim,
        args.pred_dim,
        args.mask_ratio / 100,
        args.use_ac,
    )

    # infer learning rate before changing batch size
    init_lr = args.lr * math.sqrt(args.batch_size) / math.sqrt(32)
    # init_lr = args.lr
    logger.info(f"=> use init_lr of {init_lr:.4f}")

    # Apply SyncBN
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)

    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    logger.info(f"=> use batch size of {args.batch_size}")
    logger.info(f"=> use workers of {args.workers}")

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    if args.rank == 0:
        logger.info(model)  # print model after SyncBatchNorm

    # Data loading code
    context_aug = [
        albu.RandomResizedCrop(224, 224, scale=(0.5, 1.0), p=1),
        albu.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
        albu.ToGray(p=0.2),
        albu.OneOf(
            [
                albu.GaussianBlur(blur_limit=[19, 23], sigma_limit=[0.1, 2.0], p=0.5),
                albu.Sharpen(p=0.5),
            ],
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

    target_aug = [
        albu.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
        albu.ToGray(p=0.2),
        albu.OneOf(
            [
                albu.GaussianBlur(blur_limit=[19, 23], sigma_limit=[0.1, 2.0], p=0.5),
                albu.Sharpen(p=0.5),
            ],
            p=0.5,
        ),
    ]
    misc_aug = [
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

    logger.info(f"=> Context augmentation pipeline: {pformat(context_aug, indent=4)}")
    logger.info(f"=> Target augmentation pipeline: {pformat(target_aug, indent=4)}")

    if args.data_name == "bcss":
        train_dataset = BcssPretrainDataset(
            args.data,
            (
                albu.Compose(context_aug),
                albu.Compose(target_aug),
                albu.Compose(misc_aug),
            ),
            return_index=False,
            fold=args.fold,
        )
    elif args.data_name == "paip":
        train_dataset = PaipPretrainDataset(
            args.data,
            (
                albu.Compose(context_aug),
                albu.Compose(target_aug),
                albu.Compose(misc_aug),
            ),
            return_index=False,
            fold=args.fold,
        )
    elif args.data_name == "camelyon16":
        train_dataset = Camelyon16PretrainDataset(
            data_path=args.data,
            transforms=(
                albu.Compose(context_aug),
                albu.Compose(target_aug),
                albu.Compose(misc_aug),
            ),
            mode="train",
        )
    else:
        logger.error("Unsupported dataset!")
        sys.exit(1)

    if args.quick_test:
        # train_dataset.data_df = train_dataset.data_df[:7680]
        train_dataset.filename_imgs = train_dataset.filename_imgs[:7680]

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

    context_params = [
        i[1]
        for i in filter(
            lambda kv: kv[0].startswith("context_"), model.module.named_parameters()
        )
    ]
    target_params = [
        i[1]
        for i in filter(
            lambda kv: kv[0].startswith("target_"),
            model.module.named_parameters(),
        )
    ]
    inter_params = [
        i[1]
        for i in filter(
            lambda kv: kv[0].startswith("inter_"),
            model.module.named_parameters(),
        )
    ]
    ms_lr = [init_lr * i for i in args.ms_lr]
    logger.info(f"=> use ms_lr of {ms_lr}")
    optim_params = [
        {"params": context_params, "lr": ms_lr[0]},
        {"params": target_params, "lr": ms_lr[1]},
        {"params": inter_params, "lr": ms_lr[2]},
    ]

    optimizer = torch.optim.Adam(optim_params, lr=init_lr)  # todo: hardcoded
    logger.info("=> use custom Adam optimiser!")

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            for i in optimizer.param_groups:
                i["eps"] = 0.1  # todo: hardcoded
            scaler.load_state_dict(checkpoint["scaler"])
            logger.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
            del checkpoint
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    best_loss = 255
    for epoch in range(args.start_epoch, args.epochs):
        start = time.time()

        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        loss = train(
            train_loader,
            model,
            optimizer,
            epoch,
            scaler,
        )

        # reset dataset for next epoch
        if "camelyon16" in args.data_name:
            if args.rank == 0:
                objects = [train_loader.dataset.reset_data()]
            else:
                objects = [None]
            dist.broadcast_object_list(objects, src=0)
            train_loader.dataset.filename_imgs = objects[0]
            logger.info(f"=> reset dataset for next epoch!")

        if not args.multiprocessing_distributed or args.rank % ngpus_per_node == 0:
            is_best = loss <= best_loss
            best_loss = min(loss, best_loss)

            if args.tensorboard:
                tb_writer.add_scalar("train/loss", loss, epoch)

            # Wandb log
            if args.wandb:
                wandb.log({f"train_loss": loss})
                wandb.run.summary["train_loss"] = best_loss

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
    if args.rank == 0:
        if args.tensorboard:
            tb_writer.close()

        if args.wandb:
            shutil.copyfile(
                os.path.join(args.log_dir, "log.txt"),
                os.path.join(wandb.run.dir, "train_output.log"),
            )
            logger.info(f"=> Log is copied into Wandb folder!")
            wandb.finish()


def train(train_loader, model, optimizer, epoch, scaler):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time],
        prefix="Epoch: [{}]".format(epoch),
    )
    logger = logging.getLogger("MSF-WSI")
    logger.info(f"=> begin epoch {epoch}")

    # switch to train mode
    model.train()
    ddp_loss = torch.zeros(2).cuda(args.gpu)
    contrast_loss = nn.CosineSimilarity(dim=1).cuda(args.gpu)

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
        with autocast(enabled=args.amp, dtype=args.amp_dtype):
            outputs = model(
                (context_img[0], target_img[0]),
                (context_img[1], target_img[1]),
                jigsaw_idx,
            )

            context_loss = 0
            for i, (p1, p2, z1, z2) in enumerate(zip(*outputs[0])):
                context_loss += (
                    -(contrast_loss(p1, z2).mean() + contrast_loss(p2, z1).mean()) * 0.5
                ) * args.fuser_weights[i]

            target_loss = 0
            for i, (p1, p2, z1, z2) in enumerate(zip(*outputs[1])):
                target_loss += (
                    -(contrast_loss(p1, z2).mean() + contrast_loss(p2, z1).mean()) * 0.5
                ) * args.fuser_weights[i]

            fuser_loss = 0
            for i, (p1, p2, z1, z2) in enumerate(zip(*outputs[2])):
                fuser_loss += (
                    -(contrast_loss(p1, z2).mean() + contrast_loss(p2, z1).mean()) * 0.5
                ) * args.fuser_weights[i]

            loss = context_loss + target_loss + fuser_loss
            ddp_loss[0] += loss.item() * bs
            ddp_loss[1] += bs

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

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    losses = ddp_loss[0] / ddp_loss[1]

    return losses.item()


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


def simsiam_loss(p1, p2, z1, z2, loss_fn):
    """
    Compute the contrastive loss between the two images.
    """
    return -(loss_fn(p1, z2).mean() + loss_fn(p2, z1).mean()) * 0.5


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
    parser = argparse.ArgumentParser(description="MSF-WSI pre-training")
    parser.add_argument("-a", "--arch", default="resnet18")
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument("-j", "--workers", default=8, type=int)
    parser.add_argument("-p", "--print-freq", default=50, type=int)
    parser.add_argument("--lr", "--learning-rate", default=1e-3, type=float)
    parser.add_argument("--wd", "--weight-decay", default=1e-2, type=float)
    parser.add_argument(
        "--epochs", default=300, type=int, help="number of total epochs to run"
    )
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--world-size", default=-1, type=int)
    parser.add_argument(
        "--rank", default=-1, type=int, help="node rank for distributed training"
    )
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
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
        "--seed", default=3407, type=int, help="seed for initializing training. "
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
    parser.add_argument("--pred-dim", default=512, type=int)

    # Data settings
    parser.add_argument("--data-name", type=str)
    parser.add_argument("--data", metavar="DIR", help="path to dataset")
    parser.add_argument("--mean", nargs=3, type=float, default=[0.485, 0.456, 0.406])
    parser.add_argument("--std", nargs=3, type=float, default=[0.229, 0.224, 0.225])
    parser.add_argument("-i", "--img-sz", type=int, default=224)
    parser.add_argument("--fold", type=int, default=0)

    # Log setting
    parser.add_argument("--logger-name", default="MSF-WSI", type=str)
    parser.add_argument("--log-dir", default="./logs/temp", type=str)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="use wandb as log tool.")
    parser.add_argument("--run-group", default=None, type=str)
    parser.add_argument("--run-tag", nargs="*", default=None, type=str)
    parser.add_argument("--run-name", default=None, type=str)
    parser.add_argument("--run-notes", default="PyTorch MSF-WSI training", type=str)

    # MSF-WSI specific configs:
    parser.add_argument("--quick-test", action="store_true", help="quick test mode")
    parser.add_argument(
        "--save-freq", default=50, type=int, help="save frequency (default: 50)"
    )
    parser.add_argument("--mask_ratio", type=int, default=50)
    parser.add_argument("--tf32", action="store_true", help="use tf32 for training")
    parser.add_argument("--amp", action="store_true", help="use amp for training")
    parser.add_argument("--bf16", action="store_true", help="use bf16 for training")
    parser.add_argument(
        "--use-ac", action="store_true", help="use activation checkpoint"
    )
    parser.add_argument("--ms_lr", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument(
        "--fuser_weights", nargs=4, type=float, default=[0.1, 0.4, 0.7, 1.0]
    )

    args = parser.parse_args()
    main(args)
