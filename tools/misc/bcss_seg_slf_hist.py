import os
import sys
import time
import random
import shutil
import logging
import warnings
import argparse
import traceback
from collections import OrderedDict

import segmentation_models_pytorch as smp
import albumentations as albu
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.cuda.amp import autocast, GradScaler

import torchvision.models as models

from simsiam.loader import *
from simsiam.resnet import *
from simsiam.unet import MSUnet
from dataset.bcss import *
from utils.utils import increment_path
from utils.logger import setup_logger

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


def main(args):
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


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    logger = logging.getLogger()

    # only log if master(rank0)
    if args.multiprocessing_distributed and args.gpu == 0:
        logger = setup_logger(args.log_dir)
        logger.info(" ".join([sys.executable, *sys.argv]))
        logger.info("=> initialise python logger successfully!")

        if args.wandb:
            import wandb

            wandb.init(
                project="SLF-WSI Ablation",
                notes=args.run_notes,
                tags=args.run_tag,
                group=args.run_group,
                name=args.run_name,
                job_type="fine-tune",
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

    # create model
    logger.info("=> creating model '{}'".format(args.arch))
    if args.use_ms:
        model = MSUnet(encoder_name=args.arch, encoder_weights=None, classes=6)
    else:
        model = smp.Unet(
            encoder_name=args.arch,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=None,
            classes=6,
        )
    
    weights = torch.load("./logs/hooknet/bcss/slf_hist/tenpercent_resnet18_v2.pth", map_location="cpu")
    model.context_branch.encoder.load_state_dict(weights)
    model.target_branch.encoder.load_state_dict(weights)

    # infer learning rate before changing batch size
    # init_lr = args.lr * args.batch_size / 32
    # logger.info(f"=> changed lr based on batch size to {init_lr}")
    init_lr = args.lr
    logger.info(f"=> use lr of {init_lr}")

    if args.distributed:
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
    logger.info(model)  # print model after SyncBatchNorm

    # define loss function (criterion) and optimizer
    criterion = smp.losses.DiceLoss(
        smp.losses.MULTICLASS_MODE, classes=[1, 2, 3, 4, 5], from_logits=True
    )
    optimizer = optim.Adam(model.parameters(), init_lr)

    if args.amp:
        logger.info(f"=> enable automatic mix precision training!")
    scaler = GradScaler(enabled=args.amp)

    cudnn.benchmark = True

    # Data loading code
    if args.use_ms:
        train_context_transform = [
            albu.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            albu.HorizontalFlip(p=0.5),
        ]
        train_target_transform = [
            albu.CenterCrop(256, 256, always_apply=True),
        ]
        train_misc_transform = [
            albu.Resize(256, 256, always_apply=True),
            albu.Normalize(
                mean=args.mean,
                std=args.std,
                max_pixel_value=255.0,
                always_apply=True,
                p=1.0,
            ),
            ToTensorV2(transpose_mask=True),
        ]
        logger.info(f"=> train context aug pipeline: {train_context_transform}")
        logger.info(f"=> train target aug pipeline: {train_target_transform}")
        train_aug = [
            albu.Compose(train_context_transform),
            albu.Compose(train_target_transform),
            albu.Compose(train_misc_transform),
        ]

        val_context_transform = [
            albu.Resize(256, 256, always_apply=True),
            albu.Normalize(
                mean=args.mean,
                std=args.std,
                max_pixel_value=255.0,
                always_apply=True,
                p=1.0,
            ),
            ToTensorV2(transpose_mask=True),
        ]
        val_target_transform = [
            albu.CenterCrop(256, 256, always_apply=True),
            albu.Normalize(
                mean=args.mean,
                std=args.std,
                max_pixel_value=255.0,
                always_apply=True,
                p=1.0,
            ),
            ToTensorV2(transpose_mask=True),
        ]
        val_aug = [
            albu.Compose(val_context_transform),
            albu.Compose(val_target_transform),
        ]

        train_dataset = BcssSegDatasetMS(args.train_data, train_aug, frac=args.frac, fold=args.fold)
        val_dataset = BcssSegDatasetValMS(args.train_data, val_aug, fold=args.fold)

    else:
        train_transform = [
            albu.CenterCrop(256, 256, always_apply=True),
            albu.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            albu.HorizontalFlip(p=0.5),
            albu.Normalize(
                mean=args.mean,
                std=args.std,
                max_pixel_value=255.0,
                always_apply=True,
                p=1.0,
            ),
            ToTensorV2(transpose_mask=True),
        ]
        logger.info(f"=> train aug pipeline: {train_transform}")
        train_aug = albu.Compose(train_transform)

        val_transform = [
            albu.CenterCrop(256, 256, always_apply=True),
            albu.Normalize(
                mean=args.mean,
                std=args.std,
                max_pixel_value=255.0,
                always_apply=True,
                p=1.0,
            ),
            ToTensorV2(transpose_mask=True),
        ]
        val_aug = albu.Compose(val_transform)

        train_dataset = BcssSegDataset(args.train_data, train_aug, frac=args.frac)
        val_dataset = BcssSegDatasetVal(args.train_data, val_aug)

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
        drop_last=False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    loss_recorder = BestRecorder("min")
    micro_f1_recorder = BestRecorder("max")
    macro_f1_recorder = BestRecorder("max")
    tumor_f1_recorder = BestRecorder("max")
    stroma_f1_recorder = BestRecorder("max")
    infla_f1_recorder = BestRecorder("max")
    necr_f1_recorder = BestRecorder("max")
    other_f1_recorder = BestRecorder("max")
    micro_acc_recorder = BestRecorder("max")
    macro_acc_recorder = BestRecorder("max")

    for epoch in range(args.epochs):
        start = time.time()

        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        loss, tp, fp, fn, tn = train(
            train_loader, model, criterion, optimizer, epoch, scaler, args
        )
        (
            val_f1_micro,
            val_f1_macro,
            tumor_f1_,
            stroma_f1_,
            infla_f1_,
            necr_f1_,
            other_f1_,
            val_acc_micro,
            val_acc_macro,
        ) = validate(val_loader, model, epoch, args)

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            scores = smp.metrics.f1_score(tp, fp, fn, tn, reduction=None)
            train_f1_micro = smp.metrics.f1_score(
                tp, fp, fn, tn, reduction="micro-imagewise"
            )
            train_f1_macro = smp.metrics.f1_score(
                tp, fp, fn, tn, reduction="macro-imagewise"
            )
            tumor_f1, stroma_f1, infla_f1, necr_f1, other_f1 = scores.mean(0)

            best_f1_micro, is_best = micro_f1_recorder.update(val_f1_micro)
            best_f1_macro, _ = macro_f1_recorder.update(val_f1_macro)
            best_tumor_f1, _ = tumor_f1_recorder.update(tumor_f1_)
            best_stroma_f1, _ = stroma_f1_recorder.update(stroma_f1_)
            best_infla_f1, _ = infla_f1_recorder.update(infla_f1_)
            best_necr_f1, _ = necr_f1_recorder.update(necr_f1_)
            best_other_f1, _ = other_f1_recorder.update(other_f1_)
            best_acc_micro, _ = micro_acc_recorder.update(val_acc_micro)
            best_acc_macro, _ = macro_acc_recorder.update(val_acc_macro)

            # Wandb log
            if args.wandb:
                wandb.log({f"train_f1_micro": train_f1_micro})
                wandb.log({f"val_f1_micro": val_f1_micro})
                wandb.run.summary["best_val_f1_micro"] = best_f1_micro

            if is_best:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": args.arch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                    },
                    is_best=False,
                    # filename="{}/checkpoint_{:04d}.pth.tar".format(args.log_dir, epoch),
                    filename=f"{args.log_dir}/best_ft_model.pth.tar",
                )
                logger.info(f"=> Best model saved at epoch {epoch}!")

            elapsed_time = (time.time() - start) / 60
            logger.info(
                "=======\n"
                f"TIME: {elapsed_time:.2f} mins, LOSS: {loss:.4f}\n"
                f"MICRO F1: {train_f1_micro:.4f}/{val_f1_micro:.4f}/{best_f1_micro:.4f}, MACRO F1: {train_f1_macro:.4f}/{val_f1_macro:.4f}/{best_f1_macro:.4f}\n"
                f"TUMOR F1: {tumor_f1:.4f}/{tumor_f1_:.4f}/{best_tumor_f1:.4f}, STROMA F1: {stroma_f1:.4f}/{stroma_f1_:.4f}/{best_stroma_f1:.4f}\n"
                f"INFLA F1: {infla_f1:.4f}/{infla_f1_:.4f}/{best_infla_f1:.4f}, NECR F1: {necr_f1:.4f}/{necr_f1_:.4f}/{best_necr_f1:.4f}\n"
                f"OTHER F1: {other_f1:.4f}/{other_f1_:.4f}/{best_other_f1:.4f}\n"
                f"MICRO ACC: {val_acc_micro:.4f}/{best_acc_micro:.4f}, MACRO ACC: {val_acc_macro:.4f}/{best_acc_macro:.4f}\n"
                "======="
            )

    # Wandb log
    if (
        args.multiprocessing_distributed
        and args.rank % ngpus_per_node == 0
        and args.wandb
    ):
        shutil.copyfile(
            os.path.join(args.log_dir, "log.txt"),
            os.path.join(wandb.run.dir, "finetune_output.log"),
        )
        logger.info(f"=> Log is copied into Wandb folder!")
        wandb.finish()



def train(loader, model, criterion, optimizer, epoch, scaler, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses],
        prefix="Train epoch: [{}]".format(epoch),
    )
    logger = logging.getLogger("SLF-WSI")

    tp_all = []
    fp_all = []
    fn_all = []
    tn_all = []

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, masks) in enumerate(loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            if args.use_ms:
                images[0] = images[0].cuda(args.gpu, non_blocking=True)
                images[1] = images[1].cuda(args.gpu, non_blocking=True)
                masks[0] = masks[0].long().cuda(args.gpu, non_blocking=True)
                masks[1] = masks[1].long().cuda(args.gpu, non_blocking=True)
            else:
                images = images.cuda(args.gpu, non_blocking=True)
                masks = masks.long().cuda(args.gpu, non_blocking=True)

        # compute output and loss
        with autocast(enabled=args.amp):
            if args.use_ms:
                context_logits_mask, target_logits_mask = model(images[0], images[1])
                loss = (1 - args.lam) * criterion(
                    context_logits_mask, masks[0]
                ) + args.lam * criterion(
                    target_logits_mask, masks[1]
                )  # TODO: fine-tune the lambda
            else:
                logits_mask = model(images)
                loss = criterion(logits_mask, masks)

        losses.update(loss.item(), args.batch_size)

        if args.use_ms:
            pred_mask = torch.argmax(target_logits_mask.detach(), dim=1)
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred_mask.long() - 1,
                masks[1].detach() - 1,
                mode="multiclass",
                ignore_index=-1,
                num_classes=5,
            )
        else:
            pred_mask = torch.argmax(logits_mask.detach(), dim=1)
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred_mask.long() - 1,
                masks.detach() - 1,
                mode="multiclass",
                ignore_index=-1,
                num_classes=5,
            )

        tp_all.append(tp)
        fp_all.append(fp)
        fn_all.append(fn)
        tn_all.append(tn)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if args.use_ms:
            del loss, images, masks, context_logits_mask, target_logits_mask, pred_mask
        else:
            del loss, images, masks, logits_mask, pred_mask

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(progress.display(i))

    tp_all = torch.cat(tp_all, 0)
    fp_all = torch.cat(fp_all, 0)
    fn_all = torch.cat(fn_all, 0)
    tn_all = torch.cat(tn_all, 0)

    return losses.avg, tp_all, fp_all, fn_all, tn_all


def validate(loader, model, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time],
        prefix="Val epoch: [{}]".format(epoch),
    )
    logger = logging.getLogger("SLF-WSI")

    val_f1_micros = []
    val_f1_macros = []
    tumor_f1 = []
    stroma_f1 = []
    infla_f1 = []
    necr_f1 = []
    other_f1 = []
    val_acc_micros = []
    val_acc_macros = []

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, masks) in enumerate(loader):

            # measure data loading time
            data_time.update(time.time() - end)

            if args.use_ms:
                context_imgs = images[0][0]
                target_imgs = images[1][0]
                context_masks = masks[0][0]
                target_masks = masks[1][0]

                context_imgs_split = torch.split(context_imgs, 128)
                target_imgs_split = torch.split(target_imgs, 128)
                context_masks_split = torch.split(context_masks, 128)
                target_masks_split = torch.split(target_masks, 128)
            else:
                images = images[0]
                masks = masks[0]
                images_split = torch.split(images, 128)
                masks_split = torch.split(masks, 128)

            loss = []
            preds = []
            if args.use_ms:
                for img1, mask1, img2, mask2 in zip(
                    context_imgs_split,
                    context_masks_split,
                    target_imgs_split,
                    target_masks_split,
                ):
                    if args.gpu is not None:
                        img1 = img1.cuda(args.gpu, non_blocking=True)
                        mask1 = mask1.long().cuda(args.gpu, non_blocking=True)
                        img2 = img2.cuda(args.gpu, non_blocking=True)
                        mask2 = mask2.long().cuda(args.gpu, non_blocking=True)

                    # compute output and loss
                    with autocast(enabled=args.amp):
                        context_logits_mask, target_logits_mask = model(img1, img2)
                        preds.append(target_logits_mask.detach().cpu())

            else:
                for img, mask in zip(images_split, masks_split):
                    if args.gpu is not None:
                        img = img.cuda(args.gpu, non_blocking=True)
                        mask = mask.long().cuda(args.gpu, non_blocking=True)

                    # compute output and loss
                    with autocast(enabled=args.amp):
                        logits_mask = model(img)
                        preds.append(logits_mask.detach().cpu())

            preds = torch.cat(preds, dim=0)
            pred_mask = torch.argmax(preds, dim=1)

            if args.use_ms:
                tp, fp, fn, tn = smp.metrics.get_stats(
                    pred_mask.long() - 1,
                    target_masks.long() - 1,
                    mode="multiclass",
                    ignore_index=-1,
                    num_classes=5,
                )
            else:
                tp, fp, fn, tn = smp.metrics.get_stats(
                    pred_mask.long() - 1,
                    masks.long() - 1,
                    mode="multiclass",
                    ignore_index=-1,
                    num_classes=5,
                )

            val_f1_micro = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            val_f1_macro = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")
            val_acc_micro = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
            val_acc_macro = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
            tumor_f1_ = (
                smp.metrics.f1_score(
                    tp, fp, fn, tn, reduction="weighted", class_weights=[1, 0, 0, 0, 0]
                )
                * 5
            )
            stroma_f1_ = (
                smp.metrics.f1_score(
                    tp, fp, fn, tn, reduction="weighted", class_weights=[0, 1, 0, 0, 0]
                )
                * 5
            )
            infla_f1_ = (
                smp.metrics.f1_score(
                    tp, fp, fn, tn, reduction="weighted", class_weights=[0, 0, 1, 0, 0]
                )
                * 5
            )
            necr_f1_ = (
                smp.metrics.f1_score(
                    tp, fp, fn, tn, reduction="weighted", class_weights=[0, 0, 0, 1, 0]
                )
                * 5
            )
            other_f1_ = (
                smp.metrics.f1_score(
                    tp, fp, fn, tn, reduction="weighted", class_weights=[0, 0, 0, 0, 1]
                )
                * 5
            )

            val_f1_micros.append(val_f1_micro.item())
            val_f1_macros.append(val_f1_macro.item())
            tumor_f1.append(tumor_f1_.item())
            stroma_f1.append(stroma_f1_.item())
            infla_f1.append(infla_f1_.item())
            necr_f1.append(necr_f1_.item())
            other_f1.append(other_f1_.item())
            val_acc_micros.append(val_acc_micro.item())
            val_acc_macros.append(val_acc_macro.item())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info(progress.display(i))

    return (
        np.mean(val_f1_micros),
        np.mean(val_f1_macros),
        np.mean(tumor_f1),
        np.mean(stroma_f1),
        np.mean(infla_f1),
        np.mean(necr_f1),
        np.mean(other_f1),
        np.mean(val_acc_micros),
        np.mean(val_acc_macros),
    )


def test(model, args):
    pass


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


class BestRecorder(object):
    def __init__(self, mode):
        self.mode = mode

        if mode == "min":
            self.best = 10000
        elif mode == "max":
            self.best = -10000
        else:
            print("invalid mode!")

    def update(self, val):
        if self.mode == "min":
            res = val < self.best
            self.best = min(val, self.best)
            return (self.best, res)

        elif self.mode == "max":
            res = val > self.best
            self.best = max(val, self.best)
            return (self.best, res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch SLF-WSI Testing")

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
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        default=50,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=16,
        type=int,
        metavar="N",
        help="mini-batch size (default: 512), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        metavar="LR",
        help="initial (base) learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        metavar="M",
        help="momentum of SGD solver",
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
    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
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

    # Data settings
    parser.add_argument("--train-data", type=str, help="train set path")
    parser.add_argument("--test-data", type=str, help="test set path")
    parser.add_argument("--mean", nargs=3, type=float, default=[0.485, 0.456, 0.406])
    parser.add_argument("--std", nargs=3, type=float, default=[0.229, 0.224, 0.225])
    parser.add_argument("--fold", type=int, default=0)

    # Log setting
    parser.add_argument("--log-dir", default="./logs/temp", type=str)
    parser.add_argument("--wandb", action="store_true", help="use wandb as log tool.")
    parser.add_argument("--run-group", default=None, type=str)
    parser.add_argument("--run-tag", nargs="*", default=None, type=str)
    parser.add_argument("--run-name", default=None, type=str)
    parser.add_argument("--run-notes", default="PyTorch SLF-WSI training", type=str)

    # Others
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--weights", type=str)
    parser.add_argument("--frac", type=float, default=1)
    parser.add_argument("--use_ms", action="store_true")
    parser.add_argument("--imagenet", action="store_true")
    parser.add_argument("--lam", type=float, default=1)

    args = parser.parse_args()

    main(args)
