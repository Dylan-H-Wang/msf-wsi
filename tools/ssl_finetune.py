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
from collections import OrderedDict

import numpy as np
import segmentation_models_pytorch as smp
import albumentations as albu
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn.parallel
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.split(SCRIPT_DIR)[0]
sys.path.insert(0, ROOT_PATH)


from src.models.hooknet import HookNet
from src.utils.logger import setup_logger
from src.utils.utils import increment_path
from src.utils.data.bcss import BcssSegDatasetMS, BcssSegDatasetValMS
from src.utils.data.paip import PaipSegDatasetMS, PaipSegDatasetValMS


PAIP_CLASSES = ["tissue", "whole", "viable"]
BCSS_CLASSES = ["tumor", "stroma", "infla", "necr", "other"]
C16_CLASSES = ["tissue", "tumour"]


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


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
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
        logger = setup_logger(args.log_dir)
        logger.info(" ".join([sys.executable, *sys.argv]))
        logger.info("=> initialise python logger successfully!")

        if args.tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            tb_path = str(increment_path(f"{args.log_dir}/tb_log/exp", sep="_", mkdir=True))
            tb_writer = SummaryWriter(tb_path)
            logger.info("Initialise tensorboard logger successfully!")

        if args.wandb:
            import wandb

            wandb.init(
                project="DSF-WSI Experiments",
                notes=args.run_notes,
                tags=args.run_tag,
                group=args.run_group,
                name=args.run_name,
                job_type="fine-tune",
                dir=args.log_dir,
                config=args,
            )
            logger.info("=> initialise wandb logger successfully!")

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()

    if args.data_name == "bcss":
        args.class_names = BCSS_CLASSES
    elif args.data_name == "paip":
        args.class_names = PAIP_CLASSES

    # create model
    logger.info("=> creating model '{}'".format(args.arch))
    model = HookNet(encoder_name=args.arch, encoder_weights=None, classes=len(args.class_names) + 1)

    if os.path.isfile(args.weights):
        logger.info(f"=> loading DSF-WSI pretrained weights {args.weights} into encoder")
        state_dict = torch.load(args.weights, map_location="cpu")["state_dict"]

        context_state_dict = OrderedDict()
        target_state_dict = OrderedDict()

        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith("module.context_encoder") and not k.startswith(
                "module.context_encoder.fc"
            ):
                # remove prefix
                context_state_dict[k[len("module.context_encoder.") :]] = state_dict[k]

            elif k.startswith("module.target_encoder") and not k.startswith(
                "module.target_encoder.fc"
            ):
                target_state_dict[k[len("module.target_encoder.") :]] = state_dict[k]

            # delete renamed or unused k
            del state_dict[k]

        model.context_branch.encoder.load_state_dict(context_state_dict)
        model.target_branch.encoder.load_state_dict(target_state_dict)

        logger.info(f"=> loaded DSF-WSI pretrained weights {args.weights} into encoder")
    else:
        logger.warning(f"=> Invalid model weights!")
        sys.exit(1)

    # infer learning rate before changing batch size
    init_lr = args.lr * math.sqrt(args.batch_size) / math.sqrt(64)
    # init_lr = args.lr
    logger.info(f"=> scale lr from {args.lr:.4f} to {init_lr:.4f}")

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
    train_aug = [
        albu.Compose(
            [
                albu.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                albu.HorizontalFlip(p=0.5),
            ]
        ),  # context aug
        albu.Compose(
            [
                albu.CenterCrop(256, 256, always_apply=True),
            ]
        ),  # target aug
        albu.Compose(
            [
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
        ),  # general aug
    ]
    logger.info(f"=> train aug pipeline: {pformat(train_aug)}")

    val_aug = [
        albu.Compose(
            [
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
        ),
        albu.Compose(
            [
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
        ),
    ]
    logger.info(f"=> val aug pipeline: {pformat(val_aug)}")

    if args.data_name == "bcss":
        train_dataset = BcssSegDatasetMS(args.train_data, train_aug, frac=args.frac, fold=args.fold)
        val_dataset = BcssSegDatasetValMS(args.train_data, val_aug, fold=args.fold)

    elif args.data_name == "paip":
        train_dataset = PaipSegDatasetMS(args.train_data, train_aug, frac=args.frac, fold=args.fold)
        val_dataset = PaipSegDatasetValMS(args.train_data, val_aug, fold=args.fold)

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

    # define loss function (criterion) and optimizer
    cls_idx = list(range(1, len(args.class_names) + 1))
    criterion = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, classes=cls_idx, from_logits=True)
    optimizer = optim.Adam(model.parameters(), init_lr)
    logger.info("=> use custom Adam optimiser!")

    micro_f1_recorder = BestRecorder("max")
    micro_iou_recorder = BestRecorder("max")
    micro_acc_recorder = BestRecorder("max")
    raw_f1_recorders = {i: BestRecorder("max") for i in args.class_names}
    raw_iou_recorders = {i: BestRecorder("max") for i in args.class_names}
    raw_acc_recorders = {i: BestRecorder("max") for i in args.class_names}

    for epoch in range(args.epochs):
        start = time.time()

        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        loss, tp, fp, fn, tn = train(train_loader, model, criterion, optimizer, epoch, scaler, args)

        if not args.multiprocessing_distributed or args.rank == 0:
            # validation
            (
                val_f1_micro,
                val_iou_micro,
                val_acc_micro,
                val_raw_f1,
                val_raw_iou,
                val_raw_acc,
            ) = validate(val_loader, model, epoch, args)

            train_f1_micro = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")

            best_f1_micro, is_best = micro_f1_recorder.update(val_f1_micro)
            best_iou_micro, _ = micro_iou_recorder.update(val_iou_micro)
            best_acc_micro, _ = micro_acc_recorder.update(val_acc_micro)

            for cls_name in args.class_names:
                raw_f1_recorders[cls_name].update(np.mean(val_raw_f1[cls_name]))
                raw_iou_recorders[cls_name].update(np.mean(val_raw_iou[cls_name]))
                raw_acc_recorders[cls_name].update(np.mean(val_raw_acc[cls_name]))

            if args.tensorboard:
                tb_writer.add_scalar("train/loss", loss, epoch)
                tb_writer.add_scalars(
                    "train/f1", {"micro": train_f1_micro}, epoch
                )

                tb_writer.add_scalars(
                    "val/f1", {"micro": val_f1_micro}, epoch
                )
                tb_writer.add_scalars(
                    "val/iou", {"micro": val_iou_micro}, epoch
                )
                tb_writer.add_scalars(
                    "val/acc", {"micro": val_acc_micro}, epoch
                )

            # Wandb log
            if args.wandb:
                wandb.log({"train_f1_micro": train_f1_micro, "val_f1_micro": val_f1_micro})
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
                    filename=f"{args.log_dir}/best_ft_model.pth.tar",
                )
                logger.info(f"=> Best model saved at epoch {epoch}!")

            elapsed_time = (time.time() - start) / 60
            logger.info(
                "=======\n"
                f"TIME: {elapsed_time:.2f} mins, LOSS: {loss:.4f}\n"
                f"MICRO F1: {train_f1_micro:.4f}/{val_f1_micro:.4f}/{best_f1_micro:.4f}\n"
                f"MICRO IOU: {val_iou_micro:.4f}/{best_iou_micro:.4f}\n"
                f"MICRO ACC: {val_acc_micro:.4f}/{best_acc_micro:.4f}\n"
                "======="
            )

    if args.rank == 0:
        logger.info("=> Best scores:")
        logger.info(
            "=======\n"
            f"MICRO F1: {micro_f1_recorder.best:.4f}\n"
            f"MICRO IOU: {micro_iou_recorder.best:.4f}\n"
            f"MICRO ACC: {micro_acc_recorder.best:.4f}\n"
        )
        for cls_name in args.class_names:
            logger.info(
                f"{cls_name} F1: {raw_f1_recorders[cls_name].best:.4f}, IOU: {raw_iou_recorders[cls_name].best:.4f}, ACC: {raw_acc_recorders[cls_name].best:.4f}"
            )

    # Wandb log
    if args.rank == 0:
        if args.tensorboard:
            tb_writer.close()

        if args.wandb:
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
    logger = logging.getLogger("DSF-WSI")

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

        images[0] = images[0].cuda(args.gpu, non_blocking=True)
        images[1] = images[1].cuda(args.gpu, non_blocking=True)
        masks[0] = masks[0].long().cuda(args.gpu, non_blocking=True)
        masks[1] = masks[1].long().cuda(args.gpu, non_blocking=True)

        # compute output and loss
        with autocast(enabled=args.amp):
            context_logits_mask, target_logits_mask = model(images[0], images[1])
            loss = (1 - args.lam) * criterion(context_logits_mask, masks[0]) + args.lam * criterion(
                target_logits_mask, masks[1]
            )

        losses.update(loss.item(), args.batch_size)

        pred_mask = torch.argmax(target_logits_mask.detach(), dim=1)
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long() - 1,
            masks[1] - 1,
            mode="multiclass",
            ignore_index=-1,
            num_classes=len(args.class_names),
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

        del loss, images, masks, context_logits_mask, target_logits_mask, pred_mask

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


@torch.no_grad()
def validate(loader, model, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time],
        prefix="Val epoch: [{}]".format(epoch),
    )
    logger = logging.getLogger("DSF-WSI")

    val_f1_micros = []
    val_iou_micros = []
    val_acc_micros = []
    class_f1 = {i: [] for i in args.class_names}
    class_iou = {i: [] for i in args.class_names}
    class_acc = {i: [] for i in args.class_names}

    # switch to eval mode
    model.eval()

    end = time.time()
    for i, (images, masks) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        context_imgs = images[0][0]
        target_imgs = images[1][0]
        target_masks = masks[1][0]

        context_imgs_split = torch.split(context_imgs, 128)
        target_imgs_split = torch.split(target_imgs, 128)

        preds = []
        for img1, img2 in zip(context_imgs_split, target_imgs_split):
            img1 = img1.cuda(args.gpu, non_blocking=True)
            img2 = img2.cuda(args.gpu, non_blocking=True)

            # compute output and loss
            with autocast(enabled=args.amp):
                _, target_logits_mask = model(img1, img2)
                preds.append(target_logits_mask.detach().cpu())

        preds = torch.cat(preds, dim=0)
        pred_mask = torch.argmax(preds, dim=1)
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long() - 1,
            target_masks.long() - 1,
            mode="multiclass",
            ignore_index=-1,
            num_classes=len(args.class_names),
        )

        val_f1_micro = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        val_iou_micro = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        val_acc_micro = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        val_f1_micros.append(val_f1_micro.item())
        val_iou_micros.append(val_iou_micro.item())
        val_acc_micros.append(val_acc_micro.item())

        tp = tp.sum(0)
        fp = fp.sum(0)
        fn = fn.sum(0)
        tn = tn.sum(0)
        val_raw_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction=None)
        val_raw_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None)
        val_raw_acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction=None)
        for idx, cls_name in enumerate(args.class_names):
            class_f1[cls_name].append(val_raw_f1[idx].item())
            class_iou[cls_name].append(val_raw_iou[idx].item())
            class_acc[cls_name].append(val_raw_acc[idx].item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(progress.display(i))

    return (
        np.mean(val_f1_micros),
        np.mean(val_iou_micros),
        np.mean(val_acc_micros),
        class_f1,
        class_iou,
        class_acc,
    )


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
    parser = argparse.ArgumentParser(description="PyTorch DSF-WSI Testing")

    parser.add_argument("-a", "--arch", default="resnet18", help="model architecture")
    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("-j", "--workers", default=4, type=int)
    parser.add_argument("-p", "--print-freq", default=50, type=int, help="print frequency")
    parser.add_argument("--epochs", default=50, type=int, help="number of total epochs to run")
    parser.add_argument("--lr", default=1e-3, type=float, help="initial (base) learning rate")
    parser.add_argument("--world-size", default=-1, type=int)
    parser.add_argument("--rank", default=-1, type=int, help="node rank for distributed training")
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        type=str,
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--seed", type=int, help="seed for initializing training. ")
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
    parser.add_argument("--data-name", type=str, default="bcss")
    parser.add_argument("--train-data", type=str, help="train set path")
    parser.add_argument("--mean", nargs=3, type=float, default=[0.485, 0.456, 0.406])
    parser.add_argument("--std", nargs=3, type=float, default=[0.229, 0.224, 0.225])
    parser.add_argument("--fold", type=int, default=0)

    # Log setting
    parser.add_argument("--log-dir", default="./logs/temp", type=str)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="use wandb as log tool.")
    parser.add_argument("--run-group", default=None, type=str)
    parser.add_argument("--run-tag", nargs="*", default=None, type=str)
    parser.add_argument("--run-name", default=None, type=str)
    parser.add_argument("--run-notes", default="PyTorch DSF-WSI training", type=str)

    # Others
    parser.add_argument("--tf32", action="store_true", help="use tf32 for training")
    parser.add_argument("--amp", action="store_true", help="use amp for training")
    parser.add_argument("--bf16", action="store_true", help="use bf16 for training")
    parser.add_argument("--weights", type=str)
    parser.add_argument("--frac", type=float, default=1)
    parser.add_argument("--lam", type=float, default=1)

    args = parser.parse_args()

    main(args)
