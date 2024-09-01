import os
import sys
import time
import random
import logging
import argparse
import traceback
from pprint import pformat

import numpy as np
import segmentation_models_pytorch as smp
import albumentations as albu
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn.parallel
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
from src.utils.data.bcss import BcssSegDatasetValMS
from src.utils.data.paip import PaipSegDatasetValMS


PAIP_CLASSES = ["tissue", "whole", "viable"]
BCSS_CLASSES = ["tumor", "stroma", "infla", "necr", "other"]


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
    model = HookNet(
        encoder_name=args.arch, encoder_weights=None, classes=len(args.class_names) + 1
    )

    logger.info(f"=> loading pretrained weights {args.weights}")
    state_dict = torch.load(args.weights, map_location="cpu")["state_dict"]

    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith("module."):
            # remove prefix
            state_dict[k[len("module.") :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

    model.load_state_dict(state_dict)
    logger.info(f"=> loaded pretrained weights {args.weights}")

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
        val_dataset = BcssSegDatasetValMS(args.train_data, val_aug, fold=args.fold)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
    elif args.data_name == "paip":
        val_dataset = PaipSegDatasetValMS(args.train_data, val_aug, fold=args.fold)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    micro_f1_recorder = BestRecorder("max")
    micro_iou_recorder = BestRecorder("max")
    micro_acc_recorder = BestRecorder("max")
    raw_f1_recorders = {i: BestRecorder("max") for i in args.class_names}
    raw_iou_recorders = {i: BestRecorder("max") for i in args.class_names}
    raw_acc_recorders = {i: BestRecorder("max") for i in args.class_names}

    # validation
    (
        val_f1_micro,
        val_iou_micro,
        val_acc_micro,
        val_raw_f1,
        val_raw_iou,
        val_raw_acc,
    ) = validate(val_loader, model, 1, args)

    best_f1_micro, is_best = micro_f1_recorder.update(val_f1_micro)
    best_iou_micro, _ = micro_iou_recorder.update(val_iou_micro)
    best_acc_micro, _ = micro_acc_recorder.update(val_acc_micro)

    for cls_name in args.class_names:
        raw_f1_recorders[cls_name].update(np.mean(val_raw_f1[cls_name]))
        raw_iou_recorders[cls_name].update(np.mean(val_raw_iou[cls_name]))
        raw_acc_recorders[cls_name].update(np.mean(val_raw_acc[cls_name]))

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
    parser.add_argument(
        "-p", "--print-freq", default=50, type=int, help="print frequency"
    )
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

    # Data settings
    parser.add_argument("--data-name", type=str, default="bcss")
    parser.add_argument("--train-data", type=str, help="train set path")
    parser.add_argument("--mean", nargs=3, type=float, default=[0.485, 0.456, 0.406])
    parser.add_argument("--std", nargs=3, type=float, default=[0.229, 0.224, 0.225])
    parser.add_argument("--fold", type=int, default=0)

    # Log setting
    parser.add_argument("--log-dir", default="./logs/temp", type=str)

    # Others
    parser.add_argument("--tf32", action="store_true", help="use tf32 for training")
    parser.add_argument("--amp", action="store_true", help="use amp for training")
    parser.add_argument("--bf16", action="store_true", help="use bf16 for training")
    parser.add_argument("--weights", type=str)
    parser.add_argument("--weight-name", type=str)
    parser.add_argument("--frac", type=float, default=1)
    parser.add_argument("--lam", type=float, default=1)

    args = parser.parse_args()

    main(args)
