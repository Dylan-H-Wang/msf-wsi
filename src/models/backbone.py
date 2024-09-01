from functools import partial

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)


def make_projector(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, in_dim, bias=False),
        nn.BatchNorm1d(in_dim),
        nn.ReLU(inplace=True),  # first layer
        nn.Linear(in_dim, in_dim, bias=False),
        nn.BatchNorm1d(in_dim),
        nn.ReLU(inplace=True),  # second layer
        nn.Linear(in_dim, out_dim, bias=False),
        nn.BatchNorm1d(out_dim, affine=False),
    )


def make_predictor(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim, bias=False),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True),  # hidden layer
        nn.Linear(out_dim, in_dim),  # output layer
    )


class MSFWSI(nn.Module):
    """
    Build a MSF-WSI backbone model
    """

    def __init__(
        self,
        base_encoder,
        scale,
        dim=2048,
        pred_dim=512,
        mask_ratio=0.5,
        use_checkpoint=False,
    ):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super().__init__()

        self.K = int(scale**2)  # corresbonding target patches for each context patch
        self.n_keep = int(self.K * (1 - mask_ratio))  # unmasked target features

        # create encoders
        self.context_encoder = base_encoder(
            zero_init_residual=True, pretrained=True, return_features=True
        )
        self.target_encoder = base_encoder(
            zero_init_residual=True, pretrained=True, return_features=True
        )
        self.context_encoder.fc = nn.Identity()
        self.target_encoder.fc = nn.Identity()

        self.inter_dim = torch.as_tensor([64, 128, 256, 512])  # TODO: hardcoded
        self.ms_inter_dim = self.inter_dim * (
            self.n_keep + 1
        )  # feature dim after concat context and target

        # build a 3-layer projector
        self.context_projector = nn.ModuleList(
            [make_projector(d, d) for d in self.inter_dim]
        )
        self.target_projector = nn.ModuleList(
            [make_projector(d, d) for d in self.inter_dim]
        )
        self.inter_projector = nn.ModuleList(
            [make_projector(d, d) for d in self.ms_inter_dim]
        )

        # build a 2-layer predictor
        self.context_predictor = nn.ModuleList(
            [
                make_predictor(d, torch.div(d, 4, rounding_mode="floor"))
                for d in self.inter_dim
            ]
        )
        self.target_predictor = nn.ModuleList(
            [
                make_predictor(d, torch.div(d, 4, rounding_mode="floor"))
                for d in self.inter_dim
            ]
        )
        self.inter_predictor = nn.ModuleList(
            [
                make_predictor(d, torch.div(d, 4, rounding_mode="floor"))
                for d in self.ms_inter_dim
            ]
        )

        if use_checkpoint:
            self._apply_checkpoint()

    def _apply_checkpoint(self):
        checkpoint_impl = CheckpointImpl.NO_REENTRANT
        checkpoint_wrapper_fn = partial(
            checkpoint_wrapper,
            offload_to_cpu=False,
            checkpoint_impl=checkpoint_impl,
        )

        check_fn = lambda submodule: isinstance(submodule, (nn.Conv2d, nn.Linear))
        apply_activation_checkpointing(
            self,
            checkpoint_wrapper_fn=checkpoint_wrapper_fn,
            check_fn=check_fn,
        )

        # disable checkpointing for the first conv layer
        self.context_encoder.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.target_encoder.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

    def forward(self, x1, x2, jigsaw_idx=None):
        """
        Input:
            x1: first views of images [context_images, target_images]
            x2: second views of images [context_images, target_images]
        Output:
            p1, p2, z1, z2: predictors and targets of the network
        """
        B = x1[0].shape[0]

        # Features from encoder
        context_f1, context_f2 = self.context_encoder(x1[0]), self.context_encoder(
            x2[0]
        )  # BxC
        target_f1, target_f2 = self.target_encoder(x1[1]), self.target_encoder(
            x2[1]
        )  # BKxC

        target_f1_split = tuple(i.reshape(B, self.K, -1) for i in target_f1)  # BxKxC
        target_f2_split = tuple(i.reshape(B, self.K, -1) for i in target_f2)  # BxKxC

        # reorder target patches
        batch_idx = torch.arange(B).repeat(self.K, 1).t()
        assert batch_idx.shape == jigsaw_idx[0].shape == jigsaw_idx[1].shape
        target_f1_sort = tuple(
            i[batch_idx, jigsaw_idx[0], :].flatten(0, 1) for i in target_f1_split
        )  # BKxC
        target_f2_sort = tuple(
            i[batch_idx, jigsaw_idx[1], :].flatten(0, 1) for i in target_f2_split
        )  # BKxC

        # Features from projector
        context_z1 = tuple(
            f(x) for f, x in zip(self.context_projector, context_f1)
        )  # BxC
        context_z2 = tuple(
            f(x) for f, x in zip(self.context_projector, context_f2)
        )  # BxC
        target_z1 = tuple(
            f(x) for f, x in zip(self.target_projector, target_f1_sort)
        )  # BKxC
        target_z2 = tuple(
            f(x) for f, x in zip(self.target_projector, target_f2_sort)
        )  # BKxC

        # Features from predictor
        context_p1 = tuple(
            f(x) for f, x in zip(self.context_predictor, context_z1)
        )  # BxC
        context_p2 = tuple(
            f(x) for f, x in zip(self.context_predictor, context_z2)
        )  # BxC
        target_p1 = tuple(
            f(x) for f, x in zip(self.target_predictor, target_z1)
        )  # BKxC
        target_p2 = tuple(
            f(x) for f, x in zip(self.target_predictor, target_z2)
        )  # BKxC

        context_z1_detach = tuple(i.detach() for i in context_z1)
        context_z2_detach = tuple(i.detach() for i in context_z2)
        target_z1_detach = tuple(i.detach() for i in target_z1)
        target_z2_detach = tuple(i.detach() for i in target_z2)

        ### Fuser path
        # Combine context and target features
        ms_f1 = tuple(
            torch.cat((i, j[:, : self.n_keep, :].flatten(1)), dim=1)
            for i, j in zip(context_f1, target_f1_split)
        )  # Bx(K+1)C
        ms_f2 = tuple(
            torch.cat((i, j[:, : self.n_keep, :].flatten(1)), dim=1)
            for i, j in zip(context_f2, target_f2_split)
        )  # Bx(K+1)C

        # fuser contrastive learning
        ms_z1, ms_z2, ms_p1, ms_p2 = [], [], [], []
        for i, (f_proj, f_pred) in enumerate(
            zip(self.inter_projector, self.inter_predictor)
        ):
            ms_z1.append(f_proj(ms_f1[i]))
            ms_z2.append(f_proj(ms_f2[i]))
            ms_p1.append(f_pred(ms_z1[i]))
            ms_p2.append(f_pred(ms_z2[i]))

        ms_z1_detach = tuple(i.detach() for i in ms_z1)
        ms_z2_detach = tuple(i.detach() for i in ms_z2)
        ms_p1, ms_p2 = tuple(ms_p1), tuple(ms_p2)

        return (
            (context_p1, context_p2, context_z1_detach, context_z2_detach),
            (target_p1, target_p2, target_z1_detach, target_z2_detach),
            (ms_p1, ms_p2, ms_z1_detach, ms_z2_detach),
        )
