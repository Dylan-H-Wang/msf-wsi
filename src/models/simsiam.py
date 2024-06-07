import random
from functools import partial

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)

from segmentation_models_pytorch.encoders._base import EncoderMixin

from .resnet import MultiPrototypes


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super().__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()

class SimSiam2(SimSiam):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512, feats_idx=[0,1,2], pool_type="max", use_proj=True, use_pred=True):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super().__init__(base_encoder, dim, pred_dim)

        self.out_dims = [64, 128, 256, 512]
        # self.dim = dim
        # self.pred_dim = pred_dim
        self.feats_idx = feats_idx
        self.use_proj = use_proj
        self.use_pred = use_pred

        import types
        def _resnet_fwd(self, x: Tensor) -> Tensor:
            # See note [TorchScript super()]
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            c1 = self.layer1(x)
            c2 = self.layer2(c1)
            c3 = self.layer3(c2)
            c4 = self.layer4(c3)

            x = self.avgpool(c4)
            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x, [c1, c2, c3, c4]
        self.encoder._forward_impl = types.MethodType(_resnet_fwd, self.encoder)
        self.pool = nn.AdaptiveMaxPool2d(7) if pool_type=="max" else nn.AdaptiveAvgPool2d(7) # todo: need experiments

        if self.use_proj:
            self.projectors = nn.ModuleDict([[f"proj{i}", self._make_projector(self.out_dims[i]*7*7, dim)] for i in self.feats_idx])

            if self.use_pred:
                self.predictors = nn.ModuleDict([[f"pred{i}", self._make_predictor(dim, pred_dim)] for i in self.feats_idx])

    def _make_projector(self, prev_dim, dim):
        return nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                nn.BatchNorm1d(prev_dim),
                                nn.ReLU(inplace=True), # first layer
                                nn.Linear(prev_dim, prev_dim, bias=False),
                                nn.BatchNorm1d(prev_dim),
                                nn.ReLU(inplace=True), # second layer
                                nn.Linear(prev_dim, dim, bias=False),
                                nn.BatchNorm1d(dim, affine=False))

    def _make_predictor(self, dim, pred_dim):
        return nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                nn.BatchNorm1d(pred_dim),
                                nn.ReLU(inplace=True), # hidden layer
                                nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1, c1 = self.encoder(x1) # NxC
        z2, c2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        c1 = [c1[i] for i in self.feats_idx]
        c2 = [c2[i] for i in self.feats_idx]
        c1z = None
        c2z = None
        c1p = None
        c2p = None
        
        if self.use_proj:
            c1z = [ self.projectors[f"proj{j}"](self.pool(c1[i]).flatten(1)) for i,j in enumerate(self.feats_idx) ]
            c2z = [ self.projectors[f"proj{j}"](self.pool(c2[i]).flatten(1)) for i,j in enumerate(self.feats_idx) ]

            if self.use_pred:
                c1p = [ self.predictors[f"pred{j}"](c1z[i]) for i,j in enumerate(self.feats_idx) ]
                c2p = [ self.predictors[f"pred{j}"](c2z[i]) for i,j in enumerate(self.feats_idx) ]

        return p1, p2, z1.detach(), z2.detach(), (c1,c2,c1z,c2z,c1p,c2p)

class SimSiam3(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super().__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs``
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(nn.Linear(prev_dim*2, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(prev_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

        self.fuser = base_encoder(num_classes=dim, zero_init_residual=True)
        self.fuser.fc = nn.Identity()
        for param in self.fuser.parameters():
            param.requires_grad = False  # not update by gradient

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        f1 = torch.cat([self.encoder(x1),self.fuser(x1)], -1) # Nx2C
        f2 = torch.cat([self.encoder(x2),self.fuser(x2)], -1) # Nx2C

        z1 = self.projector(f1) # NxC
        z2 = self.projector(f2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()

class SimSiam4(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super().__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs``
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(prev_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

        self.fuser = base_encoder(num_classes=dim, zero_init_residual=True)
        self.fuser.fc = nn.Identity()
        for param in self.fuser.parameters():
            param.requires_grad = False  # not update by gradient

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        f1 = self.encoder(x1) + self.fuser(x1) # NxC
        f2 = self.encoder(x2) + self.fuser(x2) # NxC

        z1 = self.projector(f1) # NxC
        z2 = self.projector(f2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()


class SimSiam5(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512, pred_lam=False):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super().__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs``
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(prev_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

        self.pred_lam = pred_lam
        if pred_lam:
            self.lam_predictor = nn.Linear(prev_dim, 1)

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        f1 = self.encoder(x1) # NxC
        f2 = self.encoder(x2) # NxC

        z1 = self.projector(f1) # NxC
        z2 = self.projector(f2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        if self.pred_lam:
            lams = torch.cat([self.lam_predictor(f1), self.lam_predictor(f2)], 0)
            return p1, p2, z1.detach(), z2.detach(), lams
        else:
            return p1, p2, z1.detach(), z2.detach()


class SimSiam6(nn.Module):
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        super().__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Identity()

        # build a 3-layer projector
        self.projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(prev_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer

        prev_dim2 = prev_dim*9
        self.projector2 = nn.Sequential(nn.Linear(prev_dim2, prev_dim2, bias=False),
                                        nn.BatchNorm1d(prev_dim2),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim2, prev_dim2, bias=False),
                                        nn.BatchNorm1d(prev_dim2),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(prev_dim2, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

        self.predictor2 = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, images):

        # compute features for one view
        z1 = self.projector(self.encoder(images[0])) # BxC
        z2 = self.projector(self.encoder(images[1])) # BxC
        p1 = self.predictor(z1) # BxC
        p2 = self.predictor(z2) # BxC

        B, N, _,_,_= images[2].shape
        f3 = self.encoder(images[2].flatten(0, 1)).reshape(B, N, -1).flatten(1) # BxNC
        f4 = self.encoder(images[3].flatten(0, 1)).reshape(B, N, -1).flatten(1) # BxNC
        z3 = self.projector2(f3) # BxNC
        z4 = self.projector2(f4) # BxNC
        p3 = self.predictor2(z3) # NxC
        p4 = self.predictor2(z4) # NxC

        return p1, p2, p3, p4, z1.detach(), z2.detach(), z3.detach(), z4.detach()


class ContraCluster(nn.Module):
    def __init__(self, base_encoder, nmb_prototypes, dim=2048, pred_dim=512):
        super().__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(
            normalize=True,
            hidden_mlp=2048,
            output_dim=128,
            nmb_prototypes=nmb_prototypes
        )

        # build a 3-layer projector
        prev_dim = self.encoder.num_out_filters
        self.projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(prev_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x):
        idx_crops, feats, (emb, output) = self.encoder(x)

        z = self.projector(feats)
        z1, z2 = torch.tensor_split(z, 2)

        p = self.predictor(z)
        p1, p2 = torch.tensor_split(p, 2)

        return p1, p2, z1.detach(), z2.detach(), emb.detach(), output


class ContraCluster2(nn.Module):
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        super().__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(
            normalize=True,
            hidden_mlp=2048,
            output_dim=128,
            nmb_prototypes=0
        )

        # build a 3-layer projector
        prev_dim = self.encoder.num_out_filters
        self.projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(prev_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x, valid_idx=None):
        if valid_idx is None:
            valid_idx = torch.arange(x[0].shape[0])

        idx_crops, feats, emb = self.encoder(x)

        f1, f2 = torch.tensor_split(feats, 2)
        feats = torch.cat([f1[valid_idx], f2[valid_idx]], dim=0)

        z = self.projector(feats)
        z1, z2 = torch.tensor_split(z, 2)

        p = self.predictor(z)
        p1, p2 = torch.tensor_split(p, 2)

        return p1, p2, z1.detach(), z2.detach(), emb.detach()

class ClusterHead(nn.Module):
    def __init__(self):
        super().__init__()

        num_out_filters = 512
        hidden_mlp = 2048
        output_dim=128
        nmb_prototypes = [3000]

        self.projection_head = nn.Sequential(
            # nn.Linear(num_out_filters*2, hidden_mlp),
            nn.Linear(num_out_filters, hidden_mlp),
            nn.BatchNorm1d(hidden_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_mlp, output_dim),
        )

        self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)

    def forward(self, x):
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        
        return x, self.prototypes(x)

class ContraCluster3(nn.Module):
    def __init__(self, base_encoder, nmb_prototypes, dim=2048, pred_dim=512):
        super().__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder()
        self.cluster_head = ClusterHead()
        self.fuser = base_encoder()

        # build a 3-layer projector
        prev_dim = self.encoder.num_out_filters
        self.projector = nn.Sequential(nn.Linear(prev_dim*2, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(prev_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

        for param in self.fuser.parameters():
            param.requires_grad = False  # not update by gradient

    def forward(self, x):
        _, f1, _ = self.encoder(x)
        _, f2, _ = self.fuser(x)
        feats = torch.cat([f1, f2], dim=-1)

        # emb, output = self.cluster_head(feats)
        emb, output = self.cluster_head(f1)

        z = self.projector(feats)
        z1, z2 = torch.tensor_split(z, 2)

        p = self.predictor(z)
        p1, p2 = torch.tensor_split(p, 2)

        return p1, p2, z1.detach(), z2.detach(), emb.detach(), output


class MSModel(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, scale, dim=2048, pred_dim=512, use_clr=True):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super().__init__()

        self.K = int(scale**2)
        self.use_clr = use_clr

        # create encoders
        self.context_encoder = base_encoder(zero_init_residual=True)
        self.target_encoder = base_encoder(zero_init_residual=True)
        prev_dim = self.context_encoder.fc.weight.shape[1]
        self.context_encoder.fc = nn.Identity()
        self.target_encoder.fc = nn.Identity()

        # build a 3-layer projector
        self.context_projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                                nn.BatchNorm1d(prev_dim),
                                                nn.ReLU(inplace=True), # first layer
                                                nn.Linear(prev_dim, prev_dim, bias=False),
                                                nn.BatchNorm1d(prev_dim),
                                                nn.ReLU(inplace=True), # second layer
                                                nn.Linear(prev_dim, dim, bias=False),
                                                nn.BatchNorm1d(dim, affine=False)) # output layer

        self.target_projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(prev_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer

        fuser_dim = prev_dim * (self.K+1)
        self.fuser_projector = nn.Sequential(nn.Linear(fuser_dim, fuser_dim, bias=False),
                                        nn.BatchNorm1d(fuser_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(fuser_dim, fuser_dim, bias=False),
                                        nn.BatchNorm1d(fuser_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(fuser_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer



        # build a 2-layer predictor
        self.context_predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                                nn.BatchNorm1d(pred_dim),
                                                nn.ReLU(inplace=True), # hidden layer
                                                nn.Linear(pred_dim, dim)) # output layer

        self.target_predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                                nn.BatchNorm1d(pred_dim),
                                                nn.ReLU(inplace=True), # hidden layer
                                                nn.Linear(pred_dim, dim)) # output layer
        
        self.fuser_predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                                nn.BatchNorm1d(pred_dim),
                                                nn.ReLU(inplace=True), # hidden layer
                                                nn.Linear(pred_dim, dim)) # output layer

        # build fuser classifier for jigsaw
        if self.use_clr:
            self.fuser_clr = nn.Sequential(nn.Linear(fuser_dim, pred_dim, bias=False),
                                                    nn.BatchNorm1d(pred_dim),
                                                    nn.ReLU(inplace=True), # hidden layer
                                                    nn.Linear(pred_dim, self.K))

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
        context_f1, context_f2 = self.context_encoder(x1[0]), self.context_encoder(x2[0]) # BxC
        target_f1, target_f2 = self.target_encoder(x1[1]), self.target_encoder(x2[1]) # BKxC

        # Combine context and target features
        ms_f1, ms_f2 = None, None
        target_f1_split, target_f2_split = target_f1.reshape(B, self.K, -1), target_f2.reshape(B, self.K, -1) # BxKxC

        ### cat context with each of target feats
        # for i in range(len(context_f1)):
        #     context_f1_group, context_f2_group = context_f1[i].repeat(self.K, 1), context_f2[i].repeat(self.K, 1)
        #     ms_f1.append(torch.cat((context_f1_group, target_f1_split[i]), dim=1))
        #     ms_f2.append(torch.cat((context_f2_group, target_f2_split[i]), dim=1))
        # ms_f1, ms_f2 = torch.stack(ms_f1), torch.stack(ms_f2)

        ### cat context with grouped target feats
        ms_f1, ms_f2 = torch.cat((context_f1, target_f1_split.flatten(1)), dim=1), torch.cat((context_f2, target_f2_split.flatten(1)), dim=1) # Bx(K+1)C
        assert ms_f1.shape[0] == ms_f2.shape[0] == B

        # reorder target patches
        batch_idx = torch.arange(B).repeat(self.K, 1).t()
        assert batch_idx.shape == jigsaw_idx[0].shape == jigsaw_idx[1].shape, f"batch_idx and jigsaw_idx shape mismatch {batch_idx.shape} {jigsaw_idx[0].shape} {jigsaw_idx[1].shape}"
        target_f1, target_f2 = target_f1_split[batch_idx, jigsaw_idx[0], :].flatten(0, 1), target_f2_split[batch_idx, jigsaw_idx[1], :].flatten(0, 1) # BKxC

        # Features from projector
        context_z1, context_z2 = self.context_projector(context_f1), self.context_projector(context_f2) # BxC
        target_z1, target_z2 = self.target_projector(target_f1), self.target_projector(target_f2) # BKxC

        # Features from predictor 
        context_p1, context_p2 = self.context_predictor(context_z1), self.context_predictor(context_z2) # BxC
        target_p1, target_p2 = self.target_predictor(target_z1), self.target_predictor(target_z2) # BKxC

        # fuser contrastive learning
        ms_z1, ms_z2 = self.fuser_projector(ms_f1), self.fuser_projector(ms_f2)
        ms_p1, ms_p2 = self.fuser_predictor(ms_z1), self.fuser_predictor(ms_z2)

        return (
            (context_p1, context_p2, context_z1.detach(), context_z2.detach()),
            (target_p1, target_p2, target_z1.detach(), target_z2.detach()),
            (ms_p1, ms_p2, ms_z1.detach(), ms_z2.detach()),
        )


class MSModel2(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, scale, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super().__init__()

        self.K = int(scale**2)

        # create encoders
        self.context_encoder = base_encoder(zero_init_residual=True)
        self.target_encoder = base_encoder(zero_init_residual=True)
        prev_dim = self.context_encoder.fc.weight.shape[1]
        self.context_encoder.fc = nn.Identity()
        self.target_encoder.fc = nn.Identity()

        # build a 3-layer projector
        fuser_dim = prev_dim * (self.K+1)
        self.fuser_projector = nn.Sequential(nn.Linear(fuser_dim, fuser_dim, bias=False),
                                        nn.BatchNorm1d(fuser_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(fuser_dim, fuser_dim, bias=False),
                                        nn.BatchNorm1d(fuser_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(fuser_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer



        # build a 2-layer predictor
        self.fuser_predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                                nn.BatchNorm1d(pred_dim),
                                                nn.ReLU(inplace=True), # hidden layer
                                                nn.Linear(pred_dim, dim)) # output layer

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
        context_f1, context_f2 = self.context_encoder(x1[0]), self.context_encoder(x2[0]) # BxC
        target_f1, target_f2 = self.target_encoder(x1[1]), self.target_encoder(x2[1]) # BKxC

        # Combine context and target features
        ms_f1, ms_f2 = [], []
        # target_f1_split, target_f2_split = target_f1.reshape(B, self.K, -1), target_f2.reshape(B, self.K, -1) # BxKxC
        target_f1_split, target_f2_split = target_f1.reshape(B, -1), target_f2.reshape(B, -1) # BxKC
        
        # Jigsaw target patches
        # batch_idx = torch.arange(B).repeat(self.K, 1).t()
        # assert batch_idx.shape == jigsaw_idx[0].shape == jigsaw_idx[1].shape, f"batch_idx and jigsaw_idx shape mismatch {batch_idx.shape} {jigsaw_idx[0].shape} {jigsaw_idx[1].shape}"
        # target_f1_jigsaw, target_f2_jigsaw = target_f1_split[batch_idx, jigsaw_idx[0], :].flatten(1), target_f2_split[batch_idx, jigsaw_idx[1], :].flatten(1) # BxKC


        ### cat context with each of target feats
        # for i in range(len(context_f1)):
        #     context_f1_group, context_f2_group = context_f1[i].repeat(self.K, 1), context_f2[i].repeat(self.K, 1)
        #     ms_f1.append(torch.cat((context_f1_group, target_f1_split[i]), dim=1))
        #     ms_f2.append(torch.cat((context_f2_group, target_f2_split[i]), dim=1))
        # ms_f1, ms_f2 = torch.stack(ms_f1), torch.stack(ms_f2)

        ### cat context with grouped target feats
        # ms_f1, ms_f2 = torch.cat((context_f1, target_f1_jigsaw), dim=1), torch.cat((context_f2, target_f2_jigsaw), dim=1) # Bx(K+1)C
        ms_f1, ms_f2 = torch.cat((context_f1, target_f1_split), dim=1), torch.cat((context_f2, target_f2_split), dim=1) # Bx(K+1)C
        assert ms_f1.shape[0] == ms_f2.shape[0] == B

        # fuser contrastive learning
        ms_z1, ms_z2 = self.fuser_projector(ms_f1), self.fuser_projector(ms_f2)
        ms_p1, ms_p2 = self.fuser_predictor(ms_z1), self.fuser_predictor(ms_z2)


        return (
            (ms_p1, ms_p2, ms_z1.detach(), ms_z2.detach()),
        )


class MSModel4(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, scale, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super().__init__()

        self.K = int(scale**2)

        # create encoders
        self.context_encoder = base_encoder(zero_init_residual=True)
        self.target_encoder = base_encoder(zero_init_residual=True)
        prev_dim = self.context_encoder.fc.weight.shape[1]
        self.context_encoder.fc = nn.Identity()
        self.target_encoder.fc = nn.Identity()

        # build a 3-layer projector
        self.context_projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                                nn.BatchNorm1d(prev_dim),
                                                nn.ReLU(inplace=True), # first layer
                                                nn.Linear(prev_dim, prev_dim, bias=False),
                                                nn.BatchNorm1d(prev_dim),
                                                nn.ReLU(inplace=True), # second layer
                                                nn.Linear(prev_dim, dim, bias=False),
                                                nn.BatchNorm1d(dim, affine=False)) # output layer

        self.target_projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(prev_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer


        # build a 2-layer predictor
        self.context_predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                                nn.BatchNorm1d(pred_dim),
                                                nn.ReLU(inplace=True), # hidden layer
                                                nn.Linear(pred_dim, dim)) # output layer

        self.target_predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                                nn.BatchNorm1d(pred_dim),
                                                nn.ReLU(inplace=True), # hidden layer
                                                nn.Linear(pred_dim, dim)) # output layer


    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images [context_images, target_images]
            x2: second views of images [context_images, target_images]
        Output:
            p1, p2, z1, z2: predictors and targets of the network
        """
        B = x1[0].shape[0]

        # Features from encoder
        context_f1, context_f2 = self.context_encoder(x1[0]), self.context_encoder(x2[0]) # BxC
        target_f1, target_f2 = self.target_encoder(x1[1]), self.target_encoder(x2[1]) # BKxC

        # Features from projector
        context_z1, context_z2 = self.context_projector(context_f1), self.context_projector(context_f2) # BxC
        target_z1, target_z2 = self.target_projector(target_f1), self.target_projector(target_f2) # BKxC

        # Features from predictor 
        context_p1, context_p2 = self.context_predictor(context_z1), self.context_predictor(context_z2) # BxC
        target_p1, target_p2 = self.target_predictor(target_z1), self.target_predictor(target_z2) # BKxC

        return (
            (context_p1, context_p2, context_z1.detach(), context_z2.detach()),
            (target_p1, target_p2, target_z1.detach(), target_z2.detach()),
        )


class MSModel5(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, scale, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super().__init__()

        self.K = int(scale**2)//2

        # create encoders
        self.context_encoder = base_encoder(zero_init_residual=True)
        self.target_encoder = base_encoder(zero_init_residual=True)
        prev_dim = self.context_encoder.fc.weight.shape[1]
        self.context_encoder.fc = nn.Identity()
        self.target_encoder.fc = nn.Identity()

        # build a 3-layer projector
        fuser_dim = prev_dim * (self.K+1)
        self.fuser_projector = nn.Sequential(nn.Linear(fuser_dim, fuser_dim, bias=False),
                                        nn.BatchNorm1d(fuser_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(fuser_dim, fuser_dim, bias=False),
                                        nn.BatchNorm1d(fuser_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(fuser_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer



        # build a 2-layer predictor
        self.fuser_predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                                nn.BatchNorm1d(pred_dim),
                                                nn.ReLU(inplace=True), # hidden layer
                                                nn.Linear(pred_dim, dim)) # output layer

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
        context_f1, context_f2 = self.context_encoder(x1[0]), self.context_encoder(x2[0]) # BxC
        target_f1, target_f2 = self.target_encoder(x1[1]), self.target_encoder(x2[1]) # BKxC

        # Combine context and target features
        ms_f1, ms_f2 = [], []
        # target_f1_split, target_f2_split = target_f1.reshape(B, self.K, -1), target_f2.reshape(B, self.K, -1) # BxKxC
        target_f1_split, target_f2_split = target_f1.reshape(B, -1), target_f2.reshape(B, -1) # BxKC

        # Jigsaw target patches
        # batch_idx = torch.arange(B).repeat(self.K, 1).t()
        # assert batch_idx.shape == jigsaw_idx[0].shape == jigsaw_idx[1].shape, f"batch_idx and jigsaw_idx shape mismatch {batch_idx.shape} {jigsaw_idx[0].shape} {jigsaw_idx[1].shape}"
        # target_f1_jigsaw, target_f2_jigsaw = target_f1_split[batch_idx, jigsaw_idx[0], :].flatten(1), target_f2_split[batch_idx, jigsaw_idx[1], :].flatten(1) # BxKC

        ### cat context with grouped target feats
        # ms_f1, ms_f2 = torch.cat((context_f1, target_f1_jigsaw), dim=1), torch.cat((context_f2, target_f2_jigsaw), dim=1) # Bx(K+1)C
        ms_f1, ms_f2 = torch.cat((context_f1, target_f1_split), dim=1), torch.cat((context_f2, target_f2_split), dim=1) # Bx(K+1)C
        assert ms_f1.shape[0] == ms_f2.shape[0] == B

        # fuser contrastive learning
        ms_z1, ms_z2 = self.fuser_projector(ms_f1), self.fuser_projector(ms_f2)
        ms_p1, ms_p2 = self.fuser_predictor(ms_z1), self.fuser_predictor(ms_z2)

        return (
            (ms_p1, ms_p2, ms_z1.detach(), ms_z2.detach()),
        )


class MSModel5_2(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, scale, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super().__init__()

        self.K = int(scale**2)

        # create encoders
        self.context_encoder = base_encoder(zero_init_residual=True)
        self.target_encoder = base_encoder(zero_init_residual=True)
        prev_dim = self.context_encoder.fc.weight.shape[1]
        self.context_encoder.fc = nn.Identity()
        self.target_encoder.fc = nn.Identity()

        # build a 3-layer projector
        self.context_projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(prev_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer

        self.target_projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(prev_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer

        fuser_dim = prev_dim * (self.K//2+1)
        self.fuser_projector = nn.Sequential(nn.Linear(fuser_dim, fuser_dim, bias=False),
                                        nn.BatchNorm1d(fuser_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(fuser_dim, fuser_dim, bias=False),
                                        nn.BatchNorm1d(fuser_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(fuser_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer



        # build a 2-layer predictor
        self.context_predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                                nn.BatchNorm1d(pred_dim),
                                                nn.ReLU(inplace=True), # hidden layer
                                                nn.Linear(pred_dim, dim)) # output layer

        self.target_predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                                nn.BatchNorm1d(pred_dim),
                                                nn.ReLU(inplace=True), # hidden layer
                                                nn.Linear(pred_dim, dim)) # output layer
                                                
        self.fuser_predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                                nn.BatchNorm1d(pred_dim),
                                                nn.ReLU(inplace=True), # hidden layer
                                                nn.Linear(pred_dim, dim)) # output layer

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
        context_f1, context_f2 = self.context_encoder(x1[0]), self.context_encoder(x2[0]) # BxC
        target_f1, target_f2 = self.target_encoder(x1[1]), self.target_encoder(x2[1]) # BKxC
        target_f1_split, target_f2_split = target_f1.reshape(B, self.K, -1), target_f2.reshape(B, self.K, -1) # BxKxC

        # reorder target patches
        batch_idx = torch.arange(B).repeat(self.K, 1).t()
        assert batch_idx.shape == jigsaw_idx[0].shape == jigsaw_idx[1].shape, f"batch_idx and jigsaw_idx shape mismatch {batch_idx.shape} {jigsaw_idx[0].shape} {jigsaw_idx[1].shape}"
        target_f1, target_f2 = target_f1_split[batch_idx, jigsaw_idx[0], :].flatten(0, 1), target_f2_split[batch_idx, jigsaw_idx[1], :].flatten(0, 1) # BKxC

        # Features from projector
        context_z1, context_z2 = self.context_projector(context_f1), self.context_projector(context_f2) # BxC
        target_z1, target_z2 = self.target_projector(target_f1), self.target_projector(target_f2) # BKxC

        # Features from predictor 
        context_p1, context_p2 = self.context_predictor(context_z1), self.context_predictor(context_z2) # BxC
        target_p1, target_p2 = self.target_predictor(target_z1), self.target_predictor(target_z2) # BKxC

        # Combine context and target features
        ms_f1, ms_f2 = torch.cat((context_f1, target_f1_split[:, :8, :].flatten(1)), dim=1), torch.cat((context_f2, target_f2_split[:, :8, :].flatten(1)), dim=1) # Bx(K+1)C
        assert ms_f1.shape[0] == ms_f2.shape[0] == B

        # fuser contrastive learning
        ms_z1, ms_z2 = self.fuser_projector(ms_f1), self.fuser_projector(ms_f2)
        ms_p1, ms_p2 = self.fuser_predictor(ms_z1), self.fuser_predictor(ms_z2)

        return (
            (context_p1, context_p2, context_z1.detach(), context_z2.detach()),
            (target_p1, target_p2, target_z1.detach(), target_z2.detach()),
            (ms_p1, ms_p2, ms_z1.detach(), ms_z2.detach()),
        )


class MSModel5_3(MSModel5_2):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, scale, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super().__init__(base_encoder, scale, dim, pred_dim)

        # create encoders
        self.context_encoder = base_encoder(zero_init_residual=True, pretrained=True)
        self.target_encoder = base_encoder(zero_init_residual=True, pretrained=True)
        self.context_encoder.fc = nn.Identity()
        self.target_encoder.fc = nn.Identity()


class MSModel5_4(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, scale, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super().__init__()

        self.K = int(scale**2)
        
        # create encoders
        self.context_encoder = base_encoder(zero_init_residual=True, pretrained=True)
        self.target_encoder = base_encoder(zero_init_residual=True, pretrained=True)
        prev_dim = self.context_encoder.fc.weight.shape[1]
        self.context_encoder.fc = nn.Identity()
        self.target_encoder.fc = nn.Identity()

        fuser_dim = prev_dim * (self.K//2+1)

        # build a 3-layer projector
        self.context_projector = make_projector(prev_dim, prev_dim)
        self.target_projector = make_projector(prev_dim, prev_dim)  
        self.fuser_projector = make_projector(fuser_dim, fuser_dim)

        # build a 2-layer predictor
        self.context_predictor = make_predictor(prev_dim, prev_dim//4)
        self.target_predictor = make_predictor(prev_dim, prev_dim//4)                              
        self.fuser_predictor = make_predictor(fuser_dim, fuser_dim//4)

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
        context_f1, context_f2 = self.context_encoder(x1[0]), self.context_encoder(x2[0]) # BxC
        target_f1, target_f2 = self.target_encoder(x1[1]), self.target_encoder(x2[1]) # BKxC
        target_f1_split, target_f2_split = target_f1.reshape(B, self.K, -1), target_f2.reshape(B, self.K, -1) # BxKxC

        # reorder target patches
        batch_idx = torch.arange(B).repeat(self.K, 1).t()
        assert batch_idx.shape == jigsaw_idx[0].shape == jigsaw_idx[1].shape, f"batch_idx and jigsaw_idx shape mismatch {batch_idx.shape} {jigsaw_idx[0].shape} {jigsaw_idx[1].shape}"
        target_f1, target_f2 = target_f1_split[batch_idx, jigsaw_idx[0], :].flatten(0, 1), target_f2_split[batch_idx, jigsaw_idx[1], :].flatten(0, 1) # BKxC

        # Features from projector
        context_z1, context_z2 = self.context_projector(context_f1), self.context_projector(context_f2) # BxC
        target_z1, target_z2 = self.target_projector(target_f1), self.target_projector(target_f2) # BKxC

        # Features from predictor 
        context_p1, context_p2 = self.context_predictor(context_z1), self.context_predictor(context_z2) # BxC
        target_p1, target_p2 = self.target_predictor(target_z1), self.target_predictor(target_z2) # BKxC

        # Combine context and target features
        ms_f1, ms_f2 = torch.cat((context_f1, target_f1_split[:, :8, :].flatten(1)), dim=1), torch.cat((context_f2, target_f2_split[:, :8, :].flatten(1)), dim=1) # Bx(K+1)C
        assert ms_f1.shape[0] == ms_f2.shape[0] == B

        # fuser contrastive learning
        ms_z1, ms_z2 = self.fuser_projector(ms_f1), self.fuser_projector(ms_f2)
        ms_p1, ms_p2 = self.fuser_predictor(ms_z1), self.fuser_predictor(ms_z2)

        return (
            (context_p1, context_p2, context_z1.detach(), context_z2.detach()),
            (target_p1, target_p2, target_z1.detach(), target_z2.detach()),
            (ms_p1, ms_p2, ms_z1.detach(), ms_z2.detach()),
        )


class MSModel5_5(MSModel5_4):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, scale, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super().__init__(base_encoder, scale)
        
        # create encoders
        self.context_encoder = base_encoder(zero_init_residual=True)
        self.target_encoder = base_encoder(zero_init_residual=True)
        self.context_encoder.fc = nn.Identity()
        self.target_encoder.fc = nn.Identity()


class MSModel6(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, scale, dim=2048, pred_dim=512, cluster_out=5, overcluster_out=20):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super().__init__()

        self.K = int(scale**2)

        # create encoders
        self.context_encoder = base_encoder(zero_init_residual=True)
        self.target_encoder = base_encoder(zero_init_residual=True)
        prev_dim = self.context_encoder.fc.weight.shape[1]
        self.context_encoder.fc = nn.Identity()
        self.target_encoder.fc = nn.Identity()

        # build a 3-layer projector
        self.context_projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(prev_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer

        self.target_projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(prev_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer

        fuser_dim = prev_dim * (self.K//2+1)
        self.fuser_projector = nn.Sequential(nn.Linear(fuser_dim, fuser_dim, bias=False),
                                        nn.BatchNorm1d(fuser_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(fuser_dim, fuser_dim, bias=False),
                                        nn.BatchNorm1d(fuser_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(fuser_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer



        # build a 2-layer predictor
        self.context_predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                                nn.BatchNorm1d(pred_dim),
                                                nn.ReLU(inplace=True), # hidden layer
                                                nn.Linear(pred_dim, dim)) # output layer

        self.target_predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                                nn.BatchNorm1d(pred_dim),
                                                nn.ReLU(inplace=True), # hidden layer
                                                nn.Linear(pred_dim, dim)) # output layer
                                                
        self.fuser_predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                                nn.BatchNorm1d(pred_dim),
                                                nn.ReLU(inplace=True), # hidden layer
                                                nn.Linear(pred_dim, dim)) # output layer

        # build cluster heads
        cluster_dim = prev_dim * (self.K+1)
        self.cluster_head = nn.Sequential(nn.Linear(cluster_dim, cluster_out), nn.Softmax(dim=1))
        self.overcluster_head = nn.Sequential(nn.Linear(cluster_dim, overcluster_out), nn.Softmax(dim=1))

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
        context_f1, context_f2 = self.context_encoder(x1[0]), self.context_encoder(x2[0]) # BxC
        target_f1, target_f2 = self.target_encoder(x1[1]), self.target_encoder(x2[1]) # BKxC
        target_f1_split, target_f2_split = target_f1.reshape(B, self.K, -1), target_f2.reshape(B, self.K, -1) # BxKxC

        # reorder target patches
        batch_idx = torch.arange(B).repeat(self.K, 1).t()
        assert batch_idx.shape == jigsaw_idx[0].shape == jigsaw_idx[1].shape, f"batch_idx and jigsaw_idx shape mismatch {batch_idx.shape} {jigsaw_idx[0].shape} {jigsaw_idx[1].shape}"
        target_f1, target_f2 = target_f1_split[batch_idx, jigsaw_idx[0], :].flatten(0, 1), target_f2_split[batch_idx, jigsaw_idx[1], :].flatten(0, 1) # BKxC

        # Features from projector
        context_z1, context_z2 = self.context_projector(context_f1), self.context_projector(context_f2) # BxC
        target_z1, target_z2 = self.target_projector(target_f1), self.target_projector(target_f2) # BKxC

        # Features from predictor 
        context_p1, context_p2 = self.context_predictor(context_z1), self.context_predictor(context_z2) # BxC
        target_p1, target_p2 = self.target_predictor(target_z1), self.target_predictor(target_z2) # BKxC

        # Combine context and target features
        ms_f1, ms_f2 = torch.cat((context_f1, target_f1_split[:, :8, :].flatten(1)), dim=1), torch.cat((context_f2, target_f2_split[:, :8, :].flatten(1)), dim=1) # Bx(K+1)C
        assert ms_f1.shape[0] == ms_f2.shape[0] == B

        # fuser contrastive learning
        ms_z1, ms_z2 = self.fuser_projector(ms_f1), self.fuser_projector(ms_f2)
        ms_p1, ms_p2 = self.fuser_predictor(ms_z1), self.fuser_predictor(ms_z2)

        # cluster learning
        # cluster_z1, cluster_z2 = self.cluster_head(ms_f1), self.cluster_head(ms_f2)
        # overcluster_z1, overcluster_z2 = self.overcluster_head(ms_f1), self.overcluster_head(ms_f2)

        cluster_f1, cluster_f2 = torch.cat((context_f1, target_f1_split.flatten(1)), dim=1), torch.cat((context_f2, target_f2_split.flatten(1)), dim=1) # Bx(K+1)C
        cluster_z1, cluster_z2 = self.cluster_head(cluster_f1), self.cluster_head(cluster_f2)
        overcluster_z1, overcluster_z2 = self.overcluster_head(cluster_f1), self.overcluster_head(cluster_f2)

        return (
            (context_p1, context_p2, context_z1.detach(), context_z2.detach()),
            (target_p1, target_p2, target_z1.detach(), target_z2.detach()),
            (ms_p1, ms_p2, ms_z1.detach(), ms_z2.detach()),
            (cluster_z1, cluster_z2, overcluster_z1, overcluster_z2)
        )


class MSClusterModel(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, scale, nmb_prototypes, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super().__init__()

        self.K = int(scale**2)
        self.nmb_prototypes = nmb_prototypes

        # create encoders
        self.context_encoder = base_encoder(zero_init_residual=True)
        self.target_encoder = base_encoder(zero_init_residual=True)
        prev_dim = self.context_encoder.fc.weight.shape[1]
        self.context_encoder.fc = nn.Identity()
        self.target_encoder.fc = nn.Identity()

        # build a 3-layer projector
        self.context_projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                                nn.BatchNorm1d(prev_dim),
                                                nn.ReLU(inplace=True), # first layer
                                                nn.Linear(prev_dim, prev_dim, bias=False),
                                                nn.BatchNorm1d(prev_dim),
                                                nn.ReLU(inplace=True), # second layer
                                                nn.Linear(prev_dim, dim, bias=False),
                                                nn.BatchNorm1d(dim, affine=False)) # output layer

        self.target_projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(prev_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer

        fuser_dim = prev_dim * (self.K+1)
        self.fuser_projector = nn.Sequential(nn.Linear(fuser_dim, fuser_dim, bias=False),
                                        nn.BatchNorm1d(fuser_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(fuser_dim, fuser_dim, bias=False),
                                        nn.BatchNorm1d(fuser_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(fuser_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer

        # build a 2-layer predictor
        self.context_predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                                nn.BatchNorm1d(pred_dim),
                                                nn.ReLU(inplace=True), # hidden layer
                                                nn.Linear(pred_dim, dim)) # output layer

        self.target_predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                                nn.BatchNorm1d(pred_dim),
                                                nn.ReLU(inplace=True), # hidden layer
                                                nn.Linear(pred_dim, dim)) # output layer
        
        self.fuser_predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                                nn.BatchNorm1d(pred_dim),
                                                nn.ReLU(inplace=True), # hidden layer
                                                nn.Linear(pred_dim, dim)) # output layer

        # Cluster head
        self.cluster_projector = nn.Sequential(
                                    nn.Linear(fuser_dim, 2048, bias=False),
                                    nn.BatchNorm1d(2048),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(2048, 128),
                                )
        self.cluster_prototype = nn.Linear(128, self.nmb_prototypes[0], bias=False)


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
        context_f1, context_f2 = self.context_encoder(x1[0]), self.context_encoder(x2[0]) # BxC
        target_f1, target_f2 = self.target_encoder(x1[1]), self.target_encoder(x2[1]) # BKxC

        # Features from projector
        context_z1, context_z2 = self.context_projector(context_f1), self.context_projector(context_f2) # BxC
        target_z1, target_z2 = self.target_projector(target_f1), self.target_projector(target_f2) # BKxC

        # Features from predictor 
        context_p1, context_p2 = self.context_predictor(context_z1), self.context_predictor(context_z2) # BxC
        target_p1, target_p2 = self.target_predictor(target_z1), self.target_predictor(target_z2) # BKxC

        # Combine context and target features
        ms_f1, ms_f2 = [], []
        target_f1_split, target_f2_split = target_f1.reshape(B, self.K, -1), target_f2.reshape(B, self.K, -1) # BxKxC
        
        # Jigsaw target patches
        batch_idx = torch.arange(B).repeat(self.K, 1).t()
        assert batch_idx.shape == jigsaw_idx[0].shape == jigsaw_idx[1].shape, f"batch_idx and jigsaw_idx shape mismatch {batch_idx.shape} {jigsaw_idx[0].shape} {jigsaw_idx[1].shape}"
        target_f1_jigsaw, target_f2_jigsaw = target_f1_split[batch_idx, jigsaw_idx[0], :].flatten(1), target_f2_split[batch_idx, jigsaw_idx[1], :].flatten(1) # BxKC


        ### cat context with each of target feats
        # for i in range(len(context_f1)):
        #     context_f1_group, context_f2_group = context_f1[i].repeat(self.K, 1), context_f2[i].repeat(self.K, 1)
        #     ms_f1.append(torch.cat((context_f1_group, target_f1_split[i]), dim=1))
        #     ms_f2.append(torch.cat((context_f2_group, target_f2_split[i]), dim=1))
        # ms_f1, ms_f2 = torch.stack(ms_f1), torch.stack(ms_f2)

        ### cat context with grouped target feats
        ms_f1, ms_f2 = torch.cat((context_f1, target_f1_jigsaw), dim=1), torch.cat((context_f2, target_f2_jigsaw), dim=1) # Bx(K+1)C
        assert ms_f1.shape[0] == ms_f2.shape[0] == B

        # fuser contrastive learning
        ms_z1, ms_z2 = self.fuser_projector(ms_f1), self.fuser_projector(ms_f2)
        ms_p1, ms_p2 = self.fuser_predictor(ms_z1), self.fuser_predictor(ms_z2)

        # Cluster head
        output1 = self.cluster_projector(torch.cat((context_f1, target_f1.reshape(B, -1)), dim=1))
        output2 = self.cluster_projector(torch.cat((context_f2, target_f2.reshape(B, -1)), dim=1))
        output = torch.cat((output1, output2), dim=0)

        emb = nn.functional.normalize(output, p=2, dim=1)

        prototype = self.cluster_prototype(emb)

        return (
            (context_p1, context_p2, context_z1.detach(), context_z2.detach()),
            (target_p1, target_p2, target_z1.detach(), target_z2.detach()),
            (ms_p1, ms_p2, ms_z1.detach(), ms_z2.detach()),
            emb.detach(),
            prototype
        )


class MSClusterModel2(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, scale, nmb_prototypes, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super().__init__()

        self.K = int(scale**2)
        self.nmb_prototypes = nmb_prototypes

        # create encoders
        self.context_encoder = base_encoder(zero_init_residual=True)
        self.target_encoder = base_encoder(zero_init_residual=True)
        prev_dim = self.context_encoder.fc.weight.shape[1]
        self.context_encoder.fc = nn.Identity()
        self.target_encoder.fc = nn.Identity()

        # build a 3-layer projector
        self.context_projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                                nn.BatchNorm1d(prev_dim),
                                                nn.ReLU(inplace=True), # first layer
                                                nn.Linear(prev_dim, prev_dim, bias=False),
                                                nn.BatchNorm1d(prev_dim),
                                                nn.ReLU(inplace=True), # second layer
                                                nn.Linear(prev_dim, dim, bias=False),
                                                nn.BatchNorm1d(dim, affine=False)) # output layer

        self.target_projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(prev_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer

        fuser_dim = prev_dim * (self.K//2+1)
        self.fuser_projector = nn.Sequential(nn.Linear(fuser_dim, fuser_dim, bias=False),
                                        nn.BatchNorm1d(fuser_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(fuser_dim, fuser_dim, bias=False),
                                        nn.BatchNorm1d(fuser_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(fuser_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer

        # build a 2-layer predictor
        self.context_predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                                nn.BatchNorm1d(pred_dim),
                                                nn.ReLU(inplace=True), # hidden layer
                                                nn.Linear(pred_dim, dim)) # output layer

        self.target_predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                                nn.BatchNorm1d(pred_dim),
                                                nn.ReLU(inplace=True), # hidden layer
                                                nn.Linear(pred_dim, dim)) # output layer
        
        self.fuser_predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                                nn.BatchNorm1d(pred_dim),
                                                nn.ReLU(inplace=True), # hidden layer
                                                nn.Linear(pred_dim, dim)) # output layer

        # Cluster head
        self.cluster_projector = nn.Sequential(
                                    # nn.Linear(prev_dim * (self.K+1), 2048, bias=False),
                                    nn.Linear(prev_dim, 2048, bias=False),
                                    nn.BatchNorm1d(2048),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(2048, 128),
                                )
        self.cluster_prototype = nn.Linear(128, self.nmb_prototypes[0], bias=False)


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
        context_f1, context_f2 = self.context_encoder(x1[0]), self.context_encoder(x2[0]) # BxC
        target_f1, target_f2 = self.target_encoder(x1[1]), self.target_encoder(x2[1]) # BKxC
        target_f1_split, target_f2_split = target_f1.reshape(B, self.K, -1), target_f2.reshape(B, self.K, -1) # BxKxC

        # reorder target patches
        batch_idx = torch.arange(B).repeat(self.K, 1).t()
        assert batch_idx.shape == jigsaw_idx[0].shape == jigsaw_idx[1].shape, f"batch_idx and jigsaw_idx shape mismatch {batch_idx.shape} {jigsaw_idx[0].shape} {jigsaw_idx[1].shape}"
        target_f1, target_f2 = target_f1_split[batch_idx, jigsaw_idx[0], :].flatten(0, 1), target_f2_split[batch_idx, jigsaw_idx[1], :].flatten(0, 1) # BKxC

        # Features from projector
        context_z1, context_z2 = self.context_projector(context_f1), self.context_projector(context_f2) # BxC
        target_z1, target_z2 = self.target_projector(target_f1), self.target_projector(target_f2) # BKxC

        # Features from predictor 
        context_p1, context_p2 = self.context_predictor(context_z1), self.context_predictor(context_z2) # BxC
        target_p1, target_p2 = self.target_predictor(target_z1), self.target_predictor(target_z2) # BKxC

        # Combine context and target features
        ms_f1, ms_f2 = torch.cat((context_f1, target_f1_split[:, :8, :].flatten(1)), dim=1), torch.cat((context_f2, target_f2_split[:, :8, :].flatten(1)), dim=1) # Bx(K+1)C
        assert ms_f1.shape[0] == ms_f2.shape[0] == B

        # fuser contrastive learning
        ms_z1, ms_z2 = self.fuser_projector(ms_f1), self.fuser_projector(ms_f2)
        ms_p1, ms_p2 = self.fuser_predictor(ms_z1), self.fuser_predictor(ms_z2)

        # Cluster head
        # style 1
        # output1 = self.cluster_projector(torch.cat((context_f1, target_f1.reshape(B, -1)), dim=1))
        # output2 = self.cluster_projector(torch.cat((context_f2, target_f2.reshape(B, -1)), dim=1))
        # output = torch.cat((output1, output2), dim=0)
        # emb = nn.functional.normalize(output, p=2, dim=1)
        # prototype = self.cluster_prototype(emb)

        # style 2
        output1 = self.cluster_projector(context_f1)
        output2 = self.cluster_projector(context_f2)
        output = torch.cat((output1, output2), dim=0)
        emb = nn.functional.normalize(output, p=2, dim=1)
        prototype = self.cluster_prototype(emb)

        return (
            (context_p1, context_p2, context_z1.detach(), context_z2.detach()),
            (target_p1, target_p2, target_z1.detach(), target_z2.detach()),
            (ms_p1, ms_p2, ms_z1.detach(), ms_z2.detach()),
            emb.detach(),
            prototype
        )


def make_projector(in_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, in_dim, bias=False),
                            nn.BatchNorm1d(in_dim),
                            nn.ReLU(inplace=True), # first layer
                            nn.Linear(in_dim, in_dim, bias=False),
                            nn.BatchNorm1d(in_dim),
                            nn.ReLU(inplace=True), # second layer
                            nn.Linear(in_dim, out_dim, bias=False),
                            nn.BatchNorm1d(out_dim, affine=False)
            ) 


def make_predictor(in_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias=False),
                            nn.BatchNorm1d(out_dim),
                            nn.ReLU(inplace=True), # hidden layer
                            nn.Linear(out_dim, in_dim) # output layer
            ) 


class MSModel7(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, scale, dim=2048, pred_dim=512, mask_ratio=0.5, use_checkpoint=False):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super().__init__()

        self.K = int(scale**2)
        self.n_keep = int(self.K * (1-mask_ratio))

        # create encoders
        self.context_encoder = base_encoder(zero_init_residual=True, pretrained=True, return_features=True)
        self.target_encoder = base_encoder(zero_init_residual=True, pretrained=True, return_features=True)
        prev_dim = self.context_encoder.fc.weight.shape[1]
        self.context_encoder.fc = nn.Identity()
        self.target_encoder.fc = nn.Identity()
   
        self.inter_dim = torch.as_tensor([64, 128, 256, 512]) # TODO: hardcoded
        self.ms_inter_dim = self.inter_dim * (self.n_keep+1)

        # build a 3-layer projector
        self.context_projector = nn.ModuleList([make_projector(d, d) for d in self.inter_dim])
        self.target_projector = nn.ModuleList([make_projector(d, d) for d in self.inter_dim])
        self.inter_projector = nn.ModuleList([make_projector(d, d) for d in self.ms_inter_dim])

        # build a 2-layer predictor
        self.context_predictor = nn.ModuleList([make_predictor(d, torch.div(d,4,rounding_mode="floor")) for d in self.inter_dim])
        self.target_predictor = nn.ModuleList([make_predictor(d, torch.div(d,4,rounding_mode="floor")) for d in self.inter_dim])                
        self.inter_predictor = nn.ModuleList([make_predictor(d, torch.div(d,4,rounding_mode="floor")) for d in self.ms_inter_dim])

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
        self.context_encoder.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.target_encoder.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

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
        context_f1, context_f2 = self.context_encoder(x1[0]), self.context_encoder(x2[0]) # BxC
        target_f1, target_f2 = self.target_encoder(x1[1]), self.target_encoder(x2[1]) # BKxC
        
        target_f1_split = tuple(i.reshape(B, self.K, -1) for i in target_f1) # BxKxC
        target_f2_split = tuple(i.reshape(B, self.K, -1) for i in target_f2) # BxKxC

        # reorder target patches
        batch_idx = torch.arange(B).repeat(self.K, 1).t()
        assert batch_idx.shape == jigsaw_idx[0].shape == jigsaw_idx[1].shape
        target_f1_sort = tuple(i[batch_idx, jigsaw_idx[0], :].flatten(0, 1) for i in target_f1_split)  # BKxC
        target_f2_sort = tuple(i[batch_idx, jigsaw_idx[1], :].flatten(0, 1) for i in target_f2_split)  # BKxC

        # Features from projector
        context_z1 = tuple(f(x) for f, x in zip(self.context_projector, context_f1)) # BxC
        context_z2 = tuple(f(x) for f, x in zip(self.context_projector, context_f2)) # BxC
        target_z1 = tuple(f(x) for f, x in zip(self.target_projector, target_f1_sort)) # BKxC
        target_z2 = tuple(f(x) for f, x in zip(self.target_projector, target_f2_sort)) # BKxC

        # Features from predictor 
        context_p1 = tuple(f(x) for f, x in zip(self.context_predictor, context_z1)) # BxC
        context_p2 = tuple(f(x) for f, x in zip(self.context_predictor, context_z2)) # BxC
        target_p1 = tuple(f(x) for f, x in zip(self.target_predictor, target_z1)) # BKxC
        target_p2 = tuple(f(x) for f, x in zip(self.target_predictor, target_z2)) # BKxC

        context_z1_detach = tuple(i.detach() for i in context_z1)
        context_z2_detach = tuple(i.detach() for i in context_z2)
        target_z1_detach = tuple(i.detach() for i in target_z1)
        target_z2_detach = tuple(i.detach() for i in target_z2)

        ### Fuser path
        # Combine context and target features
        ms_f1 = tuple(torch.cat((i, j[:, :self.n_keep, :].flatten(1)), dim=1) for i,j in zip(context_f1, target_f1_split)) # Bx(K+1)C
        ms_f2 = tuple(torch.cat((i, j[:, :self.n_keep, :].flatten(1)), dim=1) for i,j in zip(context_f2, target_f2_split)) # Bx(K+1)C

        # fuser contrastive learning
        ms_z1, ms_z2, ms_p1, ms_p2 = [], [], [], []
        for i, (f_proj, f_pred) in enumerate(zip(self.inter_projector, self.inter_predictor)):
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


class MSModel7_2(MSModel7):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, scale, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super().__init__(base_encoder, scale)

        self.context_encoder = base_encoder(zero_init_residual=True, return_features=True)
        self.target_encoder = base_encoder(zero_init_residual=True, return_features=True)
        self.context_encoder.fc = nn.Identity()
        self.target_encoder.fc = nn.Identity()


class MSModel7_3(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, scale, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super().__init__()

        self.K = int(scale**2)

        # create encoders
        self.context_encoder = base_encoder(zero_init_residual=True, return_features=True)
        self.target_encoder = base_encoder(zero_init_residual=True, return_features=True)
        prev_dim = self.context_encoder.fc.weight.shape[1]
        self.context_encoder.fc = nn.Identity()
        self.target_encoder.fc = nn.Identity()
   
        self.inter_dim = torch.as_tensor([64, 128, 256, 512]) # TODO: hardcoded
        self.ms_inter_dim = self.inter_dim * (self.K//2+1) # TODO: hardcoded

        # build a 3-layer projector
        self.context_projector = nn.ModuleList([make_projector(d, dim) for d in self.inter_dim])
        self.target_projector = nn.ModuleList([make_projector(d, dim) for d in self.inter_dim])
        self.inter_projector = nn.ModuleList([make_projector(d, dim) for d in self.ms_inter_dim])

        # build a 2-layer predictor
        self.context_predictor = nn.ModuleList([make_predictor(dim, pred_dim) for d in self.inter_dim])
        self.target_predictor = nn.ModuleList([make_predictor(dim, pred_dim) for d in self.inter_dim])                
        self.inter_predictor = nn.ModuleList([make_predictor(dim, pred_dim) for d in self.ms_inter_dim])

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
        context_f1, context_f2 = self.context_encoder(x1[0]), self.context_encoder(x2[0]) # BxC
        target_f1, target_f2 = self.target_encoder(x1[1]), self.target_encoder(x2[1]) # BKxC
        
        target_f1_split = tuple(i.reshape(B, self.K, -1) for i in target_f1) # BxKxC
        target_f2_split = tuple(i.reshape(B, self.K, -1) for i in target_f2) # BxKxC

        # reorder target patches
        batch_idx = torch.arange(B).repeat(self.K, 1).t()
        assert batch_idx.shape == jigsaw_idx[0].shape == jigsaw_idx[1].shape
        target_f1_sort = tuple(i[batch_idx, jigsaw_idx[0], :].flatten(0, 1) for i in target_f1_split)  # BKxC
        target_f2_sort = tuple(i[batch_idx, jigsaw_idx[1], :].flatten(0, 1) for i in target_f2_split)  # BKxC

        # Features from projector
        context_z1 = tuple(f(x) for f, x in zip(self.context_projector, context_f1)) # BxC
        context_z2 = tuple(f(x) for f, x in zip(self.context_projector, context_f2)) # BxC
        target_z1 = tuple(f(x) for f, x in zip(self.target_projector, target_f1_sort)) # BKxC
        target_z2 = tuple(f(x) for f, x in zip(self.target_projector, target_f2_sort)) # BKxC

        # Features from predictor 
        context_p1 = tuple(f(x) for f, x in zip(self.context_predictor, context_z1)) # BxC
        context_p2 = tuple(f(x) for f, x in zip(self.context_predictor, context_z2)) # BxC
        target_p1 = tuple(f(x) for f, x in zip(self.target_predictor, target_z1)) # BKxC
        target_p2 = tuple(f(x) for f, x in zip(self.target_predictor, target_z2)) # BKxC

        context_z1_detach = tuple(i.detach() for i in context_z1)
        context_z2_detach = tuple(i.detach() for i in context_z2)
        target_z1_detach = tuple(i.detach() for i in target_z1)
        target_z2_detach = tuple(i.detach() for i in target_z2)

        ### Fuser path
        # Combine context and target features
        ms_f1 = tuple(torch.cat((i, j[:, :8, :].flatten(1)), dim=1) for i,j in zip(context_f1, target_f1_split)) # Bx(K+1)C
        ms_f2 = tuple(torch.cat((i, j[:, :8, :].flatten(1)), dim=1) for i,j in zip(context_f2, target_f2_split)) # Bx(K+1)C

        # fuser contrastive learning
        ms_z1, ms_z2, ms_p1, ms_p2 = [], [], [], []
        for i, (f_proj, f_pred) in enumerate(zip(self.inter_projector, self.inter_predictor)):
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


class ABModel(nn.Module):
    """
    same as MSModel7
    """
    def __init__(self, base_encoder, scale, mask_ratio=0.5):
        super().__init__()

        self.K = int(scale**2)
        self.n_keep = int(self.K * (1-mask_ratio))

        # create encoders
        self.context_encoder = base_encoder(zero_init_residual=True, pretrained=True, return_features=True)
        self.target_encoder = base_encoder(zero_init_residual=True, pretrained=True, return_features=True)
        self.context_encoder.fc = nn.Identity()
        self.target_encoder.fc = nn.Identity()
   
        self.inter_dim = torch.as_tensor([64, 128, 256, 512]) # TODO: hardcoded
        self.ms_inter_dim = self.inter_dim * (self.n_keep+1)

        # build a 3-layer projector
        self.context_projector = nn.ModuleList([make_projector(d, d) for d in self.inter_dim])
        self.target_projector = nn.ModuleList([make_projector(d, d) for d in self.inter_dim])
        self.inter_projector = nn.ModuleList([make_projector(d, d) for d in self.ms_inter_dim])

        # build a 2-layer predictor
        self.context_predictor = nn.ModuleList([make_predictor(d, torch.div(d,4,rounding_mode="floor")) for d in self.inter_dim])
        self.target_predictor = nn.ModuleList([make_predictor(d, torch.div(d,4,rounding_mode="floor")) for d in self.inter_dim])                
        self.inter_predictor = nn.ModuleList([make_predictor(d, torch.div(d,4,rounding_mode="floor")) for d in self.ms_inter_dim])

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
        context_f1, context_f2 = self.context_encoder(x1[0]), self.context_encoder(x2[0]) # BxC
        target_f1, target_f2 = self.target_encoder(x1[1]), self.target_encoder(x2[1]) # BKxC
        
        target_f1_split = tuple(i.reshape(B, self.K, -1) for i in target_f1) # BxKxC
        target_f2_split = tuple(i.reshape(B, self.K, -1) for i in target_f2) # BxKxC

        # reorder target patches
        batch_idx = torch.arange(B).repeat(self.K, 1).t()
        assert batch_idx.shape == jigsaw_idx[0].shape == jigsaw_idx[1].shape
        target_f1_sort = tuple(i[batch_idx, jigsaw_idx[0], :].flatten(0, 1) for i in target_f1_split)  # BKxC
        target_f2_sort = tuple(i[batch_idx, jigsaw_idx[1], :].flatten(0, 1) for i in target_f2_split)  # BKxC

        # Features from projector
        context_z1 = tuple(f(x) for f, x in zip(self.context_projector, context_f1)) # BxC
        context_z2 = tuple(f(x) for f, x in zip(self.context_projector, context_f2)) # BxC
        target_z1 = tuple(f(x) for f, x in zip(self.target_projector, target_f1_sort)) # BKxC
        target_z2 = tuple(f(x) for f, x in zip(self.target_projector, target_f2_sort)) # BKxC

        # Features from predictor 
        context_p1 = tuple(f(x) for f, x in zip(self.context_predictor, context_z1)) # BxC
        context_p2 = tuple(f(x) for f, x in zip(self.context_predictor, context_z2)) # BxC
        target_p1 = tuple(f(x) for f, x in zip(self.target_predictor, target_z1)) # BKxC
        target_p2 = tuple(f(x) for f, x in zip(self.target_predictor, target_z2)) # BKxC

        context_z1_detach = tuple(i.detach() for i in context_z1)
        context_z2_detach = tuple(i.detach() for i in context_z2)
        target_z1_detach = tuple(i.detach() for i in target_z1)
        target_z2_detach = tuple(i.detach() for i in target_z2)

        ### Fuser path
        # Combine context and target features
        ms_f1 = tuple(torch.cat((i, j[:, :self.n_keep, :].flatten(1)), dim=1) for i,j in zip(context_f1, target_f1_split)) # Bx(K+1)C
        ms_f2 = tuple(torch.cat((i, j[:, :self.n_keep, :].flatten(1)), dim=1) for i,j in zip(context_f2, target_f2_split)) # Bx(K+1)C

        # fuser contrastive learning
        ms_z1, ms_z2, ms_p1, ms_p2 = [], [], [], []
        for i, (f_proj, f_pred) in enumerate(zip(self.inter_projector, self.inter_predictor)):
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


class MSModelAB1_1(nn.Module):
    """
    from MSModel5_2, only jigsaw
    """
    def __init__(self, base_encoder, data_norm, scale, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super().__init__()

        self.data_norm = data_norm
        self.K = int(scale**2)

        # create encoders
        self.context_encoder = base_encoder(zero_init_residual=True)
        self.target_encoder = base_encoder(zero_init_residual=True)
        prev_dim = self.context_encoder.fc.weight.shape[1]
        self.context_encoder.fc = nn.Identity()
        self.target_encoder.fc = nn.Identity()

        # build a 3-layer projector
        self.context_projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(prev_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer

        self.target_projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(prev_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer

        fuser_dim = prev_dim * (self.K+1)
        self.fuser_projector = nn.Sequential(nn.Linear(fuser_dim, fuser_dim, bias=False),
                                        nn.BatchNorm1d(fuser_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(fuser_dim, fuser_dim, bias=False),
                                        nn.BatchNorm1d(fuser_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(fuser_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer



        # build a 2-layer predictor
        self.context_predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                                nn.BatchNorm1d(pred_dim),
                                                nn.ReLU(inplace=True), # hidden layer
                                                nn.Linear(pred_dim, dim)) # output layer

        self.target_predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                                nn.BatchNorm1d(pred_dim),
                                                nn.ReLU(inplace=True), # hidden layer
                                                nn.Linear(pred_dim, dim)) # output layer
                                                
        self.fuser_predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                                nn.BatchNorm1d(pred_dim),
                                                nn.ReLU(inplace=True), # hidden layer
                                                nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2, jigsaw_idx=None):
        """
        Input:
            x1: first views of images [context_images, target_images]
            x2: second views of images [context_images, target_images]
        Output:
            p1, p2, z1, z2: predictors and targets of the network
        """
        x1 = tuple(self.data_norm(i) for i in x1)
        x2 = tuple(self.data_norm(i) for i in x2)
        B = x1[0].shape[0]

        # Features from encoder
        context_f1, context_f2 = self.context_encoder(x1[0]), self.context_encoder(x2[0]) # BxC
        target_f1, target_f2 = self.target_encoder(x1[1]), self.target_encoder(x2[1]) # BKxC
        target_f1_split, target_f2_split = target_f1.reshape(B, self.K, -1), target_f2.reshape(B, self.K, -1) # BxKxC

        # reorder target patches
        batch_idx = torch.arange(B).repeat(self.K, 1).t()
        assert batch_idx.shape == jigsaw_idx[0].shape == jigsaw_idx[1].shape, f"batch_idx and jigsaw_idx shape mismatch {batch_idx.shape} {jigsaw_idx[0].shape} {jigsaw_idx[1].shape}"
        target_f1, target_f2 = target_f1_split[batch_idx, jigsaw_idx[0], :].flatten(0, 1), target_f2_split[batch_idx, jigsaw_idx[1], :].flatten(0, 1) # BKxC

        # Features from projector
        context_z1, context_z2 = self.context_projector(context_f1), self.context_projector(context_f2) # BxC
        target_z1, target_z2 = self.target_projector(target_f1), self.target_projector(target_f2) # BKxC

        # Features from predictor 
        context_p1, context_p2 = self.context_predictor(context_z1), self.context_predictor(context_z2) # BxC
        target_p1, target_p2 = self.target_predictor(target_z1), self.target_predictor(target_z2) # BKxC

        # Combine context and target features
        ms_f1, ms_f2 = torch.cat((context_f1, target_f1_split.flatten(1)), dim=1), torch.cat((context_f2, target_f2_split.flatten(1)), dim=1) # Bx(K+1)C
        assert ms_f1.shape[0] == ms_f2.shape[0] == B

        # fuser contrastive learning
        ms_z1, ms_z2 = self.fuser_projector(ms_f1), self.fuser_projector(ms_f2)
        ms_p1, ms_p2 = self.fuser_predictor(ms_z1), self.fuser_predictor(ms_z2)

        return (
            (context_p1, context_p2, context_z1.detach(), context_z2.detach()),
            (target_p1, target_p2, target_z1.detach(), target_z2.detach()),
            (ms_p1, ms_p2, ms_z1.detach(), ms_z2.detach()),
        )


class MSModelAB1_2(nn.Module):
    """
    from MSModel5_2, only masking
    """
    def __init__(self, base_encoder, data_norm, scale, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super().__init__()

        self.data_norm = data_norm
        self.K = int(scale**2)

        # create encoders
        self.context_encoder = base_encoder(zero_init_residual=True)
        self.target_encoder = base_encoder(zero_init_residual=True)
        prev_dim = self.context_encoder.fc.weight.shape[1]
        self.context_encoder.fc = nn.Identity()
        self.target_encoder.fc = nn.Identity()

        # build a 3-layer projector
        self.context_projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(prev_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer

        self.target_projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(prev_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer

        fuser_dim = prev_dim * (self.K//2+1)
        self.fuser_projector = nn.Sequential(nn.Linear(fuser_dim, fuser_dim, bias=False),
                                        nn.BatchNorm1d(fuser_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(fuser_dim, fuser_dim, bias=False),
                                        nn.BatchNorm1d(fuser_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(fuser_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer



        # build a 2-layer predictor
        self.context_predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                                nn.BatchNorm1d(pred_dim),
                                                nn.ReLU(inplace=True), # hidden layer
                                                nn.Linear(pred_dim, dim)) # output layer

        self.target_predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                                nn.BatchNorm1d(pred_dim),
                                                nn.ReLU(inplace=True), # hidden layer
                                                nn.Linear(pred_dim, dim)) # output layer
                                                
        self.fuser_predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                                nn.BatchNorm1d(pred_dim),
                                                nn.ReLU(inplace=True), # hidden layer
                                                nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2, jigsaw_idx=None):
        """
        Input:
            x1: first views of images [context_images, target_images]
            x2: second views of images [context_images, target_images]
        Output:
            p1, p2, z1, z2: predictors and targets of the network
        """
        x1 = tuple(self.data_norm(i) for i in x1)
        x2 = tuple(self.data_norm(i) for i in x2)
        B = x1[0].shape[0]

        # Features from encoder
        context_f1, context_f2 = self.context_encoder(x1[0]), self.context_encoder(x2[0]) # BxC
        target_f1, target_f2 = self.target_encoder(x1[1]), self.target_encoder(x2[1]) # BKxC
        target_f1_split, target_f2_split = target_f1.reshape(B, self.K, -1), target_f2.reshape(B, self.K, -1) # BxKxC

        # Features from projector
        context_z1, context_z2 = self.context_projector(context_f1), self.context_projector(context_f2) # BxC
        target_z1, target_z2 = self.target_projector(target_f1), self.target_projector(target_f2) # BKxC

        # Features from predictor 
        context_p1, context_p2 = self.context_predictor(context_z1), self.context_predictor(context_z2) # BxC
        target_p1, target_p2 = self.target_predictor(target_z1), self.target_predictor(target_z2) # BKxC

        # Combine context and target features
        ms_f1, ms_f2 = torch.cat((context_f1, target_f1_split[:,sorted(random.sample(range(16), 8)),:].flatten(1)), dim=1), torch.cat((context_f2, target_f2_split[:,sorted(random.sample(range(16), 8)),:].flatten(1)), dim=1) # Bx(K+1)C
        assert ms_f1.shape[0] == ms_f2.shape[0] == B

        # fuser contrastive learning
        ms_z1, ms_z2 = self.fuser_projector(ms_f1), self.fuser_projector(ms_f2)
        ms_p1, ms_p2 = self.fuser_predictor(ms_z1), self.fuser_predictor(ms_z2)

        return (
            (context_p1, context_p2, context_z1.detach(), context_z2.detach()),
            (target_p1, target_p2, target_z1.detach(), target_z2.detach()),
            (ms_p1, ms_p2, ms_z1.detach(), ms_z2.detach()),
        )

class MSModelAB1_3(nn.Module):
    """
    from MSModel7
    """
    def __init__(self, base_encoder, data_norm, scale):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super().__init__()

        self.data_norm = data_norm
        self.K = int(scale**2)

        # create encoders
        self.context_encoder = base_encoder(zero_init_residual=True)
        self.target_encoder = base_encoder(zero_init_residual=True)
        self.context_encoder.fc = nn.Identity()
        self.target_encoder.fc = nn.Identity()
   
        self.inter_dim = torch.as_tensor([64, 128, 256, 512]) # TODO: hardcoded

        # build a 3-layer projector
        self.context_projector = nn.ModuleList([make_projector(d, d) for d in self.inter_dim])
        self.target_projector = nn.ModuleList([make_projector(d, d) for d in self.inter_dim])

        # build a 2-layer predictor
        self.context_predictor = nn.ModuleList([make_predictor(d, torch.div(d,4,rounding_mode="floor")) for d in self.inter_dim])
        self.target_predictor = nn.ModuleList([make_predictor(d, torch.div(d,4,rounding_mode="floor")) for d in self.inter_dim])                

    def forward(self, x1, x2, jigsaw_idx=None):
        """
        Input:
            x1: first views of images [context_images, target_images]
            x2: second views of images [context_images, target_images]
        Output:
            p1, p2, z1, z2: predictors and targets of the network
        """
        x1 = tuple(self.data_norm(i) for i in x1)
        x2 = tuple(self.data_norm(i) for i in x2)
        B = x1[0].shape[0]

        # Features from encoder
        context_f1, context_f2 = self.context_encoder(x1[0]), self.context_encoder(x2[0]) # BxC
        target_f1, target_f2 = self.target_encoder(x1[1]), self.target_encoder(x2[1]) # BKxC

        # Features from projector
        context_z1 = tuple(f(x) for f, x in zip(self.context_projector, context_f1)) # BxC
        context_z2 = tuple(f(x) for f, x in zip(self.context_projector, context_f2)) # BxC
        target_z1 = tuple(f(x) for f, x in zip(self.target_projector, target_f1)) # BKxC
        target_z2 = tuple(f(x) for f, x in zip(self.target_projector, target_f2)) # BKxC

        # Features from predictor 
        context_p1 = tuple(f(x) for f, x in zip(self.context_predictor, context_z1)) # BxC
        context_p2 = tuple(f(x) for f, x in zip(self.context_predictor, context_z2)) # BxC
        target_p1 = tuple(f(x) for f, x in zip(self.target_predictor, target_z1)) # BKxC
        target_p2 = tuple(f(x) for f, x in zip(self.target_predictor, target_z2)) # BKxC

        context_z1_detach = tuple(i.detach() for i in context_z1)
        context_z2_detach = tuple(i.detach() for i in context_z2)
        target_z1_detach = tuple(i.detach() for i in target_z1)
        target_z2_detach = tuple(i.detach() for i in target_z2)

        return (
            (context_p1, context_p2, context_z1_detach, context_z2_detach),
            (target_p1, target_p2, target_z1_detach, target_z2_detach),
        )


class ABModelRes50(ABModel):
    """
    modified from ABModel
    """
    def __init__(self, base_encoder, scale, mask_ratio=0.5):
        super().__init__(base_encoder, scale, mask_ratio)

        self.K = int(scale**2)
        self.n_keep = int(self.K * (1-mask_ratio))

        # create encoders
        self.context_encoder = base_encoder(zero_init_residual=True, pretrained=True, return_features=True)
        self.target_encoder = base_encoder(zero_init_residual=True, pretrained=True, return_features=True)
        self.context_encoder.fc = nn.Identity()
        self.target_encoder.fc = nn.Identity()
   
        self.inter_dim = torch.as_tensor([256, 512, 1024, 2048]) # TODO: hardcoded
        self.ms_inter_dim = self.inter_dim * (self.n_keep+1)

        # build a 3-layer projector
        self.context_projector = nn.ModuleList([make_projector(d, d) for d in self.inter_dim])
        self.target_projector = nn.ModuleList([make_projector(d, d) for d in self.inter_dim])
        self.inter_projector = nn.ModuleList([make_projector(d, d) for d in self.ms_inter_dim])

        # build a 2-layer predictor
        self.context_predictor = nn.ModuleList([make_predictor(d, torch.div(d,4,rounding_mode="floor")) for d in self.inter_dim])
        self.target_predictor = nn.ModuleList([make_predictor(d, torch.div(d,4,rounding_mode="floor")) for d in self.inter_dim])                
        self.inter_predictor = nn.ModuleList([make_predictor(d, torch.div(d,4,rounding_mode="floor")) for d in self.ms_inter_dim])


class ABModelDenseNet(nn.Module):
    """
    modified from ABModel
    """
    def __init__(self, base_encoder, scale, mask_ratio=0.5):
        super().__init__()

        self.K = int(scale**2)
        self.n_keep = int(self.K * (1-mask_ratio))

        # create encoders
        self.context_encoder = base_encoder("densenet121", weights="imagenet")
        self.target_encoder = base_encoder("densenet121", weights="imagenet")
   
        self.inter_dim = torch.as_tensor([256, 512, 1024, 1024]) # TODO: hardcoded
        self.ms_inter_dim = self.inter_dim * (self.n_keep+1)

        # build a 3-layer projector
        self.context_projector = nn.ModuleList([make_projector(d, d) for d in self.inter_dim])
        self.target_projector = nn.ModuleList([make_projector(d, d) for d in self.inter_dim])
        self.inter_projector = nn.ModuleList([make_projector(d, d) for d in self.ms_inter_dim])

        # build a 2-layer predictor
        self.context_predictor = nn.ModuleList([make_predictor(d, torch.div(d,4,rounding_mode="floor")) for d in self.inter_dim])
        self.target_predictor = nn.ModuleList([make_predictor(d, torch.div(d,4,rounding_mode="floor")) for d in self.inter_dim])                
        self.inter_predictor = nn.ModuleList([make_predictor(d, torch.div(d,4,rounding_mode="floor")) for d in self.ms_inter_dim])

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
        context_f1, context_f2 = self.context_encoder(x1[0])[2:], self.context_encoder(x2[0])[2:] # BxC
        target_f1, target_f2 = self.target_encoder(x1[1])[2:], self.target_encoder(x2[1])[2:] # BKxC

        for i in range(4):
            context_f1[i] = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(context_f1[i], 1), 1)
            context_f2[i] = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(context_f2[i], 1), 1)
            target_f1[i] = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(target_f1[i], 1), 1)
            target_f2[i] = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(target_f2[i], 1), 1)
        
        target_f1_split = tuple(i.reshape(B, self.K, -1) for i in target_f1) # BxKxC
        target_f2_split = tuple(i.reshape(B, self.K, -1) for i in target_f2) # BxKxC

        # reorder target patches
        batch_idx = torch.arange(B).repeat(self.K, 1).t()
        assert batch_idx.shape == jigsaw_idx[0].shape == jigsaw_idx[1].shape
        target_f1_sort = tuple(i[batch_idx, jigsaw_idx[0], :].flatten(0, 1) for i in target_f1_split)  # BKxC
        target_f2_sort = tuple(i[batch_idx, jigsaw_idx[1], :].flatten(0, 1) for i in target_f2_split)  # BKxC

        # Features from projector
        context_z1 = tuple(f(x) for f, x in zip(self.context_projector, context_f1)) # BxC
        context_z2 = tuple(f(x) for f, x in zip(self.context_projector, context_f2)) # BxC
        target_z1 = tuple(f(x) for f, x in zip(self.target_projector, target_f1_sort)) # BKxC
        target_z2 = tuple(f(x) for f, x in zip(self.target_projector, target_f2_sort)) # BKxC

        # Features from predictor 
        context_p1 = tuple(f(x) for f, x in zip(self.context_predictor, context_z1)) # BxC
        context_p2 = tuple(f(x) for f, x in zip(self.context_predictor, context_z2)) # BxC
        target_p1 = tuple(f(x) for f, x in zip(self.target_predictor, target_z1)) # BKxC
        target_p2 = tuple(f(x) for f, x in zip(self.target_predictor, target_z2)) # BKxC

        context_z1_detach = tuple(i.detach() for i in context_z1)
        context_z2_detach = tuple(i.detach() for i in context_z2)
        target_z1_detach = tuple(i.detach() for i in target_z1)
        target_z2_detach = tuple(i.detach() for i in target_z2)

        ### Fuser path
        # Combine context and target features
        ms_f1 = tuple(torch.cat((i, j[:, :self.n_keep, :].flatten(1)), dim=1) for i,j in zip(context_f1, target_f1_split)) # Bx(K+1)C
        ms_f2 = tuple(torch.cat((i, j[:, :self.n_keep, :].flatten(1)), dim=1) for i,j in zip(context_f2, target_f2_split)) # Bx(K+1)C

        # fuser contrastive learning
        ms_z1, ms_z2, ms_p1, ms_p2 = [], [], [], []
        for i, (f_proj, f_pred) in enumerate(zip(self.inter_projector, self.inter_predictor)):
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


class ABModelEfficient(ABModelDenseNet):
    """
    modified from ABModel
    """
    def __init__(self, base_encoder, scale, mask_ratio=0.5):
        super().__init__(base_encoder, scale, mask_ratio)

        self.K = int(scale**2)
        self.n_keep = int(self.K * (1-mask_ratio))

        # create encoders
        self.context_encoder = base_encoder("efficientnet-b0", weights="imagenet")
        self.target_encoder = base_encoder("efficientnet-b0", weights="imagenet")
   
        self.inter_dim = torch.as_tensor([24, 40, 112, 320]) # TODO: hardcoded
        self.ms_inter_dim = self.inter_dim * (self.n_keep+1)

        # build a 3-layer projector
        self.context_projector = nn.ModuleList([make_projector(d, d) for d in self.inter_dim])
        self.target_projector = nn.ModuleList([make_projector(d, d) for d in self.inter_dim])
        self.inter_projector = nn.ModuleList([make_projector(d, d) for d in self.ms_inter_dim])

        # build a 2-layer predictor
        self.context_predictor = nn.ModuleList([make_predictor(d, torch.div(d,4,rounding_mode="floor")) for d in self.inter_dim])
        self.target_predictor = nn.ModuleList([make_predictor(d, torch.div(d,4,rounding_mode="floor")) for d in self.inter_dim])                
        self.inter_predictor = nn.ModuleList([make_predictor(d, torch.div(d,4,rounding_mode="floor")) for d in self.ms_inter_dim])


class ABModelRegNet(ABModelDenseNet):
    """
    modified from ABModel
    """
    def __init__(self, base_encoder, scale, mask_ratio=0.5):
        super().__init__(base_encoder, scale, mask_ratio)

        self.K = int(scale**2)
        self.n_keep = int(self.K * (1-mask_ratio))

        # create encoders
        self.context_encoder = base_encoder("timm-regnety_008", weights="imagenet")
        self.target_encoder = base_encoder("timm-regnety_008", weights="imagenet")
   
        self.inter_dim = torch.as_tensor([64, 128, 320, 768]) # TODO: hardcoded
        self.ms_inter_dim = self.inter_dim * (self.n_keep+1)

        # build a 3-layer projector
        self.context_projector = nn.ModuleList([make_projector(d, d) for d in self.inter_dim])
        self.target_projector = nn.ModuleList([make_projector(d, d) for d in self.inter_dim])
        self.inter_projector = nn.ModuleList([make_projector(d, d) for d in self.ms_inter_dim])

        # build a 2-layer predictor
        self.context_predictor = nn.ModuleList([make_predictor(d, torch.div(d,4,rounding_mode="floor")) for d in self.inter_dim])
        self.target_predictor = nn.ModuleList([make_predictor(d, torch.div(d,4,rounding_mode="floor")) for d in self.inter_dim])                
        self.inter_predictor = nn.ModuleList([make_predictor(d, torch.div(d,4,rounding_mode="floor")) for d in self.ms_inter_dim])


class ABModelMiT(ABModelDenseNet):
    """
    modified from ABModel
    """
    def __init__(self, base_encoder, scale, mask_ratio=0.5):
        super().__init__(base_encoder, scale, mask_ratio)

        self.K = int(scale**2)
        self.n_keep = int(self.K * (1-mask_ratio))

        # create encoders
        self.context_encoder = base_encoder("mit_b0", weights="imagenet")
        self.target_encoder = base_encoder("mit_b0", weights="imagenet")
   
        self.inter_dim = torch.as_tensor([32, 64, 160, 256]) # TODO: hardcoded
        self.ms_inter_dim = self.inter_dim * (self.n_keep+1)

        # build a 3-layer projector
        self.context_projector = nn.ModuleList([make_projector(d, d) for d in self.inter_dim])
        self.target_projector = nn.ModuleList([make_projector(d, d) for d in self.inter_dim])
        self.inter_projector = nn.ModuleList([make_projector(d, d) for d in self.ms_inter_dim])

        # build a 2-layer predictor
        self.context_predictor = nn.ModuleList([make_predictor(d, torch.div(d,4,rounding_mode="floor")) for d in self.inter_dim])
        self.target_predictor = nn.ModuleList([make_predictor(d, torch.div(d,4,rounding_mode="floor")) for d in self.inter_dim])                
        self.inter_predictor = nn.ModuleList([make_predictor(d, torch.div(d,4,rounding_mode="floor")) for d in self.ms_inter_dim])


class SimCLR(nn.Module):
    """
    Build a SimCLR model.
    """
    def __init__(self, base_encoder, dim=2048):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super().__init__()
        self.temperature = 0.5

        # create encoders
        self.context_encoder = base_encoder(zero_init_residual=True)
        self.target_encoder = base_encoder(zero_init_residual=True)
        prev_dim = self.context_encoder.fc.weight.shape[1]
        self.context_encoder.fc = nn.Identity()
        self.target_encoder.fc = nn.Identity()

        # build a 3-layer projector
        self.context_projector = nn.Sequential(nn.Linear(prev_dim, prev_dim),
                                                nn.ReLU(inplace=True), # first layer
                                                nn.Linear(prev_dim, dim)) # output layer

        self.target_projector = nn.Sequential(nn.Linear(prev_dim, prev_dim),
                                                nn.ReLU(inplace=True), # first layer
                                                nn.Linear(prev_dim, dim)) # output layer


    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images [context_images, target_images]
            x2: second views of images [context_images, target_images]
        Output:
            p1, p2, z1, z2: predictors and targets of the network
        """
        # Features from encoder
        context_f1, context_f2 = self.context_encoder(x1[0]), self.context_encoder(x2[0]) # BxC
        target_f1, target_f2 = self.target_encoder(x1[1]), self.target_encoder(x2[1]) # BKxC

        # Features from projector
        context_z1, context_z2 = self.context_projector(context_f1), self.context_projector(context_f2) # BxC
        target_z1, target_z2 = self.target_projector(target_f1), self.target_projector(target_f2) # BKxC

        context_z1, context_z2 = F.normalize(context_z1, dim=-1), F.normalize(context_z2, dim=-1)
        target_z1, target_z2 = F.normalize(target_z1, dim=-1), F.normalize(target_z2, dim=-1)

        context_logits, context_labels = self._cal_logits(context_z1, context_z2)
        target_logits, target_labels = self._cal_logits(target_z1, target_z2)

        return (context_logits, target_logits, context_labels, target_labels)

    def _cal_logits(self, z1, z2):
        """
        Calculate logits for a batch of features.
        Input:
            z1, z2: features of two views of images, shape [B, C]
        Output:
            logits: logits matrix, shape [B, B]
        """
        B = z1.shape[0]
        mask = F.one_hot(torch.arange(0, B, device=z1.device), num_classes=B)
        logits_aa = torch.matmul(z1, z1.T)
        logits_aa = logits_aa - mask * 1e9
        logits_bb = torch.matmul(z2, z2.T)
        logits_bb = logits_bb - mask * 1e9
        logits_ab = torch.matmul(z1, z2.T)
        logits_ba = torch.matmul(z2, z1.T)
        
        logits_a = torch.cat([logits_ab, logits_aa], dim=1)  # (n_video, 2*n_video)
        logits_b = torch.cat([logits_ba, logits_bb], dim=1)  # (n_video, 2*n_video)
        logits = (
            torch.cat([logits_a, logits_b], axis=0) / self.temperature
        )  # (2*n_video, 2*n_video)

        labels = torch.arange(0, B, device=z1.device)  # (n_video,)
        labels = torch.cat([labels, labels], axis=0)  # (2*n_video,)

        return logits, labels


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=512, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
        self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output