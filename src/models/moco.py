import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
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

class MoCo2(MoCo):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__(base_encoder, dim, K, m, T, mlp)

        self.encoder_p = base_encoder(num_classes=dim)
        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_p.fc.weight.shape[1]
            self.encoder_p.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_p.fc)

        self.fuser = nn.Sequential(nn.Conv1d(dim*4, dim, 1))
        
        for param_p in self.encoder_p.parameters():
            param_p.requires_grad = False  # not update by gradient

    def forward(self, im_q, im_k, im_p):
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

        cur_lv_feats = None
        prev_lv_feats = None
        if im_p is not None:
            cur_lv_feats = q

            # TODO: sum, concat, maxpool, attention machenism, different weights
            # sum
            # with torch.no_grad():
            #     im_p = torch.flatten(im_p, 0, 1) # (N*4)
            #     prev_lv_feats = self.encoder_p(im_p) # (N*4)xC
            #     prev_lv_feats = prev_lv_feats.reshape((im_q.shape[0],4,-1))
            #     prev_lv_feats = torch.sum(prev_lv_feats, dim=1)
            #     prev_lv_feats = nn.functional.normalize(prev_lv_feats, dim=1) 
            #     # / self.alpha

            # concat
            # with torch.no_grad():
            #     im_p = torch.flatten(im_p, 0, 1) # (N*4)
            #     prev_lv_feats = self.encoder_p(im_p) # (N*4)xC
            #     prev_lv_feats = prev_lv_feats.reshape((im_q.shape[0],-1)).unsqueeze(-1) # Nx(4*C)x1

            # prev_lv_feats = self.fuser(prev_lv_feats).squeeze() # NxC
            # prev_lv_feats = nn.functional.normalize(prev_lv_feats, dim=1) 

            # maxpool
            with torch.no_grad():
                im_p = torch.flatten(im_p, 0, 1) # (N*4)
                prev_lv_feats = self.encoder_p(im_p) # (N*4)xC
                prev_lv_feats = prev_lv_feats.reshape((im_q.shape[0],4,-1)).amax(1) # NxC
                prev_lv_feats = nn.functional.normalize(prev_lv_feats, dim=1)
                

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

        return logits, labels, cur_lv_feats, prev_lv_feats

class MoCo3(MoCo):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, fuse_style="none"):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__(base_encoder, dim, K, m, T, mlp)

        self.fuse_style = fuse_style
        self.encoder_p = base_encoder(num_classes=dim)
        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_p.fc.weight.shape[1]
            self.encoder_p.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_p.fc)

        if self.fuse_style == "cat":
            self.fuser = nn.Sequential(nn.Conv1d(dim*4, dim, 1))
        
        for param_p in self.encoder_p.parameters():
            param_p.requires_grad = False  # not update by gradient

    def forward(self, im_q, im_k, im_p):
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

        cur_lv_feats = None
        prev_lv_feats = None
        if im_p is not None:
            cur_lv_feats = q

            # TODO: sum, concat, maxpool, attention machenism, different weights
            if self.fuse_style == "sum":
                with torch.no_grad():
                    im_p = torch.flatten(im_p, 0, 1) # (N*4)
                    prev_lv_feats = self.encoder_p(im_p) # (N*4)xC
                    prev_lv_feats = prev_lv_feats.reshape((im_q.shape[0],4,-1))
                    prev_lv_feats = torch.sum(prev_lv_feats, dim=1) # NxC
                    prev_lv_feats = nn.functional.normalize(prev_lv_feats, dim=1) 
                    # / self.alpha

            elif self.fuse_style == "cat":
                with torch.no_grad():
                    im_p = torch.flatten(im_p, 0, 1) # (N*4)
                    prev_lv_feats = self.encoder_p(im_p) # (N*4)xC
                    prev_lv_feats = prev_lv_feats.reshape((im_q.shape[0],-1)).unsqueeze(-1) # Nx(4*C)x1

                prev_lv_feats = self.fuser(prev_lv_feats).squeeze() # NxC
                prev_lv_feats = nn.functional.normalize(prev_lv_feats, dim=1) 

            elif self.fuse_style == "max":
                with torch.no_grad():
                    im_p = torch.flatten(im_p, 0, 1) # (N*4)
                    prev_lv_feats = self.encoder_p(im_p) # (N*4)xC
                    prev_lv_feats = prev_lv_feats.reshape((im_q.shape[0],4,-1)).amax(1) # NxC
                    prev_lv_feats = nn.functional.normalize(prev_lv_feats, dim=1)             

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

        return logits, labels, cur_lv_feats, prev_lv_feats

class MoCo4(MoCo):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, fuse_style="none"):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__(base_encoder, dim, K, m, T, mlp)

        self.fuse_style = fuse_style
        self.encoder_p = base_encoder(num_classes=dim)
        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_p.fc.weight.shape[1]
            self.encoder_p.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_p.fc)

        self.predictor = nn.Sequential(nn.Linear(dim, 4096, bias=False), nn.BatchNorm1d(4096), nn.ReLU(inplace=True), 
                                        nn.Linear(4096, dim, bias=False))

        if self.fuse_style == "cat":
            self.fuser = nn.Sequential(nn.Conv1d(dim*4, dim, 1))
        
        for param_p in self.encoder_p.parameters():
            param_p.requires_grad = False  # not update by gradient

    def forward(self, im_q, im_k, im_p):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        cur_lv_feats = None
        prev_lv_feats = None
        if im_p is not None:
            cur_lv_feats = self.predictor(q)
            cur_lv_feats = nn.functional.normalize(cur_lv_feats, dim=1)

            # TODO: sum, concat, maxpool, attention machenism, different weights
            if self.fuse_style == "sum":
                with torch.no_grad():
                    im_p = torch.flatten(im_p, 0, 1) # (N*4)
                    prev_lv_feats = self.encoder_p(im_p) # (N*4)xC
                    prev_lv_feats = prev_lv_feats.reshape((im_q.shape[0],4,-1))
                    prev_lv_feats = torch.sum(prev_lv_feats, dim=1) # NxC
                    prev_lv_feats = nn.functional.normalize(prev_lv_feats, dim=1) 
                    # / self.alpha

            elif self.fuse_style == "cat":
                with torch.no_grad():
                    im_p = torch.flatten(im_p, 0, 1) # (N*4)
                    prev_lv_feats = self.encoder_p(im_p) # (N*4)xC
                    prev_lv_feats = prev_lv_feats.reshape((im_q.shape[0],-1)).unsqueeze(-1) # Nx(4*C)x1

                prev_lv_feats = self.fuser(prev_lv_feats).squeeze() # NxC
                prev_lv_feats = nn.functional.normalize(prev_lv_feats, dim=1) 

            elif self.fuse_style == "max":
                with torch.no_grad():
                    im_p = torch.flatten(im_p, 0, 1) # (N*4)
                    prev_lv_feats = self.encoder_p(im_p) # (N*4)xC
                    prev_lv_feats = prev_lv_feats.reshape((im_q.shape[0],4,-1)).amax(1) # NxC
                    prev_lv_feats = nn.functional.normalize(prev_lv_feats, dim=1)             


        q = nn.functional.normalize(q, dim=1)

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

        return logits, labels, cur_lv_feats, prev_lv_feats

class MoCo5(MoCo):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, fuse_style="none"):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__(base_encoder, dim, K, m, T, mlp)

        self.fuse_style = fuse_style
        self.encoder_p = base_encoder(num_classes=dim)
        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_p.fc.weight.shape[1]
            self.encoder_p.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_p.fc)

        self.predictor = nn.Sequential(nn.Linear(dim, 4096, bias=False), nn.BatchNorm1d(4096), nn.ReLU(inplace=True), 
                                        nn.Linear(4096, dim, bias=False))

        if self.fuse_style == "cat":
            self.fuser = nn.Sequential(nn.Conv1d(dim*4, dim, 1))
        
        for param_p in self.encoder_p.parameters():
            param_p.requires_grad = False  # not update by gradient

    def forward(self, im_q, im_k, im_p):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        cur_lv_feats = None
        prev_lv_feats = None
        if im_p is not None:
            cur_lv_feats = self.predictor(q)
            cur_lv_feats = nn.functional.normalize(cur_lv_feats, dim=1)

            # TODO: sum, concat, maxpool, attention machenism, different weights
            if self.fuse_style == "sum":
                with torch.no_grad():
                    im_p = torch.flatten(im_p, 0, 1) # (N*4)
                    prev_lv_feats = self.encoder_p(im_p) # (N*4)xC
                    prev_lv_feats = prev_lv_feats.reshape((im_q.shape[0],4,-1))
                    prev_lv_feats = torch.sum(prev_lv_feats, dim=1) # NxC
                    prev_lv_feats = nn.functional.normalize(prev_lv_feats, dim=1) 
                    # / self.alpha

            elif self.fuse_style == "cat":
                with torch.no_grad():
                    im_p = torch.flatten(im_p, 0, 1) # (N*4)
                    prev_lv_feats = self.encoder_p(im_p) # (N*4)xC
                    prev_lv_feats = prev_lv_feats.reshape((im_q.shape[0],-1)).unsqueeze(-1) # Nx(4*C)x1

                prev_lv_feats = self.fuser(prev_lv_feats).squeeze() # NxC
                prev_lv_feats = nn.functional.normalize(prev_lv_feats, dim=1) 

            elif self.fuse_style == "max":
                with torch.no_grad():
                    im_p = torch.flatten(im_p, 0, 1) # (N*4)
                    prev_lv_feats = self.encoder_p(im_p) # (N*4)xC
                    prev_lv_feats = prev_lv_feats.reshape((im_q.shape[0],4,-1)).amax(1) # NxC
                    prev_lv_feats = nn.functional.normalize(prev_lv_feats, dim=1)      

            logits2 = torch.einsum('nc,ck->nk', [cur_lv_feats, prev_lv_feats.transpose(0,1)])
            labels2 = torch.arange(logits2.shape[0], dtype=torch.long).cuda()


        q = nn.functional.normalize(q, dim=1)

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

        return logits, labels, logits2, labels2


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
