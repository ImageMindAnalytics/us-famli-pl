import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from .layers import TimeDistributed, AttentionChunk, MHABlock, SelfAttention, ProjectionHead, FullAttention, RoPETransformerBlock

import torchvision 
from torchvision.transforms import v2


from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import grad_norm


from lightning.pytorch.loggers import NeptuneLogger
from neptune.types import File
from torchmetrics.aggregation import CatMetric



import matplotlib.pyplot as plt
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchmetrics

import os

def gaussian_nll(y, mu, log_var, reduction="mean"):
    """
    Gaussian Negative Log-Likelihood for heteroscedastic regression.
    NLL = 0.5 * ( (y-mu)^2 / var + log_var )    (constant log(2π) omitted)
    """
    var = torch.exp(log_var)
    nll = 0.5 * ((y - mu)**2 / var + log_var)
    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    return nll


class EfwNet(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        encoder = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        encoder.classifier = nn.Identity()
        self.encoder = TimeDistributed(encoder)
        
        p_encoding_z = torch.stack([self.positional_encoding(self.hparams.n_chunks, self.hparams.embed_dim, tag) for tag in range(self.hparams.tags)])
        self.register_buffer("p_encoding_z", p_encoding_z)
        
        self.proj = ProjectionHead(input_dim=self.hparams.features, hidden_dim=self.hparams.features, output_dim=self.hparams.embed_dim, activation=nn.PReLU)
        self.attn_chunk = AttentionChunk(input_dim=self.hparams.embed_dim, hidden_dim=64, chunks=self.hparams.n_chunks)

        self.ln0 = nn.LayerNorm(self.hparams.embed_dim)
        self.mha = MHABlock(embed_dim=self.hparams.embed_dim, num_heads=self.hparams.num_heads, dropout=self.hparams.dropout, causal_mask=True, return_weights=False)
        self.ln1 = nn.LayerNorm(self.hparams.embed_dim)

        self.dropout = nn.Dropout(self.hparams.dropout)
        
        self.attn = SelfAttention(input_dim=self.hparams.embed_dim, hidden_dim=64)
        self.proj_final = ProjectionHead(input_dim=self.hparams.embed_dim, hidden_dim=64, output_dim=1, activation=nn.PReLU)
        # self.proj_final = HeteroscedasticHead(input_dim=self.hparams.embed_dim, hidden=64)

        self.loss_fn = nn.HuberLoss(delta=self.hparams.huber_delta)
        # self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.MSELoss()
        # self.loss_fn = gaussian_nll
        self.l1_fn = torch.nn.L1Loss()
        

        self.train_transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(),                
                v2.RandomChoice([
                    v2.Compose([v2.RandomRotation(180), v2.Pad(64), v2.RandomCrop(size=256)]),
                    v2.RandomResizedCrop(size=256, scale=(0.4, 1.0), ratio=(0.75, 1.3333333333333333))
                ]),
                v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
                v2.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2])
            ]
        )

        # self.all2iq3 = torch.jit.load("/mnt/famli_netapp_shared/C1_ML_Analysis/trained_models/cut/all2iq3_v1_epoch=23-val_loss=5.75.pt")
        # self.all2iq3.eval()

    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("Fetal EFW time aware Model")

        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument("--betas", type=float, nargs="+", default=(0.9, 0.999), help='Betas for Adam optimizer')
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=1e-5)
        group.add_argument('--huber_delta', help='Delta for Huber loss', type=float, default=0.5)
        
        # Image Encoder parameters                 
        group.add_argument("--features", type=int, default=1280, help='Number of output features for the encoder')
        group.add_argument("--time_dim_train", type=int, nargs="+", default=None, help='Range of time dimensions for training')
        group.add_argument("--n_chunks_e", type=int, default=2, help='Number of chunks in the encoder stage to reduce memory usage')
        group.add_argument("--n_chunks", type=int, default=16, help='Number of outputs in the time dimension, this will determine the first dimension of the 2D positional encoding')
        group.add_argument("--num_heads", type=int, default=8, help='Number of heads for multi_head attention')
        
        group.add_argument("--embed_dim", type=int, default=128, help='Embedding dimension')        
        group.add_argument("--dropout", type=float, default=0.1, help='Dropout rate')
        group.add_argument("--tags", type=int, default=18, help='Number of sweep tags for the sequences, this will determine the second dimension of the 2D positional encoding')
        group.add_argument("--loss_reg_weight", type=float, default=0.0, help='Weight for regularization loss')
        
        group.add_argument("--rho", type=float, nargs="+", default=(0.25, 0.75), help='Lower and upper bounds for score regularization')
        # group.add_argument("--lam_ent", type=float, default=1.0, help='Weight for entropy regularization loss, controls pushing scores toward 0 or 1')
        group.add_argument("--warmup_epochs", type=int, default=-1, help='Number of epochs to warmup the scores regularization')

        group.add_argument("--output_dim", type=int, default=1, help='Output dimension')

        return parent_parser
    
    def positional_encoding(self, seq_len: int, d_model: int, tag: int) -> torch.Tensor:
        """
        Sinusoidal positional encoding with tag-based offset.

        Args:
            seq_len (int): Sequence length.
            d_model (int): Embedding dimension.
            tag (int): Unique tag for the sequence.
            device (str): Device to store the tensor.

        Returns:
            torch.Tensor: Positional encoding (seq_len, d_model).
        """
        pe = torch.zeros(seq_len, d_model)
        
        # Offset positions by a tag-dependent amount to make each sequence encoding unique
        position = torch.arange(tag * seq_len, (tag + 1) * seq_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                betas=self.hparams.betas,
                                weight_decay=self.hparams.weight_decay)
        return optimizer
    
    def entropy_penalty(self, s, eps=1e-8):
        s = s.clamp(eps, 1 - eps)
        H = -(s * torch.log(s) + (1 - s) * torch.log(1 - s))
        return H.mean() 

    def regularizer(self, scores, rho0=0.25, rho1=0.75):

        loss_reg = (scores*(scores < rho0).float()).sum()/(scores < rho0).float().sum().clamp(min=1.0) + ((1.0 - scores)*(scores > rho1).float()).sum()/(scores > rho1).float().sum().clamp(min=1.0)

        return loss_reg

    def compute_loss(self, Y, X_hat, X_f_hat=None, X_s=None, X_s_ac=None, Y_s_ac=None, step="train", sync_dist=False):

        loss = self.loss_fn(Y, X_hat)

        self.log(f"{step}_loss", loss, sync_dist=sync_dist)
        
        l1 = self.l1_fn(X_hat, Y)
        self.log(f"{step}_l1", l1, sync_dist=sync_dist)

        if X_s is not None:
            
            self.log(f"{step}_scores/mean", X_s.mean(), sync_dist=sync_dist)
            self.log(f"{step}_scores/max", X_s.max(), sync_dist=sync_dist)
            self.log(f"{step}_scores/s>=0.9", (X_s >= 0.9).float().mean(), sync_dist=sync_dist)
            self.log(f"{step}_scores/s>=0.5", (X_s >= 0.5).float().mean(), sync_dist=sync_dist)            

            if self.hparams.warmup_epochs < 0 or self.current_epoch < self.hparams.warmup_epochs:
                reg_loss = 0
                loss_f = 0
            else:
                loss_f = 0
                reg_loss = self.regularizer(X_s.view(-1), rho0=self.hparams.rho[0], rho1=self.hparams.rho[1])*self.hparams.loss_reg_weight

                if step == "train":
                    loss = loss + reg_loss

                if( X_f_hat is not None):
                    loss_f = ((X_f_hat.squeeze(-1) - Y).abs()*(X_s > 0.8).float()).sum()/(X_s > 0.8).float().sum().clamp(min=1.0) 
                    if step == "train":
                        loss = loss + loss_f
            
            self.log(f"{step}_loss_f", loss_f, sync_dist=sync_dist)
            self.log(f"{step}_loss_reg", reg_loss, sync_dist=sync_dist)

            

        if X_s_ac is not None and Y_s_ac is not None:
            loss_ac = self.l1_fn(Y_s_ac, X_s_ac)
            self.log(f"{step}_loss_ac", loss_ac, sync_dist=sync_dist)
            self.log(f"{step}_scores_ac/mean", X_s_ac.mean(), sync_dist=sync_dist)
            self.log(f"{step}_scores_ac/max", X_s_ac.max(), sync_dist=sync_dist)
            self.log(f"{step}_scores_ac/s>=0.9", (X_s_ac >= 0.9).float().mean(), sync_dist=sync_dist)
            self.log(f"{step}_scores_ac/s>=0.5", (X_s_ac >= 0.5).float().mean(), sync_dist=sync_dist)
            if step == "train":
                loss = loss + loss_ac

        return loss
    
    def on_before_optimizer_step(self, optimizer):        
        norms = grad_norm(self.mha, norm_type=2)
        self.log_dict(norms)

    def training_step(self, train_batch, batch_idx):
        X = train_batch["img"]
        tags = train_batch["tag"]
        Y = train_batch["efw"]
        
        batch_size, NS, C, T, H, W = X.shape
        if self.hparams.time_dim_train is not None:
            time_r = torch.randint(low=self.hparams.time_dim_train[0], high=self.hparams.time_dim_train[1], size=(1,)).item()
            time_ridx = torch.randint(low=0, high=T, size=(time_r,))
            time_ridx = time_ridx.sort().values
            X = X[:, :, :, time_ridx, :, :].contiguous()            

        X = X.permute(0, 1, 3, 2, 4, 5)  # Shape is now [B, N, T, C, H, W]

        x_hat, x_f_hat, z_t_s, z_c_s = self(self.train_transform(X), tags)

        if 'img_ac' in train_batch:
            X_ac = train_batch["img_ac"]
            tags_ac = train_batch["tag_ac"]
            Y_ac = train_batch["score_ac"]
            
            X_ac = X_ac.unsqueeze(1).permute(0, 1, 3, 2, 4, 5)  # Shape is now [B, N, T, C, H, W]
            tags_ac = tags_ac.unsqueeze(1)

            _, _, z_t_s_ac, _ = self(self.train_transform(X_ac), tags_ac)            

            return self.compute_loss(Y=Y, X_hat=x_hat, X_f_hat=x_f_hat, X_s=z_t_s, X_s_ac=z_t_s_ac, Y_s_ac=Y_ac,step="train")
            

        return self.compute_loss(Y=Y, X_hat=x_hat, X_f_hat=x_f_hat, X_s=z_t_s, step="train")

    def validation_step(self, val_batch, batch_idx):
        
        X = val_batch["img"]
        tags = val_batch["tag"]
        Y = val_batch["efw"]

        X = X.permute(0, 1, 3, 2, 4, 5)  # Shape is now [B, N, T, C, H, W]

        x_hat, x_f_hat, z_t_s, z_c_s = self(X, tags)

        self.compute_loss(Y=Y, X_hat=x_hat, X_f_hat=x_f_hat, X_s=z_t_s, step="val", sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        X = test_batch["img"]
        tags = test_batch["tag"]
        Y = test_batch["efw"]

        X = X.permute(0, 1, 3, 2, 4, 5)  # Shape is now [B, N, T, C, H, W]

        x_hat, x_f_hat, z_t_s, z_c_s = self(X, tags)

        self.compute_loss(Y=Y, X_hat=x_hat, X_f_hat=x_f_hat, X_s=z_t_s, step="test", sync_dist=True)

    def encode(self, x: torch.Tensor, tag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        """
        Forwards an image through the spatial encoder, obtaining the latent

        Args:
            x: BxCx[SPATIAL DIMS] tensor

        """
        z_ = self.encoder(x)
        z_ = self.proj(z_) # [BS, T, self.hparams.embed_dim]
            
        z_t_, z_t_s_ = self.attn_chunk(z_) # [BS, self.hparams.n_chunks, self.hparams.embed_dim]

        p_enc_z = self.p_encoding_z[tag]
        
        z_t_ = self.dropout(z_t_)
        z_t_ = z_t_ + self.mha(self.ln0(z_t_ + p_enc_z)) #[BS, self.hparams.n_chunks, self.hparams.embed_dim]
        z_t_ = self.ln1(z_t_)

        return z_, z_t_, z_t_s_
    
    def predict(self, z_t: torch.Tensor) -> torch.Tensor:
        z_t, z_s = self.attn(z_t, z_t)
        x_hat = self.proj_final(z_t)
        return x_hat, z_s

    def forward(self, x_sweeps: torch.tensor, sweeps_tags: torch.tensor):
        
        batch_size = x_sweeps.shape[0]

        # x_sweeps shape is B, N, C, T, H, W. N for number of sweeps ex. torch.Size([2, 2, 200, 3, 256, 256]) 
        # tags shape torch.Size([2, 2])
        Nsweeps = x_sweeps.shape[1] # Number of sweeps -> T

        z = []
        z_t = []
        z_t_s = []

        for n in range(Nsweeps):

            x_sweeps_n = x_sweeps[:, n, :, :, :, :] # [BS, T, C, H, W]            
            tag = sweeps_tags[:,n]    

            z_, z_t_, z_t_s_ = self.encode(x_sweeps_n, tag) # [BS, T, self.hparams.features]

            z.append(z_)
            z_t.append(z_t_)
            z_t_s.append(z_t_s_)


        z = torch.stack(z, dim=1)  # [BS, N_sweeps, T, self.hparams.embed_dim]
        z_t = torch.stack(z_t, dim=1)  # [BS, N_sweeps, self.hparams.n_chunks, self.hparams.embed_dim]
        z_t_s = torch.stack(z_t_s, dim=1)  # [BS, N_sweeps, T, self.hparams.n_chunks]

        z = z.view(batch_size, -1, self.hparams.embed_dim)  # [BS, N_sweeps*T, self.hparams.embed_dim]
        z_t = z_t.view(batch_size, -1, self.hparams.embed_dim)  # [BS, N_s*n_chunks, self.hparams.embed_dim]
        z_t_s = z_t_s.view(batch_size, -1)  # [BS, N_s*n_chunks]

        x_hat, z_c_s = self.predict(z_t)
        x_f_hat = self.proj_final(z)

        return x_hat, x_f_hat, z_t_s, z_c_s
    

class OrdinalEMDLoss(nn.Module):
    def __init__(
        self,
        sigma=(0.12, 0.12, 0.12, 0.10, 0.07),
        bins=(0.0, 0.25, 0.5, 0.75, 1.0),
        class_weights=None,
        bin_weights=None,
        normalize_bin_weights: bool = True,
    ):
        super().__init__()

        self.register_buffer("bins", torch.tensor(bins, dtype=torch.float32))
        C = len(bins)

        # sigma: float OR sequence length C
        if isinstance(sigma, (float, int)):
            sigma_vec = torch.full((C,), float(sigma), dtype=torch.float32)
        else:
            sigma_vec = torch.tensor(list(sigma), dtype=torch.float32)
            if sigma_vec.numel() != C:
                raise ValueError(f"sigma must be float or length {C}, got length {sigma_vec.numel()}")
        self.register_buffer("sigma_vec", sigma_vec)

        if class_weights is not None:
            cw = torch.tensor(class_weights, dtype=torch.float32)
            if cw.numel() != C:
                raise ValueError(f"class_weights must be length {C}, got length {cw.numel()}")
            self.register_buffer("class_weights", cw)
        else:
            self.class_weights = None

        if bin_weights is not None:
            bw = torch.tensor(bin_weights, dtype=torch.float32)
            if bw.numel() != C:
                raise ValueError(f"bin_weights must be length {C}, got length {bw.numel()}")
            if normalize_bin_weights:
                bw = bw / (bw.mean() + 1e-8)
            self.register_buffer("bin_weights", bw)
        else:
            self.bin_weights = None

    def soft_targets(self, y: torch.Tensor, y_class: torch.Tensor | None = None):
        """
        y:       (m,) in [0,1]
        y_class: (m,) int 0..C-1 optional. If provided and sigma was per-class,
                 sigma is selected per-sample based on y_class.
                 If not provided, uses a single sigma (sigma_vec[0]) for all.
        returns: (m,C) soft target distribution
        """
        bins = self.bins  # (C,)

        if y_class is not None:
            # per-sample sigma picked by class
            sigma = self.sigma_vec[y_class].clamp_min(1e-6)  # (m,)
        else:
            # fallback: single sigma (works even if sigma_vec was per-class)
            sigma = self.sigma_vec[0].clamp_min(1e-6)        # scalar

        d2 = (bins[None, :] - y[:, None]) ** 2  # (m,C)
        # broadcast sigma: (m,1) or scalar
        p = torch.exp(-0.5 * d2 / (sigma[..., None] ** 2))
        return p / (p.sum(dim=1, keepdim=True) + 1e-8)

    def forward(self, logits, y, y_class=None, mask=None):
        """
        logits:  (B,N,C)
        y:       (B,N) or (B,N,1) in [0,1]
        y_class: (B,N) optional int 0..C-1
        mask:    (B,N) bool optional
        """
        if y.ndim == 3:
            y = y.squeeze(-1)

        B, N, C = logits.shape
        if C != self.bins.numel():
            raise ValueError(f"logits has C={C} but bins has {self.bins.numel()}")

        M = B * N
        logits_f = logits.reshape(M, C)
        y_f = y.reshape(M)

        if mask is not None:
            mask_f = mask.reshape(M)
            logits_f = logits_f[mask_f]
            y_f = y_f[mask_f]
            if y_class is not None:
                y_class = y_class.reshape(M)[mask_f]
        else:
            if y_class is not None:
                y_class = y_class.reshape(M)

        target = self.soft_targets(y_f, y_class=y_class)  # (m,C)
        p = F.softmax(logits_f, dim=1)                    # (m,C)
        cdf_p = torch.cumsum(p, dim=1)
        cdf_t = torch.cumsum(target, dim=1)

        diff = torch.abs(cdf_p - cdf_t)                   # (m,C)

        if self.bin_weights is not None:
            per = (diff * self.bin_weights).mean(dim=1)
        else:
            per = diff.mean(dim=1)

        if self.class_weights is not None and y_class is not None:
            per = per * self.class_weights[y_class]

        return per.mean()

    @torch.no_grad()
    def expected_score(self, logits):
        p = F.softmax(logits, dim=-1)
        return (p * self.bins).sum(dim=-1, keepdim=True)

def expected_score_from_logits(logits, bins):
    # logits: (B,T,C), bins: (C,)
    p = torch.softmax(logits, dim=-1)
    return (p * bins.view(1, 1, -1)).sum(dim=-1)  # (B,T)

def temporal_tv(x, power=1):
    # x: (B,T)
    d = x[:, 1:] - x[:, :-1]
    if power == 1:
        return d.abs().mean()
    return (d * d).mean()

def derivative_match_loss(s, y, power=1):
    # s: (B,T), y: (B,T)
    ds = s[:, 1:] - s[:, :-1]
    dy = y[:, 1:] - y[:, :-1]
    d = ds - dy
    return d.abs().mean() if power == 1 else (d * d).mean()

class EFWRopeEffnetV2s(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        encoder = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        encoder.classifier = nn.Linear(self.hparams.features, self.hparams.embed_dim)

        self.encoder = TimeDistributed(encoder)

        self.rope = RoPETransformerBlock(dim=self.hparams.embed_dim, num_heads=self.hparams.num_heads, mlp_ratio=self.hparams.mlp_ratio, dropout=self.hparams.dropout)
        self.norm = nn.LayerNorm(self.hparams.embed_dim)

        hidden = int(self.hparams.embed_dim*self.hparams.mlp_ratio)
        self.proj_s = nn.Sequential(
            nn.Linear(self.hparams.embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(hidden, 1)
        )
        self.proj = nn.Sequential(
            nn.Linear(self.hparams.embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(hidden, self.hparams.num_classes)
        )

        self.train_transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(180),
                v2.RandomResizedCrop(size=256, scale=(0.25, 1.0), ratio=(0.75, 1.3333333333333333)),
                v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
                v2.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2])
            ]
        )

        bin_weights = self.hparams.bin_weights if hasattr(self.hparams, 'bin_weights') else None
        self.loss_fn = OrdinalEMDLoss(sigma=self.hparams.sigma, bins=self.hparams.bins, class_weights=self.hparams.class_weights, bin_weights=bin_weights)


    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("BCE with Regression Model")

        group.add_argument("--lr", type=float, default=1e-3)
        group.add_argument("--betas", type=float, nargs="+", default=(0.9, 0.999), help='Betas for Adam optimizer')
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=1e-2)

        group.add_argument("--sigma", type=float, nargs="+", default=(0.050, 0.050, 0.055, 0.045, 0.040, 0.030, 0.030, 0.045, 0.060), help='Sigma for Ordinal EMD Loss')
        group.add_argument("--bins", type=float, nargs="+", default=(0.09614, 0.23614, 0.34, 0.45, 0.54, 0.615, 0.668, 0.73286, 0.88986), help='Bins for Ordinal EMD Loss')
        group.add_argument("--class_weights", type=float, nargs="+", default=None, help='Class weights for Ordinal EMD Loss')
        group.add_argument("--bin_weights", type=float, nargs="+", default=None, help='Bin weights for Ordinal EMD Loss')       

        group.add_argument("--num_classes", type=int, default=9, help='Output channels for projection head')
        
        # Image Encoder parameters         
        group.add_argument("--in_channels", type=int, default=3, help='Input channels for encoder')
        group.add_argument("--features", type=int, default=1280, help='Number of output features for the encoder')
        group.add_argument("--embed_dim", type=int, default=128, help='Embedding dimension for the model')
        group.add_argument("--num_heads", type=int, default=8, help='Number of heads for RoPE transformer')
        group.add_argument("--mlp_ratio", type=float, default=4.0, help='MLP ratio for RoPE transformer')
        group.add_argument("--dropout", type=float, default=0.2, help='Dropout rate for RoPE transformer')

        return parent_parser
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                betas=self.hparams.betas,
                                weight_decay=self.hparams.weight_decay)
        return optimizer
    
    def compute_loss(self, logits, Y, s_logit, Y_s, step="train", sync_dist=False):
        # logits: (B,T,C)
        # s_logit: (B,T,1)
        B, T, C = logits.shape

        # normalize target
        Y_g = Y.float().view(-1)
        Y_n = ((Y_g - 500.0) / 5000.0).clamp(0.0, 1.0)

        # ---- Top-K pooling by learned gate logits ----
        K = min(20, T)
        s_logit_2d = s_logit.squeeze(-1)                 # (B,T)
        top_s_logit, top_idx = s_logit_2d.topk(k=K, dim=1)  # (B,K)

        idx = top_idx[..., None].expand(B, K, C)         # (B,K,C)
        top_logits = logits.gather(dim=1, index=idx)     # (B,K,C)

        w = torch.softmax(top_s_logit, dim=1)            # (B,K)
        study_logit = (w[..., None] * top_logits).sum(dim=1)  # (B,C)

        # ---- Ordinal EMD ----
        loss_emd = self.loss_fn(study_logit[:, None, :], y=Y_n[:, None])

        # expected value
        p = torch.softmax(study_logit, dim=-1)
        bins = study_logit.new_tensor(self.hparams.bins)  # (C,) in [0,1]
        y_hat = (p * bins).sum(dim=-1)

        loss_s = F.smooth_l1_loss(y_hat, Y_n)

        # ---- Gate distillation (AC) on all frames ----
        Y_s = Y_s.float()
        pos = Y_s > 0.7
        neg = Y_s < 0.3
        mask = pos | neg
        y_gate = pos.float()

        loss_g_all = F.binary_cross_entropy_with_logits(s_logit_2d, y_gate, reduction="none")  # (B,T)
        if mask.any():
            loss_g = (loss_g_all * mask.float()).sum() / (mask.float().sum() + 1e-8)
        else:
            loss_g = loss_g_all.mean() * 0.0

        # ---- Optional budget (can be lighter with Top-K pooling) ----
        s = torch.sigmoid(s_logit_2d)
        target_mean = 0.2
        loss_budget = ((s.mean(dim=1) - target_mean) ** 2).mean()

        loss = loss_emd + 0.2 * loss_s + 0.1 * loss_g + 0.5 * loss_budget

        # metrics
        w_hat = 500.0 + 5000.0 * y_hat
        mae_g = torch.mean(torch.abs(w_hat - Y_g))

        self.log_dict(
            {
                f"{step}_loss": loss,
                f"{step}_emd": loss_emd,
                f"{step}_s": loss_s,
                f"{step}_g": loss_g,
                f"{step}_budget": loss_budget,
                f"{step}_mae_g": mae_g,
            },
            sync_dist=sync_dist,
            prog_bar=True,
        )
        return loss

    def training_step(self, train_batch, batch_idx):
        X = train_batch["img"]
        Y_s = train_batch["ac_scores"]
        Y = train_batch["efw"]

        X = X.permute(0, 1, 3, 2, 4, 5)  # B,N,C,T,H,W -> (B,N,T,C,H,W)
        B, N, T, C, H, W = X.shape        
        X = X.reshape(B, N*T, C, H, W).contiguous()  # (B, N*T,C,H,W)

        Y_s = Y_s.reshape(B, N*T).contiguous()  # (B, N*T)

        logits, s_logits = self(self.train_transform(X))

        return self.compute_loss(logits=logits, Y=Y, s_logit=s_logits, Y_s=Y_s, step="train")

    def validation_step(self, val_batch, batch_idx):
        X   = val_batch["img"]  # (N,B,C,T,H,W)
        Y_g = val_batch["efw"].float().view(-1)  # (B,)

        Y_n = ((Y_g - 500.0) / 5000.0).clamp(0.0, 1.0)

        logits_list = []
        s_logits_list = []

        # loop over sweeps N
        for x in X:  # x: (B,C,T,H,W)
            x = x.permute(0, 2, 1, 3, 4)   # (B,T,C,H,W)
            logit_bt, s_logit_bt = self(x) # (B,T,C), (B,T,1)
            logits_list.append(logit_bt)
            s_logits_list.append(s_logit_bt)

        logits = torch.cat(logits_list, dim=1)          # (B, M, C) where M=N*T
        s_logits = torch.cat(s_logits_list, dim=1)      # (B, M, 1)

        B, M, C = logits.shape

        # ---- Top-K pooling by learned gate ----
        K = min(20, M)
        s = torch.sigmoid(s_logits).squeeze(-1)         # (B,M)

        top_s, top_idx = s.topk(k=K, dim=1)             # (B,K)

        idx = top_idx[..., None].expand(B, K, C)        # (B,K,C)
        top_logits = logits.gather(dim=1, index=idx)    # (B,K,C)

        # weights: better to softmax over gate *logits*; but keeping your version is OK
        w = top_s / (top_s.sum(dim=1, keepdim=True) + 1e-8)  # (B,K)
        study_logit = (w[..., None] * top_logits).sum(dim=1) # (B,C)

        # ---- Expected value ----
        p = torch.softmax(study_logit, dim=-1)
        bins = study_logit.new_tensor(self.hparams.bins)      # (C,)
        y_hat = (p * bins).sum(dim=-1)                        # (B,)

        loss_s = F.smooth_l1_loss(y_hat, Y_n)

        w_hat = 500.0 + 5000.0 * y_hat
        mae_g = (w_hat - Y_g).abs().mean()
        bias_g = (w_hat - Y_g).mean()

        # ---- Optional: selection health metrics ----
        eff_frames = (1.0 / (w.pow(2).sum(dim=1) + 1e-8)).mean()

        self.log_dict(
            {
                "val_loss": loss_s,
                "val_mae_g": mae_g,
                "val_bias_g": bias_g,
                "val_eff_frames": eff_frames,  # should be ~K-ish if top-k dominates
            },
            prog_bar=True,
            sync_dist=True,
        )

    def test_step(self, test_batch, batch_idx):
        
        X   = test_batch["img"]  # (N,B,C,T,H,W)
        Y_g = test_batch["efw"].float().view(-1)  # (B,)

        Y_n = ((Y_g - 500.0) / 5000.0).clamp(0.0, 1.0)

        logits_list = []
        s_logits_list = []

        # loop over sweeps N
        for x in X:  # x: (B,C,T,H,W)
            x = x.permute(0, 2, 1, 3, 4)   # (B,T,C,H,W)
            logit_bt, s_logit_bt = self(x) # (B,T,C), (B,T,1)
            logits_list.append(logit_bt)
            s_logits_list.append(s_logit_bt)

        logits = torch.cat(logits_list, dim=1)          # (B, M, C) where M=N*T
        s_logits = torch.cat(s_logits_list, dim=1)      # (B, M, 1)

        B, M, C = logits.shape

        # ---- Top-K pooling by learned gate ----
        K = min(20, M)
        s = torch.sigmoid(s_logits).squeeze(-1)         # (B,M)

        top_s, top_idx = s.topk(k=K, dim=1)             # (B,K)

        idx = top_idx[..., None].expand(B, K, C)        # (B,K,C)
        top_logits = logits.gather(dim=1, index=idx)    # (B,K,C)

        # weights: better to softmax over gate *logits*; but keeping your version is OK
        w = top_s / (top_s.sum(dim=1, keepdim=True) + 1e-8)  # (B,K)
        study_logit = (w[..., None] * top_logits).sum(dim=1) # (B,C)

        # ---- Expected value ----
        p = torch.softmax(study_logit, dim=-1)
        bins = study_logit.new_tensor(self.hparams.bins)      # (C,)
        y_hat = (p * bins).sum(dim=-1)                        # (B,)

        loss_s = F.smooth_l1_loss(y_hat, Y_n)

        w_hat = 500.0 + 5000.0 * y_hat
        mae_g = (w_hat - Y_g).abs().mean()
        bias_g = (w_hat - Y_g).mean()

        # ---- Optional: selection health metrics ----
        eff_frames = (1.0 / (w.pow(2).sum(dim=1) + 1e-8)).mean()

        self.log_dict(
            {
                "test_loss": loss_s,
                "test_mae_g": mae_g,
                "test_bias_g": bias_g,
                "test_eff_frames": eff_frames,  # should be ~K-ish if top-k dominates
            }
        )
        

    def forward(self, x: torch.tensor):

        z = self.encoder(x)
        z = self.rope(z)
        z = self.norm(z)

        s_logit = self.proj_s(z)
        logits = self.proj(z)        

        return logits, s_logit #[B, N, C], [B, N, 1]