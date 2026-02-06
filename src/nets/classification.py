from json import encoder
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from .layers import TimeDistributed, AttentionChunk, MHABlock, SelfAttention, ProjectionHead, RoPETransformerBlock, TemporalRefinerTCN

import torchvision 
from torchvision.transforms import v2


from lightning.pytorch import LightningModule

from lightning.pytorch.loggers import NeptuneLogger
from neptune.types import File

import torchmetrics
from torchmetrics import Accuracy, MeanAbsoluteError, MeanSquaredError, R2Score, PearsonCorrCoef
from torchmetrics.aggregation import CatMetric
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt

import os
import json
import math

import numpy as np
import itertools

def plot_confusion_matrix(cm, classes, title='Confusion matrix',cmap=plt.cm.Blues, experiment=None, out_path=None):
    #This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
    fig_cm = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()    

    if experiment:
        experiment.upload(fig_cm)
    if out_path:
        plt.savefig(out_path)
    plt.close(fig_cm)

def compute_classification_report(targets, probs, experiment=None, out_path=None):
    report_dict = classification_report(targets, probs, digits=3, output_dict=True)
    report_txt = classification_report(targets, probs, digits=3)
    print(report_txt)

    if experiment:
        experiment.upload(
            File.from_content(report_txt, extension="txt")
        )        
    if(out_path):
        with open(out_path, "+w") as f:
            json.dump(report_dict, f, indent=4)

class GaussianNoise(nn.Module):    
    def __init__(self, mean=0.0, std=0.1):
        super(GaussianNoise, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    def forward(self, x):
        return x + torch.normal(mean=self.mean, std=self.std, size=x.size(), device=x.device)

class EfficientNet(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        NN = getattr(torchvision.models, self.hparams.base_encoder)
        self.convnet = NN(num_classes=self.hparams.num_classes)

        self.extract_features = False

        if hasattr(self.hparams, 'model_feat') and self.hparams.model_feat is not None:
            classifier = self.convnet.classifier
            self.convnet.classifier = nn.Identity()
            self.convnet.load_state_dict(torch.load(args.model_feat))
            # for param in self.convnet.parameters():
            #     param.requires_grad = False
            self.convnet.classifier = classifier


        class_weights = None
        if(hasattr(self.hparams, 'class_weights')):
            class_weights = torch.tensor(self.hparams.class_weights).to(torch.float32)

        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)

        self.softmax = nn.Softmax(dim=1)

        self.noise_transform = torch.nn.Sequential(
            GaussianNoise(std=0.05)
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        
        x = self(self.noise_transform(x))

        loss = self.loss(x, y)
        
        self.log('train_loss', loss)

        self.accuracy(x, y)
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        
        x = self(x)
        
        loss = self.loss(x, y)
        
        self.log('val_loss', loss, sync_dist=True)
        self.accuracy(x, y)
        self.log("val_acc", self.accuracy)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        
        x = self(x)
        
        loss = self.loss(x, y)
        
        self.log('test_loss', loss, sync_dist=True)
        self.accuracy(x, y)
        self.log("test_acc", self.accuracy)

    def forward(self, x):        
        if self.extract_features:
            x_f = self.convnet.features(x)
            x_f = self.convnet.avgpool(x_f)            
            x = torch.flatten(x_f, 1)            
            return self.convnet.classifier(x), x_f
        else:
            return self.convnet(x)


class EffnetV2s(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        self.model.classifier = nn.Identity()
        self.proj_final = ProjectionHead(input_dim=self.hparams.features, hidden_dim=self.hparams.features, output_dim=self.hparams.num_classes, activation=nn.LeakyReLU)

        weights = None if self.hparams.class_weights is None else torch.tensor(self.hparams.class_weights, dtype=torch.float32)
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)
        self.accuracy = Accuracy(task='multiclass', num_classes=self.hparams.num_classes)
        self.conf = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=self.hparams.num_classes, normalize='true')

        self.class_names = self.hparams.class_names if hasattr(self.hparams, 'class_names') and self.hparams.class_names else range(self.hparams.num_classes)

        self.probs = CatMetric()
        self.targets = CatMetric()

    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("Simple Classification Model")

        group.add_argument("--lr", type=float, default=1e-3)
        group.add_argument("--betas", type=float, nargs="+", default=(0.9, 0.999), help='Betas for Adam optimizer')
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
        # Image Encoder parameters 
        group.add_argument("--spatial_dims", type=int, default=2, help='Spatial dimensions for the encoder')
        group.add_argument("--in_channels", type=int, default=3, help='Input channels for encoder')
        group.add_argument("--features", type=int, default=1280, help='Number of output features for the encoder')

        group.add_argument("--num_classes", type=int, default=3, help='Number of output classes for the model')        
        group.add_argument("--class_names", type=str, default=None, help='Class names')        
        group.add_argument("--class_weights", nargs="+", default=None, type=float, help='Class weights for the loss function')
        

        return parent_parser
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                betas=self.hparams.betas,
                                weight_decay=self.hparams.weight_decay)
        return optimizer
    
    def compute_loss(self, Y, X_hat, step="train", sync_dist=False):

        loss = self.loss_fn(X_hat, Y)
        
        self.log(f"{step}_loss", loss, sync_dist=sync_dist)

        batch_size = Y.size(0)
        self.accuracy(X_hat, Y)
        self.log(f"{step}_acc", self.accuracy, batch_size=batch_size, sync_dist=sync_dist)

        return loss

    def training_step(self, train_batch, batch_idx):
        X = train_batch["img"]
        Y = train_batch["class"]

        x_hat = self(X)

        return self.compute_loss(Y=Y, X_hat=x_hat, step="train")

    def validation_step(self, val_batch, batch_idx):
        
        X = val_batch["img"]
        Y = val_batch["class"]

        x_hat = self(X) 

        self.compute_loss(Y=Y, X_hat=x_hat, step="val", sync_dist=True)
        
        self.probs.update(x_hat.softmax(dim=-1))
        self.targets.update(Y)
        self.conf.update(x_hat, Y)

    
    def on_validation_epoch_end(self):
        
        probs = self.probs.compute()
        targets = self.targets.compute()
        confmat  = self.conf.compute()

        if self.trainer.is_global_zero:
            
            logger = self.trainer.logger
            run = logger.experiment  
            
            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["images/val_confusion_matrix"]
                
            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "val_confusion_matrix.png")
                    
            plot_confusion_matrix(confmat.cpu().numpy(), self.class_names, experiment=experiment, out_path=out_path)

            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["reports/val_classification_report"]
                
            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "val_classification_report.json")
                    
            compute_classification_report(targets.cpu().numpy(), probs.argmax(dim=-1).cpu().numpy(), experiment=experiment, out_path=out_path)

        self.probs.reset()
        self.targets.reset()
        self.conf.reset()

    def test_step(self, test_batch, batch_idx):
        X = test_batch["img"]
        Y = test_batch["class"]

        x_hat = self(X)

        self.conf.update(x_hat, Y)

        self.probs.update(x_hat.softmax(dim=-1))
        self.targets.update(Y)

    def on_test_epoch_end(self):

        confmat  = self.conf.compute()
        probs = self.probs.compute()
        targets = self.targets.compute()

        if self.trainer.is_global_zero:

            logger = self.trainer.logger
            run = logger.experiment  
            
            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["images/test_confusion_matrix"]
                
            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "test_confusion_matrix.png")
                    
            plot_confusion_matrix(confmat.cpu().numpy(), self.class_names, experiment=experiment, out_path=out_path)

            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["reports/test_classification_report"]
                
            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "test_classification_report.json")
                    
            compute_classification_report(targets.cpu().numpy(), probs.argmax(dim=-1).cpu().numpy(), experiment=experiment, out_path=out_path)

    def forward(self, x: torch.tensor):

        z = self.model(x)
        return self.proj_final(z)
    
class EffnetV2sSoft(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        self.model.classifier = nn.Identity()
        self.proj_final = ProjectionHead(input_dim=self.hparams.features, hidden_dim=self.hparams.features, output_dim=self.hparams.num_classes, activation=nn.LeakyReLU)

        weights = None if self.hparams.class_weights is None else torch.tensor(self.hparams.class_weights, dtype=torch.float32)
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)
        self.accuracy = Accuracy(task='multiclass', num_classes=self.hparams.num_classes)
        self.conf = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=self.hparams.num_classes, normalize='true')

        self.class_names = self.hparams.class_names if hasattr(self.hparams, 'class_names') and self.hparams.class_names else range(self.hparams.num_classes)

        self.probs = CatMetric()
        self.targets = CatMetric()

    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("Simple Classification Model")

        group.add_argument("--lr", type=float, default=1e-3)
        group.add_argument("--betas", type=float, nargs="+", default=(0.9, 0.999), help='Betas for Adam optimizer')
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
        # Image Encoder parameters 
        group.add_argument("--spatial_dims", type=int, default=2, help='Spatial dimensions for the encoder')
        group.add_argument("--in_channels", type=int, default=3, help='Input channels for encoder')
        group.add_argument("--features", type=int, default=1280, help='Number of output features for the encoder')

        group.add_argument("--num_classes", type=int, default=3, help='Number of output classes for the model')        
        group.add_argument("--class_names", type=str, default=None, help='Class names')        
        group.add_argument("--class_weights", nargs="+", default=None, type=float, help='Class weights for the loss function')
        

        return parent_parser
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                betas=self.hparams.betas,
                                weight_decay=self.hparams.weight_decay)
        return optimizer
    
    def compute_loss(self, Y, X_hat, step="train", sync_dist=False):

        loss = self.loss_fn(X_hat, Y)
        
        self.log(f"{step}_loss", loss, sync_dist=sync_dist)

        batch_size = Y.size(0)
        self.accuracy(X_hat, Y)
        self.log(f"{step}_acc", self.accuracy, batch_size=batch_size, sync_dist=sync_dist)

        return loss

    def training_step(self, train_batch, batch_idx):
        X = train_batch["img"]
        Y = train_batch["scalar"]

        x_hat = self(X)

        return self.compute_loss(Y=Y, X_hat=x_hat, step="train")

    def validation_step(self, val_batch, batch_idx):
        
        X = val_batch["img"]
        Y = val_batch["scalar"]

        x_hat = self(X) 

        self.compute_loss(Y=Y, X_hat=x_hat, step="val", sync_dist=True)
        
        self.probs.update(x_hat.softmax(dim=-1))
        self.targets.update(Y)
        self.conf.update(x_hat, Y.argmax(dim=-1))

    
    def on_validation_epoch_end(self):
        
        probs = self.probs.compute()
        targets = self.targets.compute()
        confmat  = self.conf.compute()

        if self.trainer.is_global_zero:
            
            logger = self.trainer.logger
            run = logger.experiment  
            
            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["images/val_confusion_matrix"]
                
            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "val_confusion_matrix.png")
                    
            plot_confusion_matrix(confmat.cpu().numpy(), self.class_names, experiment=experiment, out_path=out_path)

            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["reports/val_classification_report"]
                
            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "val_classification_report.json")
                    
            compute_classification_report(targets.argmax(dim=-1).cpu().numpy(), probs.argmax(dim=-1).cpu().numpy(), experiment=experiment, out_path=out_path)

        self.probs.reset()
        self.targets.reset()
        self.conf.reset()

    def test_step(self, test_batch, batch_idx):
        X = test_batch["img"]
        Y = test_batch["scalar"]

        x_hat = self(X)

        self.conf.update(x_hat, Y.argmax(dim=-1))

        self.probs.update(x_hat.softmax(dim=-1))
        self.targets.update(Y)

    def on_test_epoch_end(self):

        confmat  = self.conf.compute()
        probs = self.probs.compute()
        targets = self.targets.compute()

        if self.trainer.is_global_zero:

            logger = self.trainer.logger
            run = logger.experiment  
            
            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["images/test_confusion_matrix"]
                
            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "test_confusion_matrix.png")
                    
            plot_confusion_matrix(confmat.cpu().numpy(), self.class_names, experiment=experiment, out_path=out_path)

            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["reports/test_classification_report"]
                
            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "test_classification_report.json")
                    
            compute_classification_report(targets.argmax(dim=-1).cpu().numpy(), probs.argmax(dim=-1).cpu().numpy(), experiment=experiment, out_path=out_path)

    def forward(self, x: torch.tensor):

        z = self.model(x)
        return self.proj_final(z)

class EffnetV2sTD(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        encoder = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        encoder.classifier = nn.Linear(self.hparams.features, self.hparams.embed_dim)

        self.encoder = TimeDistributed(encoder)
        
        self.proj = nn.Linear(self.hparams.embed_dim, self.hparams.num_classes)

        self.train_transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(180),
                v2.RandomResizedCrop(size=256, scale=(0.25, 1.0), ratio=(0.75, 1.3333333333333333)),
                v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
                v2.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2])
            ]
        )
        
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(self.hparams.class_weights, dtype=torch.float32))

        self.accuracy = Accuracy(task='multiclass', num_classes=self.hparams.num_classes)
        self.conf = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=self.hparams.num_classes, normalize='true')

        self.probs = CatMetric()
        self.targets = CatMetric()


    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("BCE with Regression Model")

        group.add_argument("--lr", type=float, default=1e-3)
        group.add_argument("--betas", type=float, nargs="+", default=(0.9, 0.999), help='Betas for Adam optimizer')
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=1e-2)
        
        # Image Encoder parameters 
        group.add_argument("--spatial_dims", type=int, default=2, help='Spatial dimensions for the encoder')
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
    
    def compute_loss(self, Y_c, X_hat, step="train", sync_dist=False):

        B, T, C = X_hat.shape
        X_hat = X_hat.view(B*T, C)
        Y_c = Y_c.view(B*T)
        
        loss = self.loss_fn(X_hat, Y_c)

        self.log(f"{step}_loss", loss, sync_dist=sync_dist)

        return loss

    def training_step(self, train_batch, batch_idx):
        X = train_batch["img"]
        Y_c = train_batch["class"]

        X = X.permute(0, 2, 1, 3, 4)  # Shape is now [B, T, C, H, W]

        x_hat = self(self.train_transform(X))

        return self.compute_loss(Y_c=Y_c, X_hat=x_hat, step="train")

    def validation_step(self, val_batch, batch_idx):
        X = val_batch["img"]       # (B,C,T,H,W) presumably
        Y_c = val_batch["class"]   # (B,T) int 0..C-1

        X = X.permute(0, 2, 1, 3, 4)  # -> (B,T,C,H,W)
        logits = self(X)             # (B,T,C)

        self.compute_loss(Y_c=Y_c, X_hat=logits, step="val", sync_dist=True)

        # for cm
        logits_f = logits.reshape(-1, self.hparams.num_classes)
        y_f = Y_c.reshape(-1)

        self.probs.update(logits_f.softmax(dim=-1))
        self.targets.update(y_f)
        self.conf.update(logits_f, y_f)
        
    def on_validation_epoch_end(self):

        # confusion matrix/report 
        probs = self.probs.compute()
        targets = self.targets.compute()
        confmat = self.conf.compute()

        if self.trainer.is_global_zero:
            logger = self.trainer.logger
            run = logger.experiment

            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["images/val_confusion_matrix"]

            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "val_confusion_matrix.png")

            plot_confusion_matrix(
                confmat.cpu().numpy(),
                np.arange(self.hparams.num_classes),
                experiment=experiment,
                out_path=out_path
            )

            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["reports/val_classification_report"]

            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "val_classification_report.json")

            compute_classification_report(
                targets.cpu().numpy(),
                probs.argmax(dim=-1).cpu().numpy(),
                experiment=experiment,
                out_path=out_path
            )

        self.probs.reset()
        self.targets.reset()
        self.conf.reset()



    def test_step(self, test_batch, batch_idx):
        X = test_batch["img"]
        Y_s = test_batch["scalar"]
        Y_c = test_batch["class"]

        X = X.permute(0, 2, 1, 3, 4)  # Shape is now [B, T, C, H, W]

        x_hat = self(X)

        self.compute_loss(Y_s=Y_s, Y_c=Y_c, X_hat=x_hat, step="test", sync_dist=True)
        
        Y_c = Y_c.view(-1)
        x_hat = x_hat.view(-1, self.hparams.num_classes)
        
        self.probs.update(x_hat.softmax(dim=-1))
        self.targets.update(Y_c)
        self.conf.update(x_hat, Y_c)

    def on_test_epoch_end(self):
        probs = self.probs.compute()
        targets = self.targets.compute()
        confmat  = self.conf.compute()

        if self.trainer.is_global_zero:
            logger = self.trainer.logger
            run = logger.experiment  
            
            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["images/test_confusion_matrix"]
                
            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "test_confusion_matrix.png")
                    
            plot_confusion_matrix(confmat.cpu().numpy(), np.arange(self.hparams.num_classes), experiment=experiment, out_path=out_path)

            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["reports/test_classification_report"]
                
            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "test_classification_report.json")
                    
            compute_classification_report(targets.cpu().numpy(), probs.argmax(dim=-1).cpu().numpy(), experiment=experiment, out_path=out_path)

        self.probs.reset()
        self.targets.reset()
        self.conf.reset()
        

    def forward(self, x: torch.tensor):

        z = self.encoder(x)
        return self.proj(z)

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

class RopeEffnetV2s(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        encoder = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        encoder.classifier = nn.Linear(self.hparams.features, self.hparams.embed_dim)

        self.encoder = TimeDistributed(encoder)

        self.rope = RoPETransformerBlock(dim=self.hparams.embed_dim, num_heads=self.hparams.num_heads, mlp_ratio=self.hparams.mlp_ratio, dropout=self.hparams.dropout)
        self.norm = nn.LayerNorm(self.hparams.embed_dim)        
        self.proj = nn.Linear(self.hparams.embed_dim, self.hparams.num_classes)

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
        

        self.accuracy = Accuracy(task='multiclass', num_classes=self.hparams.num_classes)
        self.conf = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=self.hparams.num_classes, normalize='true')

        self.probs = CatMetric()
        self.targets = CatMetric()

        # Recall for class 4 (multiclass)
        self.val_recall = torchmetrics.classification.MulticlassRecall(
            num_classes=self.hparams.num_classes,
            average=None,        # returns per-class recall
        )

        # measurable detection: target = 1 if true in {3,4}, pred = 1 if score_pred >= 0.75
        self.val_meas_stats = torchmetrics.classification.BinaryStatScores(threshold=0.5)  
        # (threshold irrelevant since we pass int preds, but torchmetrics requires one)

        # reject false positives at two thresholds (0.75 and 0.9)
        self.val_reject_fp75 = torchmetrics.classification.BinaryStatScores(threshold=0.75)
        self.val_reject_fp90 = torchmetrics.classification.BinaryStatScores(threshold=0.90)
        # returns (tp, fp, tn, fn) aggregated across GPUs


    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("BCE with Regression Model")

        group.add_argument("--lr", type=float, default=1e-3)
        group.add_argument("--betas", type=float, nargs="+", default=(0.9, 0.999), help='Betas for Adam optimizer')
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=1e-2)

        group.add_argument("--sigma", type=float, nargs="+", default=(0.18, 0.12, 0.12, 0.10, 0.07), help='Sigma for Ordinal EMD Loss')
        group.add_argument("--bins", type=float, nargs="+", default=(0.0, 0.25, 0.5, 0.75, 1.0), help='Bins for Ordinal EMD Loss')
        group.add_argument("--class_weights", type=float, nargs="+", default=[0.03603907, 0.14391553, 0.85467111, 1.73506923, 2.23030506], help='Class weights for Ordinal EMD Loss')
        group.add_argument("--bin_weights", type=float, nargs="+", default=[0.23809524, 0.47619048, 0.71428571, 1.19047619, 2.38095238], help='Bin weights for Ordinal EMD Loss')
        group.add_argument("--num_classes", type=int, default=5, help='Output channels for projection head')

        group.add_argument("--top_aux_weight", type=float, default=0.0, help='Weight for auxiliary loss on top class')
        group.add_argument("--top_pos_weight", type=float, default=7.0, help='Positive weight for auxiliary loss on top class')
        group.add_argument("--top_aux_warmup_steps", type=int, default=2000, help='Number of warmup steps for auxiliary loss on top class')

        group.add_argument("--reject_tail_weight", type=float, default=0.00, help='Weight for reject tail penalty')
        group.add_argument("--reject_tau", type=float, default=0.85, help='Reject tail threshold')

        group.add_argument("--temporal_score_tv_power", type=int, default=1, help='Power for temporal total variation regularization on expected score')
        group.add_argument("--temporal_score_tv_weight", type=float, default=0.00, help='Weight for temporal total variation regularization on expected score')

        group.add_argument("--temporal_derivative_weight", type=float, default=0.00, help='Weight for temporal derivative matching loss on expected score')
        group.add_argument("--temporal_derivative_warmup_steps", nargs="+", type=int, default=[0, 200], help='Number of warmup steps for temporal derivative matching loss on expected score')
        group.add_argument("--temporal_derivative_power", type=int, default=1, help='Power for temporal derivative matching loss on expected score')
        
        group.add_argument("--meas_thresh", type=float, default=0.75)
        group.add_argument("--earlystop_fp_lambda", type=float, default=0.25)
        group.add_argument("--earlystop_fp_tail_lambda", type=float, default=1.0)
        
        # Image Encoder parameters 
        group.add_argument("--spatial_dims", type=int, default=2, help='Spatial dimensions for the encoder')
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
    
    def compute_loss(self, Y_s, X_hat, Y_c=None, step="train", sync_dist=False):

        # print("Y unique:", torch.unique(Y))
        # print("Y min/max:", Y.min().item(), Y.max().item())
        # print("Number of zeros in Y:", (Y == 0).sum().item())

        # print("X_hat min/max:", X_hat.min().item(), X_hat.max().item())
        # print("Number of zeros in X_hat:", (X_hat == 0).sum().item())
        
        loss = self.loss_fn(X_hat, y=Y_s.float(), y_class=Y_c)

        C = self.hparams.num_classes
        logits = X_hat
        logits_f = logits.reshape(-1, C)

        if(Y_c is not None):
            y_class = Y_c
            y_class_f = y_class.reshape(-1)

        if self.hparams.top_aux_weight > 0.0 and Y_c is not None and step == "train":

            is_top = (y_class_f == (C - 1)).float()

            pos_w = logits_f.new_tensor(float(self.hparams.top_pos_weight))

            logit_top = logits_f[:, -1]
            logit_rest = torch.logsumexp(logits_f[:, :-1], dim=1)
            margin = logit_top - logit_rest
            aux = F.binary_cross_entropy_with_logits(margin, is_top, reduction="none", pos_weight=pos_w)           

            if self.hparams.top_aux_warmup_steps > 0:
                warm = min(1.0, float(self.global_step) / float(self.hparams.top_aux_warmup_steps))
                aux_w = self.hparams.top_aux_weight * warm
            else:
                aux_w = self.hparams.top_aux_weight

            aux_mean = aux.mean() * aux_w

            loss = loss + aux_mean
            self.log(f"{step}_aux_loss", aux_mean, sync_dist=sync_dist)

            with torch.no_grad():
                self.log(f"{step}_top_rate", is_top.mean(), sync_dist=sync_dist)
                self.log(f"{step}_margin_mean", margin.mean(), sync_dist=sync_dist)
                if is_top.bool().any():
                    self.log(f"{step}_margin_pos_mean", margin[is_top.bool()].mean(), sync_dist=sync_dist)
                self.log(f"{step}_margin_neg_mean", margin[~is_top.bool()].mean(), sync_dist=sync_dist)

        if self.hparams.reject_tail_weight > 0.0 and Y_c is not None and step == "train":

            # Reject tail penalty
            # after logits_f and y_class_f
            bins = logits_f.new_tensor(self.hparams.bins)
            p = torch.softmax(logits_f, dim=1)
            score_pred_f = (p * bins).sum(dim=1)

            is_reject = (y_class_f == 0)

            reject_tail = logits_f.new_zeros(())
            if is_reject.any():
                reject_tail = torch.relu(score_pred_f[is_reject] - self.hparams.reject_tau).mean()
                loss = loss + self.hparams.reject_tail_weight * reject_tail

            self.log(f"{step}_reject_tail", reject_tail, sync_dist=sync_dist)

        if getattr(self.hparams, "temporal_score_tv_weight", 0.0) > 0 and step == "train":
            # s = expected_score_from_logits(logits, self.loss_fn.bins)  # (B,T)
            tv_s = temporal_tv(logits, power=getattr(self.hparams, "temporal_score_tv_power", 1))
            loss = loss + self.hparams.temporal_score_tv_weight * tv_s
            self.log(f"{step}_tv_score", tv_s, sync_dist=sync_dist)  

        if getattr(self.hparams, "temporal_derivative_weight", 0.0) > 0 and step == "train":
            s = expected_score_from_logits(logits, self.loss_fn.bins)  # (B,T)
            warm = 0.0
            if self.hparams.temporal_derivative_warmup_steps[0] <= self.global_step:
                warm = min(1.0, float(self.global_step - self.hparams.temporal_derivative_warmup_steps[0]) / float(self.hparams.temporal_derivative_warmup_steps[1] - self.hparams.temporal_derivative_warmup_steps[0]))
            dml = warm * self.hparams.temporal_derivative_weight * derivative_match_loss(s, Y_s.float(), power=getattr(self.hparams, "temporal_derivative_power", 1))            
            loss = loss + dml
            self.log(f"{step}_derivative_match_loss", dml, sync_dist=sync_dist)    

        self.log(f"{step}_loss", loss, sync_dist=sync_dist)

        return loss

    def training_step(self, train_batch, batch_idx):
        X = train_batch["img"]
        Y_s = train_batch["scalar"]
        Y_c = train_batch["class"]

        X = X.permute(0, 2, 1, 3, 4)  # Shape is now [B, T, C, H, W]

        x_hat = self(self.train_transform(X))

        return self.compute_loss(Y_s=Y_s, Y_c=Y_c, X_hat=x_hat, step="train")

    def validation_step(self, val_batch, batch_idx):
        X = val_batch["img"]       # (B,C,T,H,W) presumably
        Y_s = val_batch["scalar"]  # (B,T) or (B,N) float scores
        Y_c = val_batch["class"]   # (B,T) int 0..C-1

        X = X.permute(0, 2, 1, 3, 4)  # -> (B,T,C,H,W)
        logits = self(X)             # (B,T,C)

        bins = logits.new_tensor(self.hparams.bins)  # (C,)

        # expected score in [0,1]
        p = torch.softmax(logits, dim=-1)
        score_pred = (p * bins).sum(dim=-1)          # (B,T)

        # nearest-bin predicted class
        pred_cls = torch.argmin(torch.abs(score_pred[..., None] - bins), dim=-1)  # (B,T)

        # true class: use Y_c directly (safer than Y_s->bins mapping)
        true_cls = Y_c  # (B,T)

        # flatten for metrics
        pred_cls_f = pred_cls.reshape(-1)
        true_cls_f = true_cls.reshape(-1)
        score_pred_f = score_pred.reshape(-1)

        # -------------------------
        # (A) measurable recall
        # -------------------------
        is_meas = (true_cls_f >= 3).int()  # classes 3,4
        pred_meas = (score_pred_f >= self.hparams.meas_thresh).int()
        self.val_meas_stats.update(pred_meas, is_meas)

        # -------------------------
        # (B) reject FP at 0.75/0.9
        # target_bin: 0=reject, 1=non-reject
        # pred_pos: predicted non-reject by thresholding score_pred
        # FP/(FP+TN) computed at epoch end is P(pred_pos=1 | target_bin=0)
        # -------------------------
        target_nonreject = (true_cls_f != 0).int()
        self.val_reject_fp75.update((score_pred_f >= 0.75).int(), target_nonreject)
        self.val_reject_fp90.update((score_pred_f >= 0.90).int(), target_nonreject)

        

        # for cm
        logits_f = logits.reshape(-1, self.hparams.num_classes)
        y_f = Y_c.reshape(-1)

        self.probs.update(logits_f.softmax(dim=-1))
        self.targets.update(y_f)
        self.conf.update(logits_f, y_f)

    def _unpack_bss(self, bss):
        """
        Torchmetrics BinaryStatScores compute() can return:
        - tuple(tp, fp, tn, fn)  (older)
        - tensor([tp, fp, tn, fn, sup]) or shape (...,5) (newer)
        Returns: tp, fp, tn, fn as scalars/tensors
        """
        if isinstance(bss, (tuple, list)):
            tp, fp, tn, fn = bss[:4]
            return tp, fp, tn, fn

        # tensor output
        # last dim is [tp, fp, tn, fn, sup]
        tp = bss[..., 0]
        fp = bss[..., 1]
        tn = bss[..., 2]
        fn = bss[..., 3]
        return tp, fp, tn, fn
        
    def on_validation_epoch_end(self):
        # measurable recall = TP / (TP + FN)
        bss = self.val_meas_stats.compute()
        tp, fp, tn, fn = self._unpack_bss(bss)
        recall_meas = tp / (tp + fn + 1e-8)

        # reject FP@0.75 = FP / (FP + TN) in the nonreject-vs-reject binary framing
        bss75 = self.val_reject_fp75.compute()
        tp, fp, tn, fn = self._unpack_bss(bss75)
        fp_reject_075 = fp / (fp + tn + 1e-8)

        bss90 = self.val_reject_fp90.compute()
        tp, fp, tn, fn = self._unpack_bss(bss90)
        fp_reject_090 = fp / (fp + tn + 1e-8)

        lam = getattr(self.hparams, "earlystop_fp_lambda", 0.25)
        mu  = getattr(self.hparams, "earlystop_fp_tail_lambda", 1.0)

        val_select = recall_meas - lam * fp_reject_075 - mu * fp_reject_090

        self.log("val_recall_meas", recall_meas, prog_bar=True)
        self.log("val_fp_reject_075", fp_reject_075, prog_bar=True)
        self.log("val_fp_reject_090", fp_reject_090, prog_bar=True)
        self.log("val_select", val_select, prog_bar=True)

        self.val_meas_stats.reset()
        self.val_reject_fp75.reset()
        self.val_reject_fp90.reset()

        # confusion matrix/report 
        probs = self.probs.compute()
        targets = self.targets.compute()
        confmat = self.conf.compute()

        if self.trainer.is_global_zero:
            logger = self.trainer.logger
            run = logger.experiment

            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["images/val_confusion_matrix"]

            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "val_confusion_matrix.png")

            plot_confusion_matrix(
                confmat.cpu().numpy(),
                np.arange(self.hparams.num_classes),
                experiment=experiment,
                out_path=out_path
            )

            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["reports/val_classification_report"]

            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "val_classification_report.json")

            compute_classification_report(
                targets.cpu().numpy(),
                probs.argmax(dim=-1).cpu().numpy(),
                experiment=experiment,
                out_path=out_path
            )

        self.probs.reset()
        self.targets.reset()
        self.conf.reset()



    def test_step(self, test_batch, batch_idx):
        X = test_batch["img"]
        Y_s = test_batch["scalar"]
        Y_c = test_batch["class"]

        X = X.permute(0, 2, 1, 3, 4)  # Shape is now [B, T, C, H, W]

        x_hat = self(X)

        self.compute_loss(Y_s=Y_s, Y_c=Y_c, X_hat=x_hat, step="test", sync_dist=True)
        
        Y_c = Y_c.view(-1)
        x_hat = x_hat.view(-1, self.hparams.num_classes)
        
        self.probs.update(x_hat.softmax(dim=-1))
        self.targets.update(Y_c)
        self.conf.update(x_hat, Y_c)

    def on_test_epoch_end(self):
        probs = self.probs.compute()
        targets = self.targets.compute()
        confmat  = self.conf.compute()

        if self.trainer.is_global_zero:
            logger = self.trainer.logger
            run = logger.experiment  
            
            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["images/test_confusion_matrix"]
                
            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "test_confusion_matrix.png")
                    
            plot_confusion_matrix(confmat.cpu().numpy(), np.arange(self.hparams.num_classes), experiment=experiment, out_path=out_path)

            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["reports/test_classification_report"]
                
            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "test_classification_report.json")
                    
            compute_classification_report(targets.cpu().numpy(), probs.argmax(dim=-1).cpu().numpy(), experiment=experiment, out_path=out_path)

        self.probs.reset()
        self.targets.reset()
        self.conf.reset()
        

    def forward(self, x: torch.tensor):

        z = self.encoder(x)
        z = self.rope(z)
        z = self.norm(z)

        return self.proj(z)


class MHAEffnetV2s(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        encoder = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        encoder.classifier = nn.Linear(self.hparams.features, self.hparams.embed_dim)

        self.encoder = TimeDistributed(encoder)

        pe = self.positional_encoding(seq_len=self.hparams.max_seq_len, d_model=self.hparams.embed_dim, seq_n=0)
        self.register_buffer("pe", pe)
        self.mha = MHABlock(embed_dim=self.hparams.embed_dim, num_heads=self.hparams.num_heads, dropout=self.hparams.dropout)

        self.norm = nn.LayerNorm(self.hparams.embed_dim)        
        self.proj = nn.Linear(self.hparams.embed_dim, self.hparams.num_classes)

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
        

        self.accuracy = Accuracy(task='multiclass', num_classes=self.hparams.num_classes)
        self.conf = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=self.hparams.num_classes, normalize='true')

        self.probs = CatMetric()
        self.targets = CatMetric()

        # Recall for class 4 (multiclass)
        self.val_recall = torchmetrics.classification.MulticlassRecall(
            num_classes=self.hparams.num_classes,
            average=None,        # returns per-class recall
        )

        # measurable detection: target = 1 if true in {3,4}, pred = 1 if score_pred >= 0.75
        self.val_meas_stats = torchmetrics.classification.BinaryStatScores(threshold=0.5)  
        # (threshold irrelevant since we pass int preds, but torchmetrics requires one)

        # reject false positives at two thresholds (0.75 and 0.9)
        self.val_reject_fp75 = torchmetrics.classification.BinaryStatScores(threshold=0.75)
        self.val_reject_fp90 = torchmetrics.classification.BinaryStatScores(threshold=0.90)
        # returns (tp, fp, tn, fn) aggregated across GPUs

    def positional_encoding(self, seq_len: int, d_model: int, seq_n: int) -> torch.Tensor:
        """
        Sinusoidal positional encoding with tag-based offset.

        Args:
            seq_len (int): Sequence length.
            d_model (int): Embedding dimension.
            seq_n (int): Number of distinct sequences.
            device (str): Device to store the tensor.

        Returns:
            torch.Tensor: Positional encoding (seq_len, d_model).
        """
        pe = torch.zeros(seq_len, d_model)
        
        # Offset positions by a tag-dependent amount to make each sequence encoding unique
        position = torch.arange(seq_n * seq_len, (seq_n + 1) * seq_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("BCE with Regression Model")

        group.add_argument("--lr", type=float, default=1e-3)
        group.add_argument("--betas", type=float, nargs="+", default=(0.9, 0.999), help='Betas for Adam optimizer')
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=1e-2)

        group.add_argument("--sigma", type=float, nargs="+", default=(0.18, 0.12, 0.12, 0.10, 0.07), help='Sigma for Ordinal EMD Loss')
        group.add_argument("--bins", type=float, nargs="+", default=(0.0, 0.25, 0.5, 0.75, 1.0), help='Bins for Ordinal EMD Loss')
        group.add_argument("--class_weights", type=float, nargs="+", default=[0.03603907, 0.14391553, 0.85467111, 1.73506923, 2.23030506], help='Class weights for Ordinal EMD Loss')
        group.add_argument("--bin_weights", type=float, nargs="+", default=[0.23809524, 0.47619048, 0.71428571, 1.19047619, 2.38095238], help='Bin weights for Ordinal EMD Loss')
        group.add_argument("--num_classes", type=int, default=5, help='Output channels for projection head')

        group.add_argument("--top_aux_weight", type=float, default=0.0, help='Weight for auxiliary loss on top class')
        group.add_argument("--top_pos_weight", type=float, default=7.0, help='Positive weight for auxiliary loss on top class')
        group.add_argument("--top_aux_warmup_steps", type=int, default=2000, help='Number of warmup steps for auxiliary loss on top class')

        group.add_argument("--reject_tail_weight", type=float, default=0.00, help='Weight for reject tail penalty')
        group.add_argument("--reject_tau", type=float, default=0.85, help='Reject tail threshold')

        group.add_argument("--temporal_score_tv_power", type=int, default=1, help='Power for temporal total variation regularization on expected score')
        group.add_argument("--temporal_score_tv_weight", type=float, default=0.00, help='Weight for temporal total variation regularization on expected score')

        group.add_argument("--temporal_derivative_weight", type=float, default=0.00, help='Weight for temporal derivative matching loss on expected score')
        group.add_argument("--temporal_derivative_warmup_steps", nargs="+", type=int, default=[0, 200], help='Number of warmup steps for temporal derivative matching loss on expected score')
        group.add_argument("--temporal_derivative_power", type=int, default=1, help='Power for temporal derivative matching loss on expected score')
        
        group.add_argument("--meas_thresh", type=float, default=0.75)
        group.add_argument("--earlystop_fp_lambda", type=float, default=0.25)
        group.add_argument("--earlystop_fp_tail_lambda", type=float, default=1.0)
        
        # Image Encoder parameters 
        group.add_argument("--spatial_dims", type=int, default=2, help='Spatial dimensions for the encoder')
        group.add_argument("--in_channels", type=int, default=3, help='Input channels for encoder')
        group.add_argument("--features", type=int, default=1280, help='Number of output features for the encoder')
        group.add_argument("--embed_dim", type=int, default=128, help='Embedding dimension for the model')
        group.add_argument("--num_heads", type=int, default=8, help='Number of heads for MHA')
        group.add_argument("--max_seq_len", type=int, default=1000, help='Maximum sequence length for PE')
        group.add_argument("--dropout", type=float, default=0.2, help='Dropout rate')

        return parent_parser
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                betas=self.hparams.betas,
                                weight_decay=self.hparams.weight_decay)
        return optimizer
    
    def compute_loss(self, Y_s, X_hat, Y_c=None, step="train", sync_dist=False):

        # print("Y unique:", torch.unique(Y))
        # print("Y min/max:", Y.min().item(), Y.max().item())
        # print("Number of zeros in Y:", (Y == 0).sum().item())

        # print("X_hat min/max:", X_hat.min().item(), X_hat.max().item())
        # print("Number of zeros in X_hat:", (X_hat == 0).sum().item())
        
        loss = self.loss_fn(X_hat, y=Y_s.float(), y_class=Y_c)

        C = self.hparams.num_classes
        logits = X_hat
        logits_f = logits.reshape(-1, C)

        if(Y_c is not None):
            y_class = Y_c
            y_class_f = y_class.reshape(-1)

        if self.hparams.top_aux_weight > 0.0 and Y_c is not None and step == "train":

            is_top = (y_class_f == (C - 1)).float()

            pos_w = logits_f.new_tensor(float(self.hparams.top_pos_weight))

            logit_top = logits_f[:, -1]
            logit_rest = torch.logsumexp(logits_f[:, :-1], dim=1)
            margin = logit_top - logit_rest
            aux = F.binary_cross_entropy_with_logits(margin, is_top, reduction="none", pos_weight=pos_w)           

            if self.hparams.top_aux_warmup_steps > 0:
                warm = min(1.0, float(self.global_step) / float(self.hparams.top_aux_warmup_steps))
                aux_w = self.hparams.top_aux_weight * warm
            else:
                aux_w = self.hparams.top_aux_weight

            aux_mean = aux.mean() * aux_w

            loss = loss + aux_mean
            self.log(f"{step}_aux_loss", aux_mean, sync_dist=sync_dist)

            with torch.no_grad():
                self.log(f"{step}_top_rate", is_top.mean(), sync_dist=sync_dist)
                self.log(f"{step}_margin_mean", margin.mean(), sync_dist=sync_dist)
                if is_top.bool().any():
                    self.log(f"{step}_margin_pos_mean", margin[is_top.bool()].mean(), sync_dist=sync_dist)
                self.log(f"{step}_margin_neg_mean", margin[~is_top.bool()].mean(), sync_dist=sync_dist)

        if self.hparams.reject_tail_weight > 0.0 and Y_c is not None and step == "train":

            # Reject tail penalty
            # after logits_f and y_class_f
            bins = logits_f.new_tensor(self.hparams.bins)
            p = torch.softmax(logits_f, dim=1)
            score_pred_f = (p * bins).sum(dim=1)

            is_reject = (y_class_f == 0)

            reject_tail = logits_f.new_zeros(())
            if is_reject.any():
                reject_tail = torch.relu(score_pred_f[is_reject] - self.hparams.reject_tau).mean()
                loss = loss + self.hparams.reject_tail_weight * reject_tail

            self.log(f"{step}_reject_tail", reject_tail, sync_dist=sync_dist)

        if getattr(self.hparams, "temporal_score_tv_weight", 0.0) > 0 and step == "train":
            # s = expected_score_from_logits(logits, self.loss_fn.bins)  # (B,T)
            tv_s = temporal_tv(logits, power=getattr(self.hparams, "temporal_score_tv_power", 1))
            loss = loss + self.hparams.temporal_score_tv_weight * tv_s
            self.log(f"{step}_tv_score", tv_s, sync_dist=sync_dist)  

        if getattr(self.hparams, "temporal_derivative_weight", 0.0) > 0 and step == "train":
            s = expected_score_from_logits(logits, self.loss_fn.bins)  # (B,T)
            warm = 0.0
            if self.hparams.temporal_derivative_warmup_steps[0] <= self.global_step:
                warm = min(1.0, float(self.global_step - self.hparams.temporal_derivative_warmup_steps[0]) / float(self.hparams.temporal_derivative_warmup_steps[1] - self.hparams.temporal_derivative_warmup_steps[0]))
            dml = warm * self.hparams.temporal_derivative_weight * derivative_match_loss(s, Y_s.float(), power=getattr(self.hparams, "temporal_derivative_power", 1))            
            loss = loss + dml
            self.log(f"{step}_derivative_match_loss", dml, sync_dist=sync_dist)    

        self.log(f"{step}_loss", loss, sync_dist=sync_dist)

        return loss

    def training_step(self, train_batch, batch_idx):
        X = train_batch["img"]
        Y_s = train_batch["scalar"]
        Y_c = train_batch["class"]

        X = X.permute(0, 2, 1, 3, 4)  # Shape is now [B, T, C, H, W]

        x_hat = self(self.train_transform(X))

        return self.compute_loss(Y_s=Y_s, Y_c=Y_c, X_hat=x_hat, step="train")

    def validation_step(self, val_batch, batch_idx):
        X = val_batch["img"]       # (B,C,T,H,W) presumably
        Y_s = val_batch["scalar"]  # (B,T) or (B,N) float scores
        Y_c = val_batch["class"]   # (B,T) int 0..C-1

        X = X.permute(0, 2, 1, 3, 4)  # -> (B,T,C,H,W)
        logits = self(X)             # (B,T,C)

        bins = logits.new_tensor(self.hparams.bins)  # (C,)

        # expected score in [0,1]
        p = torch.softmax(logits, dim=-1)
        score_pred = (p * bins).sum(dim=-1)          # (B,T)

        # nearest-bin predicted class
        pred_cls = torch.argmin(torch.abs(score_pred[..., None] - bins), dim=-1)  # (B,T)

        # true class: use Y_c directly (safer than Y_s->bins mapping)
        true_cls = Y_c  # (B,T)

        # flatten for metrics
        pred_cls_f = pred_cls.reshape(-1)
        true_cls_f = true_cls.reshape(-1)
        score_pred_f = score_pred.reshape(-1)

        # -------------------------
        # (A) measurable recall
        # -------------------------
        is_meas = (true_cls_f >= 3).int()  # classes 3,4
        pred_meas = (score_pred_f >= self.hparams.meas_thresh).int()
        self.val_meas_stats.update(pred_meas, is_meas)

        # -------------------------
        # (B) reject FP at 0.75/0.9
        # target_bin: 0=reject, 1=non-reject
        # pred_pos: predicted non-reject by thresholding score_pred
        # FP/(FP+TN) computed at epoch end is P(pred_pos=1 | target_bin=0)
        # -------------------------
        target_nonreject = (true_cls_f != 0).int()
        self.val_reject_fp75.update((score_pred_f >= 0.75).int(), target_nonreject)
        self.val_reject_fp90.update((score_pred_f >= 0.90).int(), target_nonreject)

        

        # for cm
        logits_f = logits.reshape(-1, self.hparams.num_classes)
        y_f = Y_c.reshape(-1)

        self.probs.update(logits_f.softmax(dim=-1))
        self.targets.update(y_f)
        self.conf.update(logits_f, y_f)

    def _unpack_bss(self, bss):
        """
        Torchmetrics BinaryStatScores compute() can return:
        - tuple(tp, fp, tn, fn)  (older)
        - tensor([tp, fp, tn, fn, sup]) or shape (...,5) (newer)
        Returns: tp, fp, tn, fn as scalars/tensors
        """
        if isinstance(bss, (tuple, list)):
            tp, fp, tn, fn = bss[:4]
            return tp, fp, tn, fn

        # tensor output
        # last dim is [tp, fp, tn, fn, sup]
        tp = bss[..., 0]
        fp = bss[..., 1]
        tn = bss[..., 2]
        fn = bss[..., 3]
        return tp, fp, tn, fn
        
    def on_validation_epoch_end(self):
        # measurable recall = TP / (TP + FN)
        bss = self.val_meas_stats.compute()
        tp, fp, tn, fn = self._unpack_bss(bss)
        recall_meas = tp / (tp + fn + 1e-8)

        # reject FP@0.75 = FP / (FP + TN) in the nonreject-vs-reject binary framing
        bss75 = self.val_reject_fp75.compute()
        tp, fp, tn, fn = self._unpack_bss(bss75)
        fp_reject_075 = fp / (fp + tn + 1e-8)

        bss90 = self.val_reject_fp90.compute()
        tp, fp, tn, fn = self._unpack_bss(bss90)
        fp_reject_090 = fp / (fp + tn + 1e-8)

        lam = getattr(self.hparams, "earlystop_fp_lambda", 0.25)
        mu  = getattr(self.hparams, "earlystop_fp_tail_lambda", 1.0)

        val_select = recall_meas - lam * fp_reject_075 - mu * fp_reject_090

        self.log("val_recall_meas", recall_meas, prog_bar=True)
        self.log("val_fp_reject_075", fp_reject_075, prog_bar=True)
        self.log("val_fp_reject_090", fp_reject_090, prog_bar=True)
        self.log("val_select", val_select, prog_bar=True)

        self.val_meas_stats.reset()
        self.val_reject_fp75.reset()
        self.val_reject_fp90.reset()

        # confusion matrix/report 
        probs = self.probs.compute()
        targets = self.targets.compute()
        confmat = self.conf.compute()

        if self.trainer.is_global_zero:
            logger = self.trainer.logger
            run = logger.experiment

            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["images/val_confusion_matrix"]

            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "val_confusion_matrix.png")

            plot_confusion_matrix(
                confmat.cpu().numpy(),
                np.arange(self.hparams.num_classes),
                experiment=experiment,
                out_path=out_path
            )

            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["reports/val_classification_report"]

            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "val_classification_report.json")

            compute_classification_report(
                targets.cpu().numpy(),
                probs.argmax(dim=-1).cpu().numpy(),
                experiment=experiment,
                out_path=out_path
            )

        self.probs.reset()
        self.targets.reset()
        self.conf.reset()



    def test_step(self, test_batch, batch_idx):
        X = test_batch["img"]
        Y_s = test_batch["scalar"]
        Y_c = test_batch["class"]

        X = X.permute(0, 2, 1, 3, 4)  # Shape is now [B, T, C, H, W]

        x_hat = self(X)

        self.compute_loss(Y_s=Y_s, Y_c=Y_c, X_hat=x_hat, step="test", sync_dist=True)
        
        Y_c = Y_c.view(-1)
        x_hat = x_hat.view(-1, self.hparams.num_classes)
        
        self.probs.update(x_hat.softmax(dim=-1))
        self.targets.update(Y_c)
        self.conf.update(x_hat, Y_c)

    def on_test_epoch_end(self):
        probs = self.probs.compute()
        targets = self.targets.compute()
        confmat  = self.conf.compute()

        if self.trainer.is_global_zero:
            logger = self.trainer.logger
            run = logger.experiment  
            
            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["images/test_confusion_matrix"]
                
            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "test_confusion_matrix.png")
                    
            plot_confusion_matrix(confmat.cpu().numpy(), np.arange(self.hparams.num_classes), experiment=experiment, out_path=out_path)

            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["reports/test_classification_report"]
                
            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "test_classification_report.json")
                    
            compute_classification_report(targets.cpu().numpy(), probs.argmax(dim=-1).cpu().numpy(), experiment=experiment, out_path=out_path)

        self.probs.reset()
        self.targets.reset()
        self.conf.reset()
        

    def forward(self, x: torch.tensor):
        # X is (B,C,T,H,W)
        z = self.encoder(x) # (B,T,embed_dim)
        z = z + self.mha(z + self.pe[None, :z.size(1), :])
        z = self.norm(z)
        return self.proj(z)


class EffnetV2sTDOEMD(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        encoder = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        encoder.classifier = nn.Linear(self.hparams.features, self.hparams.embed_dim)
        self.proj = nn.Linear(self.hparams.embed_dim, self.hparams.num_classes)
        
        self.dropout = nn.Dropout(p=self.hparams.dropout)
        self.encoder = TimeDistributed(encoder)

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

        self.accuracy = Accuracy(task='multiclass', num_classes=self.hparams.num_classes)
        self.conf = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=self.hparams.num_classes, normalize='true')

        self.probs = CatMetric()
        self.targets = CatMetric()

        # Recall for class 4 (multiclass)
        self.val_recall = torchmetrics.classification.MulticlassRecall(
            num_classes=self.hparams.num_classes,
            average=None,        # returns per-class recall
        )

        # measurable detection: target = 1 if true in {3,4}, pred = 1 if score_pred >= 0.75
        self.val_meas_stats = torchmetrics.classification.BinaryStatScores(threshold=0.5)  
        # (threshold irrelevant since we pass int preds, but torchmetrics requires one)

        # reject false positives at two thresholds (0.75 and 0.9)
        self.val_reject_fp75 = torchmetrics.classification.BinaryStatScores(threshold=0.75)
        self.val_reject_fp90 = torchmetrics.classification.BinaryStatScores(threshold=0.90)
        # returns (tp, fp, tn, fn) aggregated across GPUs


    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("V2STDOEMD Model")

        group.add_argument("--embed_dim", type=int, default=128, help='Embedding dimension for the model')
        group.add_argument("--num_classes", type=int, default=5, help='Output channels for projection head')

        group.add_argument("--lr", type=float, default=1e-3)
        group.add_argument("--betas", type=float, nargs="+", default=(0.9, 0.999), help='Betas for Adam optimizer')
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=1e-2)        

        group.add_argument("--sigma", type=float, nargs="+", default=(0.18, 0.12, 0.12, 0.10, 0.07), help='Sigma for Ordinal EMD Loss')
        group.add_argument("--bins", type=float, nargs="+", default=(0.0, 0.25, 0.5, 0.75, 1.0), help='Bins for Ordinal EMD Loss')
        group.add_argument("--class_weights", type=float, nargs="+", default=[0.03603907, 0.14391553, 0.85467111, 1.73506923, 2.23030506], help='Class weights for Ordinal EMD Loss')
        group.add_argument("--bin_weights", type=float, nargs="+", default=[0.23809524, 0.47619048, 0.71428571, 1.19047619, 2.38095238], help='Bin weights for Ordinal EMD Loss')        

        group.add_argument("--top_aux_weight", type=float, default=0.1, help='Weight for auxiliary loss on top class')
        group.add_argument("--top_pos_weight", type=float, default=7.0, help='Positive weight for auxiliary loss on top class')
        group.add_argument("--top_aux_warmup_steps", type=int, default=2000, help='Number of warmup steps for auxiliary loss on top class')

        group.add_argument("--meas_thresh", type=float, default=0.75)
        group.add_argument("--earlystop_fp_lambda", type=float, default=0.25)
        group.add_argument("--earlystop_fp_tail_lambda", type=float, default=1.0)
        
        # Image Encoder parameters 
        group.add_argument("--spatial_dims", type=int, default=2, help='Spatial dimensions for the encoder')
        group.add_argument("--in_channels", type=int, default=3, help='Input channels for encoder')
        group.add_argument("--features", type=int, default=1280, help='Number of output features for the encoder')        
        group.add_argument("--dropout", type=float, default=0.2, help='Dropout rate for RoPE transformer')

        return parent_parser
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                betas=self.hparams.betas,
                                weight_decay=self.hparams.weight_decay)
        return optimizer
    
    def compute_loss(self, Y_s, X_hat, Y_c=None, step="train", sync_dist=False):

        # print("Y unique:", torch.unique(Y))
        # print("Y min/max:", Y.min().item(), Y.max().item())
        # print("Number of zeros in Y:", (Y == 0).sum().item())

        # print("X_hat min/max:", X_hat.min().item(), X_hat.max().item())
        # print("Number of zeros in X_hat:", (X_hat == 0).sum().item())
        
        loss = self.loss_fn(X_hat, y=Y_s.float(), y_class=Y_c)


         # aux top-bin loss (apply SAME mask)
        if self.hparams.top_aux_weight > 0.0 and Y_c is not None and step == "train":
            C = self.hparams.num_classes

            logits = X_hat
            y_class = Y_c

            # flatten
            logits_f = logits.reshape(-1, C)
            y_class_f = y_class.reshape(-1)

            is_top = (y_class_f == (C - 1)).float()

            pos_w = torch.tensor(self.hparams.top_pos_weight, device=logits_f.device, dtype=logits_f.dtype)            

            logit_top = logits_f[:, -1]
            logit_rest = torch.logsumexp(logits_f[:, :-1], dim=1)
            margin = logit_top - logit_rest
            aux = F.binary_cross_entropy_with_logits(margin, is_top, reduction="none", pos_weight=pos_w)

            aux_mean = aux.mean()

            if self.hparams.top_aux_warmup_steps > 0:
                warm = min(1.0, float(self.global_step) / float(self.hparams.top_aux_warmup_steps))
                aux_w = self.hparams.top_aux_weight * warm
            else:
                aux_w = self.hparams.top_aux_weight

            loss = loss + aux_w * aux_mean
            self.log(f"{step}_aux_loss", aux_mean, sync_dist=sync_dist)

            with torch.no_grad():
                self.log(f"{step}_top_rate", is_top.mean(), sync_dist=sync_dist)
                self.log(f"{step}_margin_mean", margin.mean(), sync_dist=sync_dist)
                if is_top.any():
                    self.log(f"{step}_margin_pos_mean", margin[is_top.bool()].mean(), sync_dist=sync_dist)
                self.log(f"{step}_margin_neg_mean", margin[~is_top.bool()].mean(), sync_dist=sync_dist)

        self.log(f"{step}_loss", loss, sync_dist=sync_dist)

        return loss

    def training_step(self, train_batch, batch_idx):
        X = train_batch["img"]
        Y_s = train_batch["scalar"]
        Y_c = train_batch["class"]

        X = X.permute(0, 2, 1, 3, 4)  # Shape is now [B, T, C, H, W]

        x_hat = self(self.train_transform(X))

        return self.compute_loss(Y_s=Y_s, Y_c=Y_c, X_hat=x_hat, step="train")

    def validation_step(self, val_batch, batch_idx):
        X = val_batch["img"]       # (B,C,T,H,W) presumably
        Y_s = val_batch["scalar"]  # (B,T) or (B,N) float scores
        Y_c = val_batch["class"]   # (B,T) int 0..C-1

        X = X.permute(0, 2, 1, 3, 4)  # -> (B,T,C,H,W)
        logits = self(X)             # (B,T,C)

        bins = logits.new_tensor(self.hparams.bins)  # (C,)

        # expected score in [0,1]
        p = torch.softmax(logits, dim=-1)
        score_pred = (p * bins).sum(dim=-1)          # (B,T)

        # nearest-bin predicted class
        pred_cls = torch.argmin(torch.abs(score_pred[..., None] - bins), dim=-1)  # (B,T)

        # true class: use Y_c directly (safer than Y_s->bins mapping)
        true_cls = Y_c  # (B,T)

        # flatten for metrics
        pred_cls_f = pred_cls.reshape(-1)
        true_cls_f = true_cls.reshape(-1)
        score_pred_f = score_pred.reshape(-1)

        # -------------------------
        # (A) measurable recall
        # -------------------------
        is_meas = (true_cls_f >= 3).int()  # classes 3,4
        pred_meas = (score_pred_f >= self.hparams.meas_thresh).int()
        self.val_meas_stats.update(pred_meas, is_meas)

        # -------------------------
        # (B) reject FP at 0.75/0.9
        # target_bin: 0=reject, 1=non-reject
        # pred_pos: predicted non-reject by thresholding score_pred
        # FP/(FP+TN) computed at epoch end is P(pred_pos=1 | target_bin=0)
        # -------------------------
        target_nonreject = (true_cls_f != 0).int()
        self.val_reject_fp75.update((score_pred_f >= 0.75).int(), target_nonreject)
        self.val_reject_fp90.update((score_pred_f >= 0.90).int(), target_nonreject)

        

        # for cm
        logits_f = logits.reshape(-1, self.hparams.num_classes)
        y_f = Y_c.reshape(-1)

        self.probs.update(logits_f.softmax(dim=-1))
        self.targets.update(y_f)
        self.conf.update(logits_f, y_f)

    def _unpack_bss(self, bss):
        """
        Torchmetrics BinaryStatScores compute() can return:
        - tuple(tp, fp, tn, fn)  (older)
        - tensor([tp, fp, tn, fn, sup]) or shape (...,5) (newer)
        Returns: tp, fp, tn, fn as scalars/tensors
        """
        if isinstance(bss, (tuple, list)):
            tp, fp, tn, fn = bss[:4]
            return tp, fp, tn, fn

        # tensor output
        # last dim is [tp, fp, tn, fn, sup]
        tp = bss[..., 0]
        fp = bss[..., 1]
        tn = bss[..., 2]
        fn = bss[..., 3]
        return tp, fp, tn, fn
        
    def on_validation_epoch_end(self):
        # measurable recall = TP / (TP + FN)
        bss = self.val_meas_stats.compute()
        tp, fp, tn, fn = self._unpack_bss(bss)
        recall_meas = tp / (tp + fn + 1e-8)

        # reject FP@0.75 = FP / (FP + TN) in the nonreject-vs-reject binary framing
        bss75 = self.val_reject_fp75.compute()
        tp, fp, tn, fn = self._unpack_bss(bss75)
        fp_reject_075 = fp / (fp + tn + 1e-8)

        bss90 = self.val_reject_fp90.compute()
        tp, fp, tn, fn = self._unpack_bss(bss90)
        fp_reject_090 = fp / (fp + tn + 1e-8)

        lam = getattr(self.hparams, "earlystop_fp_lambda", 0.25)
        mu  = getattr(self.hparams, "earlystop_fp_tail_lambda", 1.0)

        val_select = recall_meas - lam * fp_reject_075 - mu * fp_reject_090

        self.log("val_recall_meas", recall_meas, prog_bar=True)
        self.log("val_fp_reject_075", fp_reject_075, prog_bar=True)
        self.log("val_fp_reject_090", fp_reject_090, prog_bar=True)
        self.log("val_select", val_select, prog_bar=True)

        self.val_meas_stats.reset()
        self.val_reject_fp75.reset()
        self.val_reject_fp90.reset()

        # confusion matrix/report 
        probs = self.probs.compute()
        targets = self.targets.compute()
        confmat = self.conf.compute()

        if self.trainer.is_global_zero:
            logger = self.trainer.logger
            run = logger.experiment

            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["images/val_confusion_matrix"]

            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "val_confusion_matrix.png")

            plot_confusion_matrix(
                confmat.cpu().numpy(),
                np.arange(self.hparams.num_classes),
                experiment=experiment,
                out_path=out_path
            )

            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["reports/val_classification_report"]

            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "val_classification_report.json")

            compute_classification_report(
                targets.cpu().numpy(),
                probs.argmax(dim=-1).cpu().numpy(),
                experiment=experiment,
                out_path=out_path
            )

        self.probs.reset()
        self.targets.reset()
        self.conf.reset()



    def test_step(self, test_batch, batch_idx):
        X = test_batch["img"]
        Y_s = test_batch["scalar"]
        Y_c = test_batch["class"]

        X = X.permute(0, 2, 1, 3, 4)  # Shape is now [B, T, C, H, W]

        x_hat = self(X)

        self.compute_loss(Y_s=Y_s, Y_c=Y_c, X_hat=x_hat, step="test", sync_dist=True)
        
        Y_c = Y_c.view(-1)
        x_hat = x_hat.view(-1, self.hparams.num_classes)
        
        self.probs.update(x_hat.softmax(dim=-1))
        self.targets.update(Y_c)
        self.conf.update(x_hat, Y_c)

    def on_test_epoch_end(self):
        probs = self.probs.compute()
        targets = self.targets.compute()
        confmat  = self.conf.compute()

        if self.trainer.is_global_zero:
            logger = self.trainer.logger
            run = logger.experiment  
            
            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["images/test_confusion_matrix"]
                
            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "test_confusion_matrix.png")
                    
            plot_confusion_matrix(confmat.cpu().numpy(), np.arange(self.hparams.num_classes), experiment=experiment, out_path=out_path)

            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["reports/test_classification_report"]
                
            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "test_classification_report.json")
                    
            compute_classification_report(targets.cpu().numpy(), probs.argmax(dim=-1).cpu().numpy(), experiment=experiment, out_path=out_path)

        self.probs.reset()
        self.targets.reset()
        self.conf.reset()
        

    def forward(self, x: torch.tensor):

        z = self.encoder(x)
        z = self.dropout(z)

        return self.proj(z)
    


class TemporalRefinerEffnetV2s(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()       
        

        if self.hparams.model_freeze:
            effnet = EffnetV2sTDOEMD.load_from_checkpoint(self.hparams.model_freeze, map_location='cpu')
            effnet.freeze()

            self.encoder = TimeDistributed(effnet.encoder.module)
            self.head = effnet.proj
            
        else:
            encoder = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')
            encoder.classifier = nn.Linear(self.hparams.features, self.hparams.embed_dim)
            self.encoder = TimeDistributed(encoder)
            self.head = nn.Linear(self.hparams.embed_dim, self.hparams.num_classes)

        self.rope = RoPETransformerBlock(dim=self.hparams.embed_dim, num_heads=self.hparams.num_heads, mlp_ratio=self.hparams.mlp_ratio, dropout=self.hparams.dropout)
        self.norm = nn.LayerNorm(self.hparams.embed_dim)

        self.film_gamma = nn.Linear(self.hparams.num_classes, self.hparams.embed_dim)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)

        self.film_beta = nn.Linear(self.hparams.num_classes, self.hparams.embed_dim)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        self.rope_head = nn.Linear(self.hparams.embed_dim, self.hparams.num_classes)
        nn.init.zeros_(self.rope_head.weight)
        nn.init.zeros_(self.rope_head.bias)
        

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
        

        self.accuracy = Accuracy(task='multiclass', num_classes=self.hparams.num_classes)
        self.conf = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=self.hparams.num_classes, normalize='true')

        self.probs = CatMetric()
        self.targets = CatMetric()

        # Recall for class 4 (multiclass)
        self.val_recall = torchmetrics.classification.MulticlassRecall(
            num_classes=self.hparams.num_classes,
            average=None,        # returns per-class recall
        )

        # measurable detection: target = 1 if true in {3,4}, pred = 1 if score_pred >= 0.75
        self.val_meas_stats = torchmetrics.classification.BinaryStatScores(threshold=0.5)  
        # (threshold irrelevant since we pass int preds, but torchmetrics requires one)

        # reject false positives at two thresholds (0.75 and 0.9)
        self.val_reject_fp75 = torchmetrics.classification.BinaryStatScores(threshold=0.75)
        self.val_reject_fp90 = torchmetrics.classification.BinaryStatScores(threshold=0.90)
        # returns (tp, fp, tn, fn) aggregated across GPUs

    def positional_encoding(self, seq_len: int, d_model: int, seq_n: int) -> torch.Tensor:
        """
        Sinusoidal positional encoding with tag-based offset.

        Args:
            seq_len (int): Sequence length.
            d_model (int): Embedding dimension.
            seq_n (int): Number of distinct sequences.
            device (str): Device to store the tensor.

        Returns:
            torch.Tensor: Positional encoding (seq_len, d_model).
        """
        pe = torch.zeros(seq_len, d_model)
        
        # Offset positions by a tag-dependent amount to make each sequence encoding unique
        position = torch.arange(seq_n * seq_len, (seq_n + 1) * seq_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("BCE with Regression Model")

        group.add_argument("--lr", type=float, default=1e-3)
        group.add_argument("--betas", type=float, nargs="+", default=(0.9, 0.999), help='Betas for Adam optimizer')
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=1e-2)

        # group.add_argument("--model_freeze", type=str, default='/mnt/raid/C1_ML_Analysis/train_output/classification/EffnetV2sTDOEMD/v0.2/epoch=15-val_select=0.235.ckpt', help='Path to pretrained EffnetV2sTDOEMD checkpoint to freeze encoder and head')
        group.add_argument("--model_freeze", type=str, default=None, help='Path to pretrained EffnetV2sTDOEMD checkpoint to freeze encoder and head')

        group.add_argument("--sigma", type=float, nargs="+", default=(0.16, 0.12, 0.12, 0.1, 0.06), help='Sigma for Ordinal EMD Loss')
        group.add_argument("--bins", type=float, nargs="+", default=(0.0, 0.25, 0.5, 0.75, 1.0), help='Bins for Ordinal EMD Loss')
        group.add_argument("--class_weights", type=float, nargs="+", default=[0.03603907, 0.14391553, 0.85467111, 1.73506923, 2.23030506], help='Class weights for Ordinal EMD Loss')
        group.add_argument("--bin_weights", type=float, nargs="+", default=[0.23809524, 0.47619048, 0.71428571, 1.19047619, 2.38095238], help='Bin weights for Ordinal EMD Loss')
        group.add_argument("--num_classes", type=int, default=5, help='Output channels for projection head')

        group.add_argument("--top_aux_weight", type=float, default=0.0, help='Weight for auxiliary loss on top class')
        group.add_argument("--top_pos_weight", type=float, default=7.0, help='Positive weight for auxiliary loss on top class')
        group.add_argument("--top_aux_warmup_steps", type=int, default=2000, help='Number of warmup steps for auxiliary loss on top class')

        group.add_argument("--reject_tail_weight", type=float, default=0.00, help='Weight for reject tail penalty')
        group.add_argument("--reject_tau", type=float, default=0.85, help='Reject tail threshold')

        group.add_argument("--temporal_score_tv_power", type=int, default=1, help='Power for temporal total variation regularization on expected score')
        group.add_argument("--temporal_score_tv_weight", type=float, default=0.00, help='Weight for temporal total variation regularization on expected score')

        group.add_argument("--temporal_derivative_weight", type=float, default=0.00, help='Weight for temporal derivative matching loss on expected score')
        group.add_argument("--temporal_derivative_warmup_steps", nargs="+", type=int, default=[0, 200], help='Number of warmup steps for temporal derivative matching loss on expected score')
        group.add_argument("--temporal_derivative_power", type=int, default=1, help='Power for temporal derivative matching loss on expected score')
        
        group.add_argument("--meas_thresh", type=float, default=0.75)
        group.add_argument("--earlystop_fp_lambda", type=float, default=0.25)
        group.add_argument("--earlystop_fp_tail_lambda", type=float, default=1.0)
        
        # Image Encoder parameters 
        group.add_argument("--spatial_dims", type=int, default=2, help='Spatial dimensions for the encoder')
        group.add_argument("--in_channels", type=int, default=3, help='Input channels for encoder')
        group.add_argument("--features", type=int, default=1280, help='Number of output features for the encoder')
        group.add_argument("--embed_dim", type=int, default=128, help='Embedding dimension for the model')
        group.add_argument("--num_heads", type=int, default=8, help='Number of heads for MHA')
        group.add_argument("--mlp_ratio", type=float, default=4.0, help='MLP ratio for RoPE transformer')
        group.add_argument("--max_seq_len", type=int, default=1000, help='Maximum sequence length for PE')
        group.add_argument("--dropout", type=float, default=0.2, help='Dropout rate')

        return parent_parser
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                betas=self.hparams.betas,
                                weight_decay=self.hparams.weight_decay)
        return optimizer
    
    def compute_loss(self, Y_s, logits, logits_res, Y_c=None, gate=None, step="train", sync_dist=False):

        # print("Y unique:", torch.unique(Y))
        # print("Y min/max:", Y.min().item(), Y.max().item())
        # print("Number of zeros in Y:", (Y == 0).sum().item())

        # print("X_hat min/max:", X_hat.min().item(), X_hat.max().item())
        # print("Number of zeros in X_hat:", (X_hat == 0).sum().item())

        C = self.hparams.num_classes        

        logits_ctx = logits + logits_res

        loss_ctx = self.loss_fn(logits_ctx, y=Y_s.float(), y_class=Y_c)
        loss_base = self.loss_fn(logits, y=Y_s.float(), y_class=Y_c)

        p_base = torch.softmax(logits.detach(), dim=-1)
        p_ctx  = torch.softmax(logits_ctx, dim=-1)
        loss_kl = F.kl_div(torch.log(p_ctx.clamp_min(1e-8)), p_base, reduction="batchmean")
        loss_l2 = (logits_res ** 2).mean()

        loss = 1.0 * loss_ctx + 0.1 * loss_base + 0.00 * loss_kl + 3e-4 * loss_l2

        if(gate is not None):
            loss_gate = torch.relu(gate - 0.02).mean()
            self.log(f"{step}_gate_mean", loss_gate, prog_bar=True, sync_dist=True)
            self.log(f"{step}_gate_p95",  torch.quantile(gate.detach().flatten(), 0.95), sync_dist=True)

            loss = loss + 1e-4 * loss_gate


        logits_f = logits_ctx.reshape(-1, C)

        if(Y_c is not None):
            y_class = Y_c
            y_class_f = y_class.reshape(-1)

        if self.hparams.top_aux_weight > 0.0 and Y_c is not None and step == "train":

            is_top = (y_class_f == (C - 1)).float()

            pos_w = logits_f.new_tensor(float(self.hparams.top_pos_weight))

            logit_top = logits_f[:, -1]
            logit_rest = torch.logsumexp(logits_f[:, :-1], dim=1)
            margin = logit_top - logit_rest
            aux = F.binary_cross_entropy_with_logits(margin, is_top, reduction="none", pos_weight=pos_w)           

            if self.hparams.top_aux_warmup_steps > 0:
                warm = min(1.0, float(self.global_step) / float(self.hparams.top_aux_warmup_steps))
                aux_w = self.hparams.top_aux_weight * warm
            else:
                aux_w = self.hparams.top_aux_weight

            aux_mean = aux.mean() * aux_w

            loss = loss + aux_mean
            self.log(f"{step}_aux_loss", aux_mean, sync_dist=sync_dist)

            with torch.no_grad():
                self.log(f"{step}_top_rate", is_top.mean(), sync_dist=sync_dist)
                self.log(f"{step}_margin_mean", margin.mean(), sync_dist=sync_dist)
                if is_top.bool().any():
                    self.log(f"{step}_margin_pos_mean", margin[is_top.bool()].mean(), sync_dist=sync_dist) # this must increase during training
                self.log(f"{step}_margin_neg_mean", margin[~is_top.bool()].mean(), sync_dist=sync_dist)

        if self.hparams.reject_tail_weight > 0.0 and Y_c is not None and step == "train":

            # Reject tail penalty
            # after logits_f and y_class_f
            bins = logits_f.new_tensor(self.hparams.bins)
            p = torch.softmax(logits_f, dim=1)
            score_pred_f = (p * bins).sum(dim=1)

            is_reject = (y_class_f == 0)

            reject_tail = logits_f.new_zeros(())
            if is_reject.any():
                reject_tail = torch.relu(score_pred_f[is_reject] - self.hparams.reject_tau).mean()
                loss = loss + self.hparams.reject_tail_weight * reject_tail

            self.log(f"{step}_reject_tail", reject_tail, sync_dist=sync_dist)

        if getattr(self.hparams, "temporal_score_tv_weight", 0.0) > 0 and step == "train":
            # s = expected_score_from_logits(logits, self.loss_fn.bins)  # (B,T)
            tv_s = temporal_tv(logits, power=getattr(self.hparams, "temporal_score_tv_power", 1))
            loss = loss + self.hparams.temporal_score_tv_weight * tv_s
            self.log(f"{step}_tv_score", tv_s, sync_dist=sync_dist)  

        if getattr(self.hparams, "temporal_derivative_weight", 0.0) > 0 and step == "train":
            s = expected_score_from_logits(logits, self.loss_fn.bins)  # (B,T)
            warm = 0.0
            if self.hparams.temporal_derivative_warmup_steps[0] <= self.global_step:
                warm = min(1.0, float(self.global_step - self.hparams.temporal_derivative_warmup_steps[0]) / float(self.hparams.temporal_derivative_warmup_steps[1] - self.hparams.temporal_derivative_warmup_steps[0]))
            dml = warm * self.hparams.temporal_derivative_weight * derivative_match_loss(s, Y_s.float(), power=getattr(self.hparams, "temporal_derivative_power", 1))            
            loss = loss + dml
            self.log(f"{step}_derivative_match_loss", dml, sync_dist=sync_dist)                

        self.log(f"{step}_loss", loss, sync_dist=sync_dist)
        self.log(f"{step}_loss_ctx", loss_ctx, sync_dist=sync_dist)
        self.log(f"{step}_loss_base", loss_base, sync_dist=sync_dist)
        self.log(f"{step}_loss_kl", loss_kl, sync_dist=sync_dist)
        self.log(f"{step}_loss_l2", loss_l2, sync_dist=sync_dist)


        return loss

    def training_step(self, train_batch, batch_idx):
        X = train_batch["img"]
        Y_s = train_batch["scalar"]
        Y_c = train_batch["class"]

        X = X.permute(0, 2, 1, 3, 4)  # Shape is now [B, T, C, H, W]
        
        z = self.encoder(self.train_transform(X)) # (B,T,embed_dim)
        
        logits = self.head(z)

        z_r = self.rope(z)
        z_r = self.norm(z_r)
        g = torch.sigmoid(self.gate(z_r))
        logits_res = g * self.rope_head(z_r)        

        return self.compute_loss(Y_s=Y_s, Y_c=Y_c, gate=g, logits=logits, logits_res=logits_res, step="train")

    def validation_step(self, val_batch, batch_idx):
        X = val_batch["img"]       # (B,C,T,H,W) presumably
        Y_s = val_batch["scalar"]  # (B,T) or (B,N) float scores
        Y_c = val_batch["class"]   # (B,T) int 0..C-1

        X = X.permute(0, 2, 1, 3, 4)  # -> (B,T,C,H,W)
        logits = self(X)             # (B,T,C)

        bins = logits.new_tensor(self.hparams.bins)  # (C,)

        # expected score in [0,1]
        p = torch.softmax(logits, dim=-1)
        score_pred = (p * bins).sum(dim=-1)          # (B,T)

        # nearest-bin predicted class
        pred_cls = torch.argmin(torch.abs(score_pred[..., None] - bins), dim=-1)  # (B,T)

        # true class: use Y_c directly (safer than Y_s->bins mapping)
        true_cls = Y_c  # (B,T)

        # flatten for metrics
        pred_cls_f = pred_cls.reshape(-1)
        true_cls_f = true_cls.reshape(-1)
        score_pred_f = score_pred.reshape(-1)

        # -------------------------
        # (A) measurable recall
        # -------------------------
        is_meas = (true_cls_f >= 3).int()  # classes 3,4
        pred_meas = (score_pred_f >= self.hparams.meas_thresh).int()
        self.val_meas_stats.update(pred_meas, is_meas)

        # -------------------------
        # (B) reject FP at 0.75/0.9
        # target_bin: 0=reject, 1=non-reject
        # pred_pos: predicted non-reject by thresholding score_pred
        # FP/(FP+TN) computed at epoch end is P(pred_pos=1 | target_bin=0)
        # -------------------------
        target_nonreject = (true_cls_f != 0).int()
        self.val_reject_fp75.update((score_pred_f >= 0.75).int(), target_nonreject)
        self.val_reject_fp90.update((score_pred_f >= 0.90).int(), target_nonreject)

        # for cm
        logits_f = logits.reshape(-1, self.hparams.num_classes)
        y_f = Y_c.reshape(-1)

        self.probs.update(logits_f.softmax(dim=-1))
        self.targets.update(y_f)
        self.conf.update(logits_f, y_f)

    def _unpack_bss(self, bss):
        """
        Torchmetrics BinaryStatScores compute() can return:
        - tuple(tp, fp, tn, fn)  (older)
        - tensor([tp, fp, tn, fn, sup]) or shape (...,5) (newer)
        Returns: tp, fp, tn, fn as scalars/tensors
        """
        if isinstance(bss, (tuple, list)):
            tp, fp, tn, fn = bss[:4]
            return tp, fp, tn, fn

        # tensor output
        # last dim is [tp, fp, tn, fn, sup]
        tp = bss[..., 0]
        fp = bss[..., 1]
        tn = bss[..., 2]
        fn = bss[..., 3]
        return tp, fp, tn, fn
        
    def on_validation_epoch_end(self):
        # measurable recall = TP / (TP + FN)
        bss = self.val_meas_stats.compute()
        tp, fp, tn, fn = self._unpack_bss(bss)
        recall_meas = tp / (tp + fn + 1e-8)

        # reject FP@0.75 = FP / (FP + TN) in the nonreject-vs-reject binary framing
        bss75 = self.val_reject_fp75.compute()
        tp, fp, tn, fn = self._unpack_bss(bss75)
        fp_reject_075 = fp / (fp + tn + 1e-8)

        bss90 = self.val_reject_fp90.compute()
        tp, fp, tn, fn = self._unpack_bss(bss90)
        fp_reject_090 = fp / (fp + tn + 1e-8)

        lam = getattr(self.hparams, "earlystop_fp_lambda", 0.25)
        mu  = getattr(self.hparams, "earlystop_fp_tail_lambda", 1.0)

        val_select = recall_meas - lam * fp_reject_075 - mu * fp_reject_090

        self.log("val_recall_meas", recall_meas, prog_bar=True)
        self.log("val_fp_reject_075", fp_reject_075, prog_bar=True)
        self.log("val_fp_reject_090", fp_reject_090, prog_bar=True)
        self.log("val_select", val_select, prog_bar=True)

        self.val_meas_stats.reset()
        self.val_reject_fp75.reset()
        self.val_reject_fp90.reset()

        # confusion matrix/report 
        probs = self.probs.compute()
        targets = self.targets.compute()
        confmat = self.conf.compute()

        if self.trainer.is_global_zero:
            logger = self.trainer.logger
            run = logger.experiment

            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["images/val_confusion_matrix"]

            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "val_confusion_matrix.png")

            plot_confusion_matrix(
                confmat.cpu().numpy(),
                np.arange(self.hparams.num_classes),
                experiment=experiment,
                out_path=out_path
            )

            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["reports/val_classification_report"]

            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "val_classification_report.json")

            compute_classification_report(
                targets.cpu().numpy(),
                probs.argmax(dim=-1).cpu().numpy(),
                experiment=experiment,
                out_path=out_path
            )

        self.probs.reset()
        self.targets.reset()
        self.conf.reset()



    def test_step(self, test_batch, batch_idx):
        X = test_batch["img"]
        Y_s = test_batch["scalar"]
        Y_c = test_batch["class"]

        X = X.permute(0, 2, 1, 3, 4)  # Shape is now [B, T, C, H, W]

        x_hat = self(X)

        self.compute_loss(Y_s=Y_s, Y_c=Y_c, X_hat=x_hat, step="test", sync_dist=True)
        
        Y_c = Y_c.view(-1)
        x_hat = x_hat.view(-1, self.hparams.num_classes)
        
        self.probs.update(x_hat.softmax(dim=-1))
        self.targets.update(Y_c)
        self.conf.update(x_hat, Y_c)

    def on_test_epoch_end(self):
        probs = self.probs.compute()
        targets = self.targets.compute()
        confmat  = self.conf.compute()

        if self.trainer.is_global_zero:
            logger = self.trainer.logger
            run = logger.experiment  
            
            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["images/test_confusion_matrix"]
                
            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "test_confusion_matrix.png")
                    
            plot_confusion_matrix(confmat.cpu().numpy(), np.arange(self.hparams.num_classes), experiment=experiment, out_path=out_path)

            experiment = None
            if isinstance(logger, NeptuneLogger):
                experiment = run["reports/test_classification_report"]
                
            out_path = None
            if os.path.exists(self.hparams.out):
                out_path = os.path.join(self.hparams.out, "test_classification_report.json")
                    
            compute_classification_report(targets.cpu().numpy(), probs.argmax(dim=-1).cpu().numpy(), experiment=experiment, out_path=out_path)

        self.probs.reset()
        self.targets.reset()
        self.conf.reset()
        

    def forward(self, x: torch.tensor):
        # X is (B,C,T,H,W)
        z = self.encoder(x) # (B,T,embed_dim)        
        logits = self.head(z)

        p = torch.softmax(logits.detach(), dim=-1)          # (B,T,C)
        gamma = self.film_gamma(p)                          # (B,T,D)
        beta  = self.film_beta(p)                           # (B,T,D)
        z = z * (1 + gamma) + beta

        z = self.rope(z)
        z = self.norm(z)
        z = self.rope_head(z)
        
        logits_ctx = logits + self.rope_head(z)

        return logits_ctx