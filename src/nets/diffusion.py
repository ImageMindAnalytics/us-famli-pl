import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision 

import pytorch_lightning as pl

import torchmetrics

from generative.inferers import LatentDiffusionInferer
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
# from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks import nets
from generative.networks.schedulers import DDPMScheduler
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from monai import transforms

class GaussianNoise(nn.Module):    
    def __init__(self, mean=0.0, std=0.05):
        super(GaussianNoise, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    def forward(self, x):
        if self.training:
            return x + torch.normal(mean=self.mean, std=self.std, size=x.size(), device=x.device)
        return x
    
class RandCoarseShuffle(nn.Module):    
    def __init__(self, prob=0.75, holes=16, spatial_size=32):
        super(RandCoarseShuffle, self).__init__()
        self.t = transforms.RandCoarseShuffle(prob=prob, holes=holes, spatial_size=spatial_size)
    def forward(self, x):
        if self.training:
            return self.t(x)
        return x

class SaltAndPepper(nn.Module):    
    def __init__(self, prob=0.05):
        super(SaltAndPepper, self).__init__()
        self.prob = prob
    def __call__(self, x):
        noise_tensor = torch.rand(x.shape)
        salt = torch.max(x)
        pepper = torch.min(x)
        x[noise_tensor < self.prob/2] = salt
        x[noise_tensor > 1-self.prob/2] = pepper
        return x

class AutoEncoderKL(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.autoencoderkl = nets.AutoencoderKL(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(128, 128, 256),
            latent_channels=3,
            num_res_blocks=2,
            attention_levels=(False, False, False),
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
        )

        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")

        self.automatic_optimization = False

        self.discriminator = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)        

        self.adversarial_loss = PatchAdversarialLoss(criterion="least_squares")        

        # For mixed precision training
        # self.scaler_g = GradScaler()
        # self.scaler_d = GradScaler()

        self.noise_transform = torch.nn.Sequential(
            GaussianNoise(0.0, 0.05),
            RandCoarseShuffle(),
            SaltAndPepper()
            
        )
        

    def configure_optimizers(self):
        optimizer_g = optim.AdamW(self.autoencoderkl.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        optimizer_d = optim.AdamW(self.discriminator.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        return [optimizer_g, optimizer_d]

    def training_step(self, train_batch, batch_idx):
        x = train_batch

        optimizer_g, optimizer_d = self.optimizers()
        
        optimizer_g.zero_grad()

        # with autocast(enabled=True):
        #     reconstruction, z_mu, z_sigma = self.autoencoderkl(x)

        #     recons_loss = F.l1_loss(reconstruction.float(), x.float())
        #     p_loss = self.perceptual_loss(reconstruction.float(), x.float())
        #     kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        #     kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        #     loss_g = recons_loss + (self.hparams.kl_weight * kl_loss) + (self.hparams.perceptual_weight * p_loss)

        #     if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:
        #         logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
        #         generator_loss = self.adversarial_loss(logits_fake, target_is_real=True, for_discriminator=False)
        #         loss_g += self.hparams.adversarial_weight * generator_loss

        # self.scaler_g.scale(loss_g).backward()
        # self.scaler_g.step(optimizer_g)
        # self.scaler_g.update()

        reconstruction, z_mu, z_sigma = self.autoencoderkl(self.noise_transform(x))

        recons_loss = F.l1_loss(reconstruction.float(), x.float())
        p_loss = self.perceptual_loss(reconstruction.float(), x.float())
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss_g = recons_loss + (self.hparams.kl_weight * kl_loss) + (self.hparams.perceptual_weight * p_loss)

        if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:
            logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = self.adversarial_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g += self.hparams.adversarial_weight * generator_loss

        loss_g.backward()
        optimizer_g.step()
        
        loss_d = 0
        if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:

            # with autocast(enabled=True):
            #     optimizer_d.zero_grad()

            #     logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
            #     loss_d_fake = self.adversarial_loss(logits_fake, target_is_real=False, for_discriminator=True)
            #     logits_real = self.discriminator(x.contiguous().detach())[-1]
            #     loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            #     discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

            #     loss_d = self.hparams.adversarial_weight * discriminator_loss

            # self.scaler_d.scale(loss_d).backward()
            # self.scaler_d.step(optimizer_d)
            # self.scaler_d.update()
            
            optimizer_d.zero_grad()

            logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = self.adversarial_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = self.discriminator(x.contiguous().detach())[-1]
            loss_d_real = self.adversarial_loss(logits_real, target_is_real=True, for_discriminator=True)
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

            loss_d = self.hparams.adversarial_weight * discriminator_loss

            loss_d.backward()
            optimizer_d.step()

        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)

        return {"train_loss_g": loss_g, "train_loss_d": loss_d}
        

    def validation_step(self, val_batch, batch_idx):
        x = val_batch

        # with autocast(enabled=True):
        #     reconstruction, z_mu, z_sigma = self.autoencoderkl(x)
        #     recon_loss = F.l1_loss(x.float(), reconstruction.float())

        reconstruction, z_mu, z_sigma = self.autoencoderkl(x)
        recon_loss = F.l1_loss(x.float(), reconstruction.float())

        self.log("val_loss", recon_loss, sync_dist=True)

        


    def forward(self, images):        
        return self.autoencoderkl(images)



class CycleAutoEncoderKL(pl.LightningModule):
    def __init__(self, autoencoderkl_mr, autoencoderkl_us, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["autoencoderkl_mr", "autoencoderkl_us"])

        self.autoencoderkl_mr = autoencoderkl_mr
        self.autoencoderkl_us = autoencoderkl_us

        self.autoencoderkl_mr_us = nets.AutoencoderKL(
            spatial_dims=2,
            in_channels=3,
            out_channels=3,
            num_channels=(128, 128, 256),
            latent_channels=3,
            num_res_blocks=2,
            attention_levels=(False, False, False),
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
        )

        self.autoencoderkl_us_mr = nets.AutoencoderKL(
            spatial_dims=2,
            in_channels=3,
            out_channels=3,
            num_channels=(128, 128, 256),
            latent_channels=3,
            num_res_blocks=2,
            attention_levels=(False, False, False),
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
        )

        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")

        self.automatic_optimization = False

        self.discriminator_mr = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)        
        self.discriminator_us = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)        

        self.adversarial_loss = PatchAdversarialLoss(criterion="least_squares")

    def encode_mr(self, x: torch.Tensor) -> torch.Tensor:
        z_mu, z_sigma = self.autoencoderkl_mr.autoencoderkl.encode(x)
        z = self.autoencoderkl_mr.autoencoderkl.sampling(z_mu, z_sigma)
        return z

    def decode_mr(self, h: torch.Tensor) -> torch.Tensor:

        z_mu = self.autoencoderkl_mr.autoencoderkl.quant_conv_mu(h)
        z_log_var = self.autoencoderkl_mr.autoencoderkl.quant_conv_log_sigma(h)
        z_log_var = torch.clamp(z_log_var, -30.0, 20.0)
        z_sigma = torch.exp(z_log_var / 2)

        z = self.autoencoderkl_mr.autoencoderkl.sampling(z_mu, z_sigma)
        reconstruction = self.autoencoderkl_mr.autoencoderkl.decode(z)

        return reconstruction

    def encode_us(self, x: torch.Tensor) -> torch.Tensor:
        z_mu, z_sigma = self.autoencoderkl_us.autoencoderkl.encode(x)
        z = self.autoencoderkl_us.autoencoderkl.sampling(z_mu, z_sigma)
        return z

    def decode_us(self, h: torch.Tensor) -> torch.Tensor:

        z_mu = self.autoencoderkl_us.autoencoderkl.quant_conv_mu(h)
        z_log_var = self.autoencoderkl_us.autoencoderkl.quant_conv_log_sigma(h)
        z_log_var = torch.clamp(z_log_var, -30.0, 20.0)
        z_sigma = torch.exp(z_log_var / 2)

        z = self.autoencoderkl_us.autoencoderkl.sampling(z_mu, z_sigma)
        reconstruction = self.autoencoderkl_us.autoencoderkl.decode(z)

        return reconstruction

    def get_us(self, x_mr):
        # 1. Encode the MR using the trained mr encoder
        z_mr = self.encode_mr(x_mr)
        # 2. Pass it through the autoencoder to transform it to the US domain
        z_mr_us, z_mu_mr_us, z_sigma_mr_us = self.autoencoderkl_mr_us(z_mr)
        # 3. Reconstruct an US using the trained US decoder this is the fake MR from US
        reconstruction_mr_us = self.decode_us(z_mr_us)

        return reconstruction_mr_us, z_mu_mr_us, z_sigma_mr_us

    def get_mr(self, x_us):
        z_us = self.encode_us(x_us)        
        z_us_mr, z_mu_us_mr, z_sigma_us_mr = self.autoencoderkl_us_mr(z_us)        
        reconstruction_us_mr = self.decode_mr(z_us_mr)
        return reconstruction_us_mr, z_mu_us_mr, z_sigma_us_mr

    def configure_optimizers(self):
        g_params = list(self.autoencoderkl_mr_us.parameters()) + list(self.autoencoderkl_us_mr.parameters())
        optimizer_g = optim.AdamW(g_params,
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        d_params = list(self.discriminator_mr.parameters()) + list(self.discriminator_us.parameters())
        optimizer_d = optim.AdamW(d_params,
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        return [optimizer_g, optimizer_d]

    def training_step(self, train_batch, batch_idx):
        x_mr, x_us = train_batch

        optimizer_g, optimizer_d = self.optimizers()

        optimizer_g.zero_grad()

        # # MR -> US -> MR
        # # 1. Encode the MR using the trained mr encoder
        # z_mr = self.encode_mr(x_mr)
        # # 2. Pass it through the autoencoder to transform it to the US domain
        # z_mr_us, z_mu_mr_us, z_sigma_mr_us = self.autoencoderkl_mr_us(z_mr)
        # # 3. Reconstruct an US using the trained US decoder this is the fake MR from US
        # reconstruction_mr_us = self.decode_us(z_mr_us)


        reconstruction_mr_us, z_mu_mr_us, z_sigma_mr_us = self.get_us(x_mr)
        reconstruction_mr, z_mu_us_mr, z_sigma_us_mr = self.get_mr(reconstruction_mr_us)

        # z_us = self.encode_us(reconstruction_mr_us)
        # z_us_mr, z_mu_us_mr, z_sigma_us_mr = self.autoencoderkl_us_mr(z_us)
        # reconstruction_mr = self.decode_mr(z_us_mr)

        recons_loss_mr = F.l1_loss(reconstruction_mr.float(), x_mr.float())
        p_loss_mr = self.perceptual_loss(reconstruction_mr.float(), x_mr.float())


        kl_loss_mr_us = 0.5 * torch.sum(z_mu_mr_us.pow(2) + z_sigma_mr_us.pow(2) - torch.log(z_sigma_mr_us.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss_mr_us = torch.sum(kl_loss_mr_us) / kl_loss_mr_us.shape[0]

        kl_loss_us_mr = 0.5 * torch.sum(z_mu_us_mr.pow(2) + z_sigma_us_mr.pow(2) - torch.log(z_sigma_us_mr.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss_us_mr = torch.sum(kl_loss_us_mr) / kl_loss_us_mr.shape[0]

        kl_loss_mr = kl_loss_mr_us + kl_loss_us_mr


        loss_g = recons_loss_mr + (self.hparams.kl_weight * kl_loss_mr) + (self.hparams.perceptual_weight * p_loss_mr)


        if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:
            logits_fake_mr = self.discriminator_mr(reconstruction_mr.contiguous().float())[-1]
            generator_loss_mr = self.adversarial_loss(logits_fake_mr, target_is_real=True, for_discriminator=False)
            loss_g += self.hparams.adversarial_weight * generator_loss_mr
        

        reconstruction_us_mr, z_mu_us_mr, z_sigma_us_mr = self.get_mr(x_us)
        reconstruction_us, z_mu_mr_us, z_sigma_mr_us = self.get_us(reconstruction_us_mr)


        recons_loss_us = F.l1_loss(reconstruction_us.float(), x_us.float())
        p_loss_us = self.perceptual_loss(reconstruction_us.float(), x_us.float())
        kl_loss_us_mr = 0.5 * torch.sum(z_mu_us_mr.pow(2) + z_sigma_us_mr.pow(2) - torch.log(z_sigma_us_mr.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss_us_mr = torch.sum(kl_loss_us_mr) / kl_loss_us_mr.shape[0]

        kl_loss_mr_us = 0.5 * torch.sum(z_mu_mr_us.pow(2) + z_sigma_mr_us.pow(2) - torch.log(z_sigma_mr_us.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss_mr_us = torch.sum(kl_loss_mr_us) / kl_loss_mr_us.shape[0]

        kl_loss_us = kl_loss_us_mr + kl_loss_mr_us

        loss_g += recons_loss_us + (self.hparams.kl_weight * kl_loss_us) + (self.hparams.perceptual_weight * p_loss_us)


        if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:
            logits_fake_us = self.discriminator_us(reconstruction_us.contiguous().float())[-1]
            generator_loss_us = self.adversarial_loss(logits_fake_us, target_is_real=True, for_discriminator=False)
            loss_g += self.hparams.adversarial_weight * generator_loss_us


        loss_g.backward()
        optimizer_g.step()


        loss_d = 0.
        if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:            
            
            optimizer_d.zero_grad()

            logits_fake_mr = self.discriminator_mr(reconstruction_mr.contiguous().detach())[-1]
            loss_d_fake_mr = self.adversarial_loss(logits_fake_mr, target_is_real=False, for_discriminator=True)
            logits_real_mr = self.discriminator_mr(x_mr.contiguous().detach())[-1]
            loss_d_real_mr = self.adversarial_loss(logits_real_mr, target_is_real=True, for_discriminator=True)
            discriminator_loss_mr = (loss_d_fake_mr + loss_d_real_mr) * 0.5

            logits_fake_us = self.discriminator_us(reconstruction_us.contiguous().detach())[-1]
            loss_d_fake_us = self.adversarial_loss(logits_fake_us, target_is_real=False, for_discriminator=True)
            logits_real_us = self.discriminator_us(x_us.contiguous().detach())[-1]
            loss_d_real_us = self.adversarial_loss(logits_real_us, target_is_real=True, for_discriminator=True)
            discriminator_loss_us = (loss_d_fake_us + loss_d_real_us) * 0.5

            loss_d = self.hparams.adversarial_weight * discriminator_loss_mr + self.hparams.adversarial_weight * discriminator_loss_us

            loss_d.backward()
            optimizer_d.step()


        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)


        return {"train_loss_g": loss_g, "train_loss_d": loss_d}
        

    def validation_step(self, val_batch, batch_idx):
        x_mr, x_us = val_batch
        
        reconstruction_mr_us, z_mu_mr_us, z_sigma_mr_us = self.get_us(x_mr)
        reconstruction_mr, z_mu_us_mr, z_sigma_us_mr = self.get_mr(reconstruction_mr_us)
        recon_loss_mr = F.l1_loss(x_mr.float(), reconstruction_mr.float())


        reconstruction_us_mr, z_mu_us_mr, z_sigma_us_mr = self.get_mr(x_us)
        reconstruction_us, z_mu_mr_us, z_sigma_mr_us = self.get_us(reconstruction_us_mr)
        recon_loss_us = F.l1_loss(x_us.float(), reconstruction_us.float())

        recon_loss = recon_loss_mr + recon_loss_us

        self.log("val_loss", recon_loss, sync_dist=True)

        


    def forward(self, images):        
        return self.get_us(images)



class CycleAutoEncoderKLV2(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.autoencoderkl_mr_us = nets.AutoencoderKL(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(128, 128, 256),
            latent_channels=3,
            num_res_blocks=2,
            attention_levels=(False, False, False),
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
        )

        self.autoencoderkl_us_mr = nets.AutoencoderKL(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(128, 128, 256),
            latent_channels=3,
            num_res_blocks=2,
            attention_levels=(False, False, False),
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
        )

        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")

        self.automatic_optimization = False

        self.discriminator_mr = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)        
        self.discriminator_us = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)        

        self.adversarial_loss = PatchAdversarialLoss(criterion="least_squares")

    def get_us(self, x_mr):
        return self.autoencoderkl_mr_us(x_mr)

    def get_mr(self, x_us):
        return self.autoencoderkl_us_mr(x_us)

    def configure_optimizers(self):
        g_params = list(self.autoencoderkl_mr_us.parameters()) + list(self.autoencoderkl_us_mr.parameters())
        optimizer_g = optim.AdamW(g_params,
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        d_params = list(self.discriminator_mr.parameters()) + list(self.discriminator_us.parameters())
        optimizer_d = optim.AdamW(d_params,
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        return [optimizer_g, optimizer_d]

    def training_step(self, train_batch, batch_idx):
        x_mr, x_us = train_batch

        optimizer_g, optimizer_d = self.optimizers()

        optimizer_g.zero_grad()

        # # MR -> US -> MR
        # # 1. Encode the MR using the trained mr encoder
        # z_mr = self.encode_mr(x_mr)
        # # 2. Pass it through the autoencoder to transform it to the US domain
        # z_mr_us, z_mu_mr_us, z_sigma_mr_us = self.autoencoderkl_mr_us(z_mr)
        # # 3. Reconstruct an US using the trained US decoder this is the fake MR from US
        # reconstruction_mr_us = self.decode_us(z_mr_us)


        reconstruction_mr_us, z_mu_mr_us, z_sigma_mr_us = self.get_us(x_mr)
        reconstruction_mr, z_mu_us_mr, z_sigma_us_mr = self.get_mr(reconstruction_mr_us)

        # z_us = self.encode_us(reconstruction_mr_us)
        # z_us_mr, z_mu_us_mr, z_sigma_us_mr = self.autoencoderkl_us_mr(z_us)
        # reconstruction_mr = self.decode_mr(z_us_mr)

        recons_loss_mr = F.l1_loss(reconstruction_mr.float(), x_mr.float())
        p_loss_mr = self.perceptual_loss(reconstruction_mr.float(), x_mr.float())


        kl_loss_mr_us = 0.5 * torch.sum(z_mu_mr_us.pow(2) + z_sigma_mr_us.pow(2) - torch.log(z_sigma_mr_us.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss_mr_us = torch.sum(kl_loss_mr_us) / kl_loss_mr_us.shape[0]

        kl_loss_us_mr = 0.5 * torch.sum(z_mu_us_mr.pow(2) + z_sigma_us_mr.pow(2) - torch.log(z_sigma_us_mr.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss_us_mr = torch.sum(kl_loss_us_mr) / kl_loss_us_mr.shape[0]

        kl_loss_mr = kl_loss_mr_us + kl_loss_us_mr


        loss_g = recons_loss_mr + (self.hparams.kl_weight * kl_loss_mr) + (self.hparams.perceptual_weight * p_loss_mr)
        

        reconstruction_us_mr, z_mu_us_mr, z_sigma_us_mr = self.get_mr(x_us)
        reconstruction_us, z_mu_mr_us, z_sigma_mr_us = self.get_us(reconstruction_us_mr)


        recons_loss_us = F.l1_loss(reconstruction_us.float(), x_us.float())
        p_loss_us = self.perceptual_loss(reconstruction_us.float(), x_us.float())
        kl_loss_us_mr = 0.5 * torch.sum(z_mu_us_mr.pow(2) + z_sigma_us_mr.pow(2) - torch.log(z_sigma_us_mr.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss_us_mr = torch.sum(kl_loss_us_mr) / kl_loss_us_mr.shape[0]

        kl_loss_mr_us = 0.5 * torch.sum(z_mu_mr_us.pow(2) + z_sigma_mr_us.pow(2) - torch.log(z_sigma_mr_us.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss_mr_us = torch.sum(kl_loss_mr_us) / kl_loss_mr_us.shape[0]

        kl_loss_us = kl_loss_us_mr + kl_loss_mr_us

        loss_g += recons_loss_us + (self.hparams.kl_weight * kl_loss_us) + (self.hparams.perceptual_weight * p_loss_us)


        if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:


            logits_fake_mr = self.discriminator_mr(reconstruction_mr.contiguous().float())[-1]
            generator_loss_mr = self.adversarial_loss(logits_fake_mr, target_is_real=True, for_discriminator=False)
            loss_g += self.hparams.adversarial_weight * generator_loss_mr

            logits_fake_mr = self.discriminator_mr(reconstruction_us_mr.contiguous().float())[-1]
            generator_loss_us_mr = self.adversarial_loss(logits_fake_mr, target_is_real=True, for_discriminator=False)
            loss_g += self.hparams.adversarial_weight * generator_loss_us_mr
            
            logits_fake_us = self.discriminator_us(reconstruction_mr_us.contiguous().float())[-1]
            generator_loss_mr_us = self.adversarial_loss(logits_fake_us, target_is_real=True, for_discriminator=False)
            loss_g += self.hparams.adversarial_weight * generator_loss_mr_us

            logits_fake_us = self.discriminator_us(reconstruction_us.contiguous().float())[-1]
            generator_loss_us = self.adversarial_loss(logits_fake_us, target_is_real=True, for_discriminator=False)
            loss_g += self.hparams.adversarial_weight * generator_loss_us


        loss_g.backward()
        optimizer_g.step()


        loss_d = 0.
        if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:            
            
            optimizer_d.zero_grad()

            logits_fake_mr = self.discriminator_mr(reconstruction_mr.contiguous().detach())[-1]
            loss_d_fake_mr = self.adversarial_loss(logits_fake_mr, target_is_real=False, for_discriminator=True)
            logits_real_mr = self.discriminator_mr(x_mr.contiguous().detach())[-1]
            loss_d_real_mr = self.adversarial_loss(logits_real_mr, target_is_real=True, for_discriminator=True)
            logits_fake_us_mr = self.discriminator_mr(reconstruction_us_mr.contiguous().detach())[-1]
            loss_d_fake_us_mr = self.adversarial_loss(logits_fake_us_mr, target_is_real=False, for_discriminator=True)
            discriminator_loss_mr = (loss_d_fake_mr + loss_d_real_mr + loss_d_fake_us_mr) * 0.5

            logits_fake_us = self.discriminator_us(reconstruction_us.contiguous().detach())[-1]
            loss_d_fake_us = self.adversarial_loss(logits_fake_us, target_is_real=False, for_discriminator=True)
            logits_real_us = self.discriminator_us(x_us.contiguous().detach())[-1]
            loss_d_real_us = self.adversarial_loss(logits_real_us, target_is_real=True, for_discriminator=True)
            logits_fake_mr_us = self.discriminator_us(reconstruction_mr_us.contiguous().detach())[-1]
            loss_d_fake_mr_us = self.adversarial_loss(logits_fake_mr_us, target_is_real=False, for_discriminator=True)
            discriminator_loss_us = (loss_d_fake_us + loss_d_real_us + loss_d_fake_mr_us) * 0.5

            loss_d = self.hparams.adversarial_weight * (discriminator_loss_mr + discriminator_loss_us)

            loss_d.backward()
            optimizer_d.step()


        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)


        return {"train_loss_g": loss_g, "train_loss_d": loss_d}
        

    def validation_step(self, val_batch, batch_idx):
        x_mr, x_us = val_batch
        
        reconstruction_mr_us, z_mu_mr_us, z_sigma_mr_us = self.get_us(x_mr)
        reconstruction_mr, z_mu_us_mr, z_sigma_us_mr = self.get_mr(reconstruction_mr_us)
        recon_loss_mr = F.l1_loss(x_mr.float(), reconstruction_mr.float())


        reconstruction_us_mr, z_mu_us_mr, z_sigma_us_mr = self.get_mr(x_us)
        reconstruction_us, z_mu_mr_us, z_sigma_mr_us = self.get_us(reconstruction_us_mr)
        recon_loss_us = F.l1_loss(x_us.float(), reconstruction_us.float())

        recon_loss = recon_loss_mr + recon_loss_us

        self.log("val_loss", recon_loss, sync_dist=True)

        


    def forward(self, images):        
        return self.get_us(images)