import itertools
import pdb
import random
import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm
import cv2
import numpy as np
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler
from model.VQGAN.vqgan import VQModel

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

import torch
import torch.nn as nn
import torch.nn.functional as F


class CameraConditionedEncoder(nn.Module):
    """
    Wraps an encoder to inject camera info into its latent representation.
    - cam: [B, 6]
    - x:   [B, C, H, W]
    """
    def __init__(
        self,
        latent_ch: int = 3,
        cam_dim: int = 6,
        mlp_hidden: int = 64,
        out_cam_dim: int = 10,
    ):
        super().__init__()
        self.out_cam_dim = out_cam_dim

        # ---- 3-layer MLP for camera values ----
        self.cam_mlp = nn.Sequential(
            nn.Linear(cam_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, out_cam_dim),
        )

        # always have a projection layer
        self.cam_proj = nn.Conv2d(
            latent_ch + out_cam_dim,
            latent_ch,
            kernel_size=1,
        )

    def forward(self, z: torch.Tensor, cam: torch.Tensor) -> torch.Tensor:
        """
        x   : [B, C, H, W]   (image)
        cam : [B, cam_dim]   (camera)
        """
        # 1) Encode image
        B, C, H, W = z.shape

        # 2) Pass camera through MLP
        cam_feat = self.cam_mlp(cam)        # [B, out_cam_dim]
        cam_feat = cam_feat.view(B, -1, 1, 1).expand(-1, -1, H, W)

        # 3) Concat along channel axis
        z_cat = torch.cat([z, cam_feat], dim=1)  # [B, C + out_cam_dim, H, W]

        # 4) Project back to latent_ch
        z_proj = self.cam_proj(z_cat)

        return z_proj


class LatentBrownianBridgeModel(BrownianBridgeModel):
    def __init__(self, model_config):
        super().__init__(model_config)

        self.vqgan = VQModel(**vars(model_config.VQGAN.params)).eval()
        self.vqgan.train = disabled_train
        for param in self.vqgan.parameters():
            param.requires_grad = False
        print(f"load vqgan from {model_config.VQGAN.params.ckpt_path}")        
        # Condition Stage Model
        if self.condition_key == "nocond":
            self.cond_stage_model = None
        elif self.condition_key == "first_stage":
            self.cond_stage_model = self.vqgan
        elif self.condition_key == "SpatialRescaler":
            self.cond_stage_model = SpatialRescaler(
                **vars(model_config.CondStageParams)
            )
        else:
            raise NotImplementedError
        
        self.cam_encoder = CameraConditionedEncoder() if model_config.CameraProj.use_cam_proj else None


    def get_ema_net(self):
        return self

    def get_parameters(self):
        if self.condition_key == "SpatialRescaler":
            print("get parameters to optimize: SpatialRescaler, UNet")
            params = itertools.chain(
                self.denoise_fn.parameters(), self.cond_stage_model.parameters()
            )
        else:
            print("get parameters to optimize: UNet")
            params = self.denoise_fn.parameters()
        return params

    def apply(self, weights_init):
        super().apply(weights_init)
        if self.cond_stage_model is not None:
            self.cond_stage_model.apply(weights_init)
        return self

    def _context_loss(self, decoded_image, x, target_mask, lbda: float = 1.0):
        
        loss = (
            torch.nn.functional.l1_loss(
                decoded_image * (1 - target_mask), x * (1 - target_mask)
            )
            * lbda
        )

        return loss
        
    def resize_and_dilate_masks(self, masks, factor=4, kernel_size = 3, iterations = 3):
        B, C, H, W = masks.shape
        assert C == 1, "This function expects masks with 1 channel"

        new_H, new_W = H // factor, W // factor
        resized_masks = []
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

        for i in range(B):
            # Convert tensor to numpy
            mask_np = masks[i, 0].cpu().numpy().astype(np.uint8)

            # Resize with nearest neighbor
            resized_np = cv2.resize(mask_np, (new_W, new_H), interpolation=cv2.INTER_NEAREST)

            # Apply dilation
            dilated_np = cv2.dilate(resized_np, kernel, iterations=iterations)
            dilated_float = (dilated_np > 0).astype(np.float32)
            # Convert back to tensor and add channel dim
            resized_tensor = torch.from_numpy(dilated_float).unsqueeze(0)  # shape: (1, new_H, new_W)
            resized_masks.append(resized_tensor)

        # Stack into batch
        return torch.stack(resized_masks)  # shape: (B, 1, new_H, new_W)
    
    def resize_and_blur_masks(self, masks, factor=4, kernel_size=3, sigma=1.0):
        B, C, H, W = masks.shape
        assert C == 1, "This function expects masks with 1 channel"

        new_H, new_W = H // factor, W // factor
        resized_masks = []

        for i in range(B):
            # Convert tensor to numpy
            mask_np = masks[i, 0].cpu().numpy().astype(np.uint8)

            # Resize with nearest neighbor (to preserve binary mask structure)
            resized_np = cv2.resize(mask_np, (new_W, new_H), interpolation=cv2.INTER_NEAREST).astype(np.float32)
            # Apply Gaussian blur
            blurred_np = cv2.GaussianBlur(resized_np, (kernel_size, kernel_size), sigma).astype(np.float32)
            feathered_mask = blurred_np.copy()
            feathered_mask[resized_np > 0] = resized_np[resized_np > 0]
            # Convert back to tensor and add channel dim
            blurred_tensor = torch.from_numpy(feathered_mask).unsqueeze(0)  # shape: (1, new_H, new_W)
            resized_masks.append(blurred_tensor)

        # Stack into batch
        return torch.stack(resized_masks)  # shape: (B, 1, new_H, new_W)


    def forward(self, x, x_mask, x_cond, x_cond_mask, loss_type = 'general', context=None, lambda_fg=1.0, lambda_bg=1.0, cam_params = None):
        # x = x_0 = franka image
        # x_cond = x_T = xArm image

        with torch.no_grad():
            x_latent = self.encode(x, cond=False)
            x_cond_latent = self.encode(x_cond, cond=True)
        if cam_params is not None and self.cam_encoder is not None:
            x_cond_latent = self.cam_encoder(x_cond_latent, cam_params)

        context = self.get_cond_stage_context(x_cond)  # None
        latent_loss, log_dict = super().forward(x_latent.detach(), x_cond_latent.detach(), context, self.resize_and_dilate_masks(x_mask,kernel_size=2,iterations=1),self.resize_and_dilate_masks(x_cond_mask,kernel_size=2,iterations=1))

        if loss_type == 'latent-mask-loss':
            return latent_loss,log_dict

        x0_recon = log_dict["x0_recon"]

        decoded_image = self.decode(x0_recon, cond=False)  

        bin_x_mask = (x_mask > 0.5)
        bin_x_cond_mask = (x_cond_mask > 0.5)

        if loss_type == 'general':
            fg_mask = bin_x_mask.to(device = x0_recon.device, dtype = torch.float32)
            bg_mask = 1 - fg_mask

        elif loss_type == 'union':
            mask_union = torch.logical_or(bin_x_mask,bin_x_cond_mask)
            fg_mask = mask_union.to(device = x0_recon.device, dtype = torch.float32)
            bg_mask = 1 - fg_mask

        else:
            mask_union = torch.logical_or(bin_x_mask,bin_x_cond_mask)
            fg_mask = bin_x_mask.to(device = x0_recon.device, dtype = torch.float32)
            bg_mask = torch.logical_not(mask_union).to(device = x0_recon.device, dtype = torch.float32)

        print(bin_x_mask)
        print(fg_mask)
        print(bin_x_cond_mask)
        print(bg_mask)

        # x_mask = x_mask.to(x0_recon.device)
        # x_cond_mask = x_cond_mask.to(x0_recon.device)

        # context_loss = self._context_loss(decoded_image, x, x_mask, lambda_context)
    
        # Foreground: match robot B (target)
        foreground_loss = torch.nn.functional.l1_loss(
            decoded_image * fg_mask, x * fg_mask
        )
        
        # Background: match background A (input)
        background_loss = torch.nn.functional.l1_loss(
            decoded_image * bg_mask, x_cond * bg_mask
        )

        loss = lambda_fg * foreground_loss + lambda_bg * background_loss

        print(
            f"foreground_loss: {foreground_loss.item()}, background_loss: {background_loss.item()}"
        )

        # print(
        #     f"loss: {loss.item()}, context_loss: {context_loss.item()}, lambda_context: {lambda_context}"
        # )

        # loss += context_loss

        return loss, log_dict

    # def forward(self, x, x_mask, x_cond, context=None, lambda_fg=1.0, lambda_bg=1.0):
    #     # x = x_0 = franka image
    #     # x_cond = x_T = xArm image

    #     with torch.no_grad():
    #         x_latent = self.encode(x, cond=False)
    #         x_cond_latent = self.encode(x_cond, cond=True)

    #     context = self.get_cond_stage_context(x_cond)  # None

    #     _, log_dict = super().forward(
    #         x_latent.detach(), x_cond_latent.detach(), context
    #     )

    #     x0_recon = log_dict["x0_recon"]
    #     decoded_image = self.decode(x0_recon, cond=False)

    #     # Foreground: match robot B (target)
    #     foreground_loss = torch.nn.functional.l1_loss(
    #         decoded_image * x_mask, x * x_mask
    #     )
    #     # Background: match background A (input)
    #     background_loss = torch.nn.functional.l1_loss(
    #         decoded_image * (1 - x_mask), x_cond * (1 - x_mask)
    #     )

    #     loss = lambda_fg * foreground_loss + lambda_bg * background_loss

    #     print(
    #         f"foreground_loss: {foreground_loss.item()}, background_loss: {background_loss.item()}"
    #     )

    #     return loss, log_dict

    def get_cond_stage_context(self, x_cond):
        if self.cond_stage_model is not None:
            context = self.cond_stage_model(x_cond)
            if self.condition_key == "first_stage":
                context = context.detach()
        else:
            context = None
        return context

    @torch.no_grad()
    def encode(self, x, cond=True, normalize=None):
        normalize = (
            self.model_config.normalize_latent if normalize is None else normalize
        )
        model = self.vqgan
        x_latent = model.encoder(x)
        if not self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        if normalize:
            if cond:
                x_latent = (x_latent - self.cond_latent_mean) / self.cond_latent_std
            else:
                x_latent = (x_latent - self.ori_latent_mean) / self.ori_latent_std
        return x_latent

    def decode(self, x_latent, cond=True, normalize=None):
        normalize = (
            self.model_config.normalize_latent if normalize is None else normalize
        )
        if normalize:
            if cond:
                x_latent = x_latent * self.cond_latent_std + self.cond_latent_mean
            else:
                x_latent = x_latent * self.ori_latent_std + self.ori_latent_mean
        model = self.vqgan
        if self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        x_latent_quant, loss, _ = model.quantize(x_latent)
        out = model.decode(x_latent_quant)
        return out

    @torch.no_grad()
    def sample(self, x_cond, clip_denoised=False, sample_mid_step=False, cam_params = None):
        x_cond_latent = self.encode(x_cond, cond=True)
        if cam_params is not None and self.cam_encoder is not None:
            x_cond_latent = self.cam_encoder(x_cond_latent, cam_params)

        if sample_mid_step:
            temp, one_step_temp = self.p_sample_loop(
                y=x_cond_latent,
                context=self.get_cond_stage_context(x_cond),
                clip_denoised=clip_denoised,
                sample_mid_step=sample_mid_step,
            )
            out_samples = []
            for i in tqdm(
                range(len(temp)),
                initial=0,
                desc="save output sample mid steps",
                dynamic_ncols=True,
                smoothing=0.01,
            ):
                with torch.no_grad():
                    out = self.decode(temp[i].detach(), cond=False)
                out_samples.append(out.to("cpu"))

            one_step_samples = []
            for i in tqdm(
                range(len(one_step_temp)),
                initial=0,
                desc="save one step sample mid steps",
                dynamic_ncols=True,
                smoothing=0.01,
            ):
                with torch.no_grad():
                    out = self.decode(one_step_temp[i].detach(), cond=False)
                one_step_samples.append(out.to("cpu"))
            return out_samples, one_step_samples
        else:
            # model's output (domain Y) during evals
            temp = self.p_sample_loop(
                y=x_cond_latent,
                context=self.get_cond_stage_context(x_cond),
                clip_denoised=clip_denoised,
                sample_mid_step=sample_mid_step,
            )
            x_latent = temp
            out = self.decode(x_latent, cond=False)
            return out

    @torch.no_grad()
    def sample_vqgan(self, x):
        x_rec, _ = self.vqgan(x)
        return x_rec

    # @torch.no_grad()
    # def reverse_sample(self, x, skip=False):
    #     x_ori_latent = self.vqgan.encoder(x)
    #     temp, _ = self.brownianbridge.reverse_p_sample_loop(x_ori_latent, x, skip=skip, clip_denoised=False)
    #     x_latent = temp[-1]
    #     x_latent = self.vqgan.quant_conv(x_latent)
    #     x_latent_quant, _, _ = self.vqgan.quantize(x_latent)
    #     out = self.vqgan.decode(x_latent_quant)
    #     return out
