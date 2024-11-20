import sys

from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import List
from diffusers import AutoencoderKL

import einops
import torch
import torch.nn.functional as F
from torchvision.transforms._functional_tensor import resize
from threestudio.models.guidance.lpips import LPIPS
from diffusers import DDPMScheduler
import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *
from threestudio.schedulers.truncated_ddim import TruncatedDDIMScheduler

from joker.prior.models.models import Joker2DPrior
from omegaconf import OmegaConf


@threestudio.register("joker-guidance")
class JokerGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        ckpt_path: Optional[
            str
        ] = None  # path to local checkpoint (None for loading from url)
        config_path: Optional[
            str
        ] = None  # path to local config (None for loading from url)
        guidance_scale: float = 50.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        camera_condition_type: str = "rotation"
        view_dependent_prompting: bool = False

        n_target_views: int = 400
        image_size: int = 512
        denoising_steps: int = 1
        ddim_steps: int = 100
        last_ddim_steps_at_once: int = 0
        w_lpips: float = 1.

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Joker 2D Prior Criterion ...")
        self.model = Joker2DPrior.from_pretrained()
        self.model.load_state_dict(torch.load(self.cfg.ckpt_path, map_location=self.device))
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")  # important to reduce vae drift during optimization in rgb space
        self.model.vae.to(self.device)
        self.model.to(torch.bfloat16)
        self.dtype = self.model.vae.parameters().__next__().dtype

        self.noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        self.multistep_scheduler = TruncatedDDIMScheduler.from_config(self.noise_scheduler.config)
        self.multistep_scheduler.set_timesteps(self.cfg.ddim_steps)
        self.denoising_timesteps = self.multistep_scheduler.timesteps.cpu().tolist()

        self.num_train_timesteps = 1000
        self.lpips = LPIPS(net_type="vgg", normalize=True)
        self.lpips.requires_grad_(False)

        self.target_imgs = torch.zeros(self.cfg.n_target_views, 3, 512, 512, device=self.device, dtype=torch.float32)

        self.to(self.device)

        threestudio.info(f"Loaded Joker 2D Prior Criterion successfully!")

    def encode_images(self, imgs):
        """
        encodes rgb images of shape N x 3 x H x W 0...1 from rgb space to latent space using vae
        """
        imgs = imgs * 2 - 1

        vae_dtype = self.model.vae.parameters().__next__().dtype
        vae_input = imgs.to(vae_dtype)

        latents = self.model.vae.encode(vae_input).latent_dist.sample()  # using mean instead of sample() to reduce vae drift during optimization
        latents = latents * self.model.vae.config.scaling_factor
        return latents

    def decode_latents(self, latents, leave_tensor=False):
        """
        decodes latent into rgb with range 0 ... 1

        if leave_tensor: returns N x 3 x H x W torch tensor
        else: returns N x H x W x 3 np.array
        """
        latents = 1 / self.model.vae.config.scaling_factor * latents
        image = self.model.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        if not leave_tensor:
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image


    def calculate_loss(self, rgb: Float[Tensor, "B H W C"], sample_idcs: List[int], rays_uv: Float[Tensor, "B H W 2"], height: int, width: int):
        """
        """
        B, H, W, C = rgb.shape
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        target_batch = self.target_imgs[sample_idcs]
        B_target, C_target, H_target, W_target = target_batch.shape
        if height != H_target or width != W_target:
            target_batch = resize(target_batch, [height, width], interpolation="area")
        batch_idx_helper = torch.arange(B).view(-1, 1, 1).expand(B, H, W)
        target_batch = target_batch[batch_idx_helper, :, rays_uv[..., 1], rays_uv[..., 0]]  # N H W C 0...1
        target_batch_BCHW = einops.rearrange(target_batch, "B H W C -> B C H W")
        lpips_loss = self.lpips(rgb_BCHW, target_batch_BCHW) if self.cfg.w_lpips > 0 else 0.
        l1_loss = F.l1_loss(rgb_BCHW, target_batch_BCHW, reduction="mean")
        loss = self.cfg.w_lpips * lpips_loss + l1_loss

        # # visualization
        # fig, axes = plt.subplots(ncols=8, nrows=8, figsize=(2*20, 20))
        # axes = axes.flatten()
        # for i in range(64):
        #     axes[i].imshow(torch.cat([rgb_BCHW.permute(0, 2, 3, 1)[i].detach().cpu().float(), target_batch_BCHW.permute(0, 2, 3, 1)[i].detach().cpu().float()], dim=1))
        # [a.axis("off") for a in axes]
        # plt.tight_layout()
        # plt.show()

        # corrective scaling term to be at approximately same scale as sds loss with resolution 64x64, dynamic range~-1...+1, 4 channels
        return loss

    @torch.no_grad()
    def update_targets(self, rgb: Float[Tensor, "B H W C"],
                       prompt_utils: PromptProcessorOutput,
                       ctrl,  # ctrl pixel values B, H, W, 3 0...+1
                       ref,  # ref pixel values B, nref, H, W, 3 0...1
                       timestep,
                       rgb_as_latents: bool = False,
                       text_embeddings=None,
                       sample_idcs=None,
                       nsteps=None,
                       **kwargs):

        if nsteps is None:
            nsteps = min(self.cfg.denoising_steps, timestep - 1)
        # get n denoising timesteps such that distance between steps is approximately constant during optimization and this distance should roughly correspond to ddim sampling from pure noise

        print(f"UPDATING TARGETS FROM {timestep} WITH {nsteps} STEPS.")
        target_latents = self.infere_2dprior(rgb=rgb, ctrl=ctrl, ref=ref, text_embeddings=text_embeddings,
                                             prompt_utils=prompt_utils, timestep=timestep, input_is_latent=False,
                                             rgb_as_latents=rgb_as_latents, denoising_steps=nsteps)
        target_imgs = self.decode_latents(target_latents, leave_tensor=True)  # N C H W 0...1
        # rgb_BCHW = rgb.permute(0, 3, 1, 2)
        # target_imgs = torch.zeros_like(rgb_BCHW)
        for i, idx in enumerate(sample_idcs):
            self.target_imgs[idx] = target_imgs[i]
        return self.target_imgs

    def infere_2dprior(self, rgb: Float[Tensor, "B H W C"], ctrl, ref, text_embeddings, prompt_utils, timestep, input_is_latent, rgb_as_latents, denoising_steps=None):
        """
        inference loop to add noise and denoise 'rgb' for 'denoising_steps' starting from 'timestep'
        """
        batch_size = len(rgb)
        denoising_steps = denoising_steps if denoising_steps is not None else self.cfg.denoising_steps
        ctrl = einops.rearrange(ctrl * 2 - 1, "b h w c -> b c h w")  # control signal must be normalized to -1 ... 1
        ref = einops.rearrange(ref, "b nref h w c -> b nref c h w")
        if text_embeddings is None:
            assert not self.cfg.view_dependent_prompting
            elevation = torch.zeros(batch_size)
            azimuth = torch.zeros(batch_size)
            camera_distances = torch.zeros(batch_size)
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )

        if input_is_latent:
            latents = rgb
        else:
            rgb_BCHW = rgb.permute(0, 3, 1, 2)
            latents: Float[Tensor, "B 4 64 64"]
            if rgb_as_latents:
                latents = (
                        F.interpolate(
                            rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
                        )
                        * 2
                        - 1
                )
            else:
                # interp to 512x512 to be fed into vae.
                if rgb_BCHW.shape[-1] != self.cfg.image_size:
                    pred_rgb = F.interpolate(
                        rgb_BCHW,
                        (self.cfg.image_size, self.cfg.image_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    pred_rgb = rgb_BCHW
                # encode image into latents with vae, requires grad!
                latents = self.encode_images(pred_rgb)

        # sample timestep
        if timestep is None:
            t_start = torch.randint(
                self.min_step,
                self.max_step + 1,
                [1],
                dtype=torch.long,
                device=latents.device,
            )
        else:
            assert timestep >= 0 and timestep < self.num_train_timesteps
            t_start = torch.full([1], timestep, dtype=torch.long, device=latents.device)
        self.multistep_scheduler.set_timesteps(num_inference_steps=denoising_steps, start_step=t_start[0].cpu().item())

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            ctrl_expand = ctrl.to(self.dtype).repeat(2, 1, 1, 1)
            text_embeddings = text_embeddings.to(self.dtype)

            # add noise
            # gen = torch.Generator(latents.device).manual_seed(0)
            # noise = torch.randn(latents.shape, generator=gen, device=latents.device).to(latents)
            noise = torch.randn_like(latents)
            latents_pretrain = self.noise_scheduler.add_noise(latents, noise, t_start)
            for t in self.multistep_scheduler.timesteps:
                t_expand = t.repeat(batch_size * 2).to(self.device)
                # pred noise
                latent_model_input = torch.cat([latents_pretrain] * 2).to(self.dtype)
                # save input tensors for UNet
                noise_pred = self.model.apply_model(latent_model_input, text_embeddings, t_expand, ctrl_expand, ref_pixel_values=ref[:1].to(self.dtype).expand(len(t_expand), -1, -1, -1, -1))

                # perform guidance
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(
                    2
                )  # Note: flipped compared to stable-dreamfusion

                noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                )

                latents_pretrain = self.multistep_scheduler.step(noise_pred, t, latents_pretrain).prev_sample

                # # debugging vae drift
                # def autodecode_fn(x):
                #     # return self.model.vae.encode(self.model.vae.decode(1 / self.model.vae.config.scaling_factor * x).sample).latent_dist.mean * self.model.vae.config.scaling_factor
                #     return self.encode_images(self.decode_latents(x, leave_tensor=True))
                #
                # denoise_results = self.multistep_scheduler.step(noise_pred, t, latents_pretrain, autodecode_fn=autodecode_fn)
                # latents_pretrain = denoise_results.prev_sample
                # if t.cpu().item() == 260:
                #     fig = plt.figure()
                #     plt.imshow(self.decode_latents(denoise_results.pred_original_sample)[0])
                #     plt.title(t.cpu().item())
                #     plt.show()
                #     plt.close(fig)

            # # visualize predictions
            # with torch.no_grad():
            #     nerf_pred_img = self.decode_latents(latents.detach())
            #     diffusion_pred_img = self.decode_latents(latents_recon.detach())
            #     ncols = len(nerf_pred_img)
            #     nrows=2
            #     fig, axes = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False, figsize=(6*ncols, 6*nrows))
            #     for i in range(ncols):
            #         axes[0, i].imshow(nerf_pred_img[i])
            #         axes[1, i].imshow(diffusion_pred_img[i])
            #     [ax.axis("off") for ax in axes.flatten()]
            #     plt.tight_layout()
            #     plt.show()
        return latents_pretrain

    def forward(
            self,
            rgb: Float[Tensor, "B H W C"],
            prompt_utils: PromptProcessorOutput,
            ctrl=None,  # ctrl pixel values B, H, W, 3 0...+1
            ref=None,  # ref pixel values B, nref, H, W, 3 0...1
            rgb_as_latents: bool = False,
            timestep=None,
            text_embeddings=None,
            input_is_latent=False,
            sample_idcs=None,
            rays_uv=None,
            height=None,
            width=None,
            **kwargs,
    ):
        loss = self.calculate_loss(rgb, sample_idcs, rays_uv, height=height, width=width)
        return {"loss_target": loss}
