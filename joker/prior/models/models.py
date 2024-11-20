import copy
from operator import attrgetter

from diffusers.utils import randn_tensor, numpy_to_pil
from PIL import Image
from diffusers.utils.pil_utils import PIL_INTERPOLATION
from .diffportrait3d import get_ref_sattn_features_n_hiddenstates
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from diffusers import AutoencoderKL, PNDMScheduler
from .controlnet import ControlNetModel
from transformers import CLIPTextModel
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import (
    _expand_mask,
    CLIPPreTrainedModel,
    CLIPModel,
)
import numpy as np
from .diffportrait3d import DiffPortrait3DUNet2DConditionModel
import types
import torchvision.transforms as T
from joker.prior.models.schedulers import TruncatedDDIMScheduler
import tqdm
from transformers import CLIPTokenizer

inference_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class Joker2DPrior(nn.Module):
    def __init__(self, text_encoder, tokenizer, vae, unet, ref_unet, controlnet, scheduler):
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.vae = vae
        self.unet = unet
        self.ref_unet = ref_unet
        self.controlnet = controlnet
        self.scheduler = scheduler
        self.use_ema = False
        self.ema_param = None
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    @staticmethod
    def from_pretrained():
        text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                     subfolder="text_encoder",
                                                     revision=None).text_model
        tokenizer = CLIPTokenizer.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="tokenizer",
            revision=None,
        )
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        unet = DiffPortrait3DUNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="unet",
            use_reference_resnets=True,
            device_map=None,
            low_cpu_mem_usage=False
        )
        ref_unet = DiffPortrait3DUNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="unet",
            use_reference_resnets=False,
        )
        scheduler = PNDMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                  subfolder="scheduler", )

        controlnet = ControlNetModel.from_unet(ref_unet, conditioning_channels=3)

        return Joker2DPrior(text_encoder=text_encoder, tokenizer=tokenizer, vae=vae, unet=unet, ref_unet=ref_unet, controlnet=controlnet, scheduler=scheduler)

    def apply_model(self, noisy_latents, encoder_hidden_states, timesteps, control_pixel_values, ref_pixel_values=None,
                    ref_sattn_hidden_states=None, num_images_per_prompt=1, do_classifier_free_guidance=False):
        """
        one denoising inference step

        control_pixel_values: N x C x H x W -1 ...+1
        ref_pixel_values: N x Nref x 3 x H x W 0 ...+1
        """
        vae_dtype = self.vae.parameters().__next__().dtype
        control_pixel_values = control_pixel_values.to(vae_dtype)
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=control_pixel_values,
            return_dict=False,
            guess_mode=False,
        )
        if ref_sattn_hidden_states is None:
            ref_sattn_hidden_states, mid_hiddenstates, down_hidden_states = get_ref_sattn_features_n_hiddenstates(
                self.ref_unet, self.vae, ref_pixel_values, encoder_hidden_states, timesteps,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance)
        pred = self.unet(sample=noisy_latents, timestep=timesteps,
                         additional_sattn_hidden_states=ref_sattn_hidden_states,
                         encoder_hidden_states=encoder_hidden_states,
                         down_block_additional_residuals=down_block_res_samples,
                         mid_block_additional_residual=mid_block_res_sample,
                         ).sample
        return pred

    @property
    def device(self):
        return self.unet.device

    @property
    def dtype(self):
        return iter(self.unet.parameters()).__next__().dtype

    def forward(self, batch, noise_scheduler):
        """
        calculates loss for training step
        """
        img = batch["img"]  # N x 3 x H x W, rgb normalized from 0 ... 1
        control = batch["normal"]  # N x C x H x W, normalized from -1 ... 1
        caption_ids = batch["caption_ids"]
        ref_imgs = batch["ref_imgs"]  # N x nref x 3 x H x W, rgb normalized from 0 ... 1
        n, nref, _, h, w = ref_imgs.shape

        vae_dtype = self.vae.parameters().__next__().dtype
        vae_input = img.to(vae_dtype) * 2 - 1

        latents = self.vae.encode(vae_input).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # (bsz, max_num_objects, num_image_tokens, dim)
        encoder_hidden_states = self.text_encoder(caption_ids)[0]  # (bsz, seq_len, dim)

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {noise_scheduler.config.prediction_type}"
            )

        pred = self.apply_model(noisy_latents, encoder_hidden_states, timesteps, control, ref_imgs)

        loss = F.mse_loss(pred.float(), target.float(), reduction="mean")

        return dict(loss=loss)

    @torch.no_grad()
    def infer(self,
              prompt,
              control_map,
              ref_imgs,
              generator=None,
              num_inference_steps=100,
              guidance_scale=3.0,
              negative_prompt: Optional[Union[str, List[str]]] = "",
              num_images_per_prompt: Optional[int] = 1,
              latents: Optional[torch.FloatTensor] = None,
              prompt_embeds: Optional[torch.FloatTensor] = None,
              negative_prompt_embeds: Optional[torch.FloatTensor] = None,
              output_type: Optional[str] = "pt",
              controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
              start_timestep=None,
              progress_bar=False,
              ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            control_map `torch.IntTensor`, List[torch.IntTensor]:
                Controlnet input conditions, images loaded with torchvision.io.read_image and concatenated to N, C x H x W,
                in case of multi-controlnet, may be list of such tensors
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            reference_subject_images torch.Tensor: N x nref x 3 x H x W normalized from 0 ... 1
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original unet. If multiple ControlNets are specified in init, you can set the
                corresponding scale as a list.
            start_timestep:
                Defines the timestep to start the denoising process at. If not none, assumes self.scheduler to be TruncatedDDIM
        """
        assert prompt is not None or prompt_embeds is not None

        # 0. Default height and width to unet
        height = self.unet.config.sample_size * self.vae_scale_factor
        width = self.unet.config.sample_size * self.vae_scale_factor

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size

        ref_imgs = ref_imgs.to(device=self.device, dtype=self.dtype)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode prompt
        prompt_embeds = torch.cat([self.encode_prompt(p) for p in prompt]) if prompt_embeds is None else prompt_embeds
        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.cat([self.encode_prompt(p) for p in negative_prompt]) if negative_prompt_embeds is None else negative_prompt_embeds
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        # 4. Prepare image
        control_map = self.prepare_image(
            image=control_map,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=self.device,
            dtype=self.controlnet.dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guess_mode=False,
        )

        # 5. Prepare timesteps
        if start_timestep is None:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        else:
            assert isinstance(self.scheduler, TruncatedDDIMScheduler)
            self.scheduler.set_timesteps(num_inference_steps, start_step=start_timestep, device=self.device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            self.dtype,
            self.device,
            generator,
            latents,
        )

        # 7. Prepare extra step kwargs. => Not necessary anymore

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with tqdm.tqdm(range(num_inference_steps), disable=~progress_bar) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.apply_model(
                    noisy_latents=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timesteps=t,
                    control_pixel_values=control_map,
                    ref_pixel_values=ref_imgs,
                    num_images_per_prompt=num_images_per_prompt,
                    do_classifier_free_guidance=do_classifier_free_guidance
                )

                # # visualizing prediction
                # pred_norefsattn = self.unet(sample=latent_model_input, timestep=torch.tensor([t], device=device), encoder_hidden_states=prompt_embeds, down_block_additional_residuals=down_block_res_samples, mid_block_additional_residual=mid_block_res_sample).sample
                # x0_pred_latent = noise_pred_2_x0(noise_pred, latent_model_input, self.scheduler, torch.tensor([t], device=device))
                # x0_pred_latent_norefsattn = noise_pred_2_x0(pred_norefsattn, latent_model_input, self.scheduler, torch.tensor([t], device=device))
                # x0_pred_img = decode_latent(x0_pred_latent, self.vae)
                # x0_pred_img_norefsattn = decode_latent(x0_pred_latent_norefsattn, self.vae)
                # xt_img = decode_latent(latent_model_input, self.vae)
                # import matplotlib.pyplot as plt
                # fig, axes = plt.subplots(ncols=5, figsize=(12, 12))
                # axes[0].imshow(gt_img[0].cpu().permute(1, 2, 0) * .5 + .5)
                # axes[1].imshow(xt_img[0])
                # axes[2].imshow(torch.cat(reference_subject_images[0].unbind(0), dim=1).cpu().permute(1, 2, 0))
                # axes[3].imshow(x0_pred_img[0])
                # axes[4].imshow(x0_pred_img_norefsattn[0])
                # plt.title(f"t={t.cpu().item()}")
                # plt.show()

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents,
                ).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        if output_type == "latent":
            image = latents
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)
            # 10. Convert to PIL
            image = numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents, leave_tensor=True)

        return image

    def encode_prompt(self, prompt):
        input_ids = self.tokenize_caption(prompt)
        input_ids = input_ids.to(self.device)

        # augment the text embedding
        embeds = self.text_encoder(input_ids)[0]

        return embeds

    def tokenize_caption(self, caption):
        tokenizer = self.tokenizer
        clean_input_ids = tokenizer.encode(caption)

        max_len = tokenizer.model_max_length

        if len(clean_input_ids) > max_len:
            clean_input_ids = clean_input_ids[:max_len]
        else:
            clean_input_ids = clean_input_ids + [tokenizer.pad_token_id] * (
                    max_len - len(clean_input_ids)
            )

        clean_input_ids = torch.tensor(clean_input_ids, dtype=torch.long)
        clean_input_ids = clean_input_ids.unsqueeze(0)

        return clean_input_ids

    def prepare_image(
            self,
            image,
            width,
            height,
            batch_size,
            num_images_per_prompt,
            device,
            dtype,
            do_classifier_free_guidance=False,
            guess_mode=False,
    ):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, Image):
                image = [image]

            if isinstance(image[0], Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = image_.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents, leave_tensor=False):
        return decode_latents(latents, self.vae, leave_tensor=leave_tensor)


@torch.no_grad()
def noise_pred_2_x0(pred, noisy_latent, scheduler, timesteps, prediction_type=None):
    prediction_type = prediction_type if prediction_type is not None else scheduler.config.prediction_type
    alphas_cumprod = scheduler.alphas_cumprod.to(device=pred.device, dtype=pred.dtype)
    timesteps = timesteps.to(pred.device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(pred.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(pred.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    if prediction_type == "epsilon":
        x_0_pred_latent = (noisy_latent - pred * sqrt_one_minus_alpha_prod) / sqrt_alpha_prod

    elif prediction_type == "v_prediction":
        x_0_pred_latent = noisy_latent * sqrt_alpha_prod - sqrt_one_minus_alpha_prod * pred

    else:
        return NotImplementedError

    return x_0_pred_latent


def decode_latents(latents, vae, leave_tensor=False):
    latents = 1 / vae.config.scaling_factor * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    if not leave_tensor:
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return image
