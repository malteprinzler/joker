import math
import warnings
import einops
import matplotlib.pyplot as plt
import torchvision.transforms.functional
from diffusers.models import UNet2DConditionModel
from diffusers.configuration_utils import register_to_config
from dataclasses import dataclass
from diffusers.utils import BaseOutput
import torch
from typing import Optional, Tuple, Union
from joker.prior.models.attn_processor import replace_AttnProcessor
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.utils import logging
from typing import Dict, Any
from torch import nn
from diffusers.models.attention import AdaGroupNorm
import torch.nn.functional as F

logger = logging.get_logger(__name__)


@dataclass
class DiffPortrait3DUNet2DConditionOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor
    sattn_hidden_states: dict
    hidden_states_mid_block: torch.Tensor
    hidden_states_down_block_res_samples: tuple


class DiffPortrait3DUNet2DConditionModel(UNet2DConditionModel):
    """
    Reimplementation of the DiffPortrait3DUNet from https://freedomgu.github.io/DiffPortrait3D/ shoutouts to the authors for their great work.
    This reimplementation is largely inspired by the reimplementation from Phong Tran (https://p0lyfish.github.io/). Merci!
    """
    @register_to_config
    def __init__(
            self,
            sample_size: Optional[int] = None,
            in_channels: int = 4,
            out_channels: int = 4,
            center_input_sample: bool = False,
            flip_sin_to_cos: bool = True,
            freq_shift: int = 0,
            down_block_types: Tuple[str] = (
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
            ),
            mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
            up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            only_cross_attention: Union[bool, Tuple[bool]] = False,
            block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
            layers_per_block: Union[int, Tuple[int]] = 2,
            downsample_padding: int = 1,
            mid_block_scale_factor: float = 1,
            act_fn: str = "silu",
            norm_num_groups: Optional[int] = 32,
            norm_eps: float = 1e-5,
            cross_attention_dim: Union[int, Tuple[int]] = 1280,
            encoder_hid_dim: Optional[int] = None,
            attention_head_dim: Union[int, Tuple[int]] = 8,
            dual_cross_attention: bool = False,
            use_linear_projection: bool = False,
            class_embed_type: Optional[str] = None,
            addition_embed_type: Optional[str] = None,
            num_class_embeds: Optional[int] = None,
            upcast_attention: bool = False,
            resnet_time_scale_shift: str = "default",
            resnet_skip_time_act: bool = False,
            resnet_out_scale_factor: int = 1.0,
            time_embedding_type: str = "positional",
            time_embedding_dim: Optional[int] = None,
            time_embedding_act_fn: Optional[str] = None,
            timestep_post_act: Optional[str] = None,
            time_cond_proj_dim: Optional[int] = None,
            conv_in_kernel: int = 3,
            conv_out_kernel: int = 3,
            projection_class_embeddings_input_dim: Optional[int] = None,
            class_embeddings_concat: bool = False,
            mid_block_only_cross_attention: Optional[bool] = None,
            cross_attention_norm: Optional[str] = None,
            addition_embed_type_num_heads=64,
    ):
        super(DiffPortrait3DUNet2DConditionModel, self).__init__(
            sample_size,
            in_channels,
            out_channels,
            center_input_sample,
            flip_sin_to_cos,
            freq_shift,
            down_block_types,
            mid_block_type,
            up_block_types,
            only_cross_attention,
            block_out_channels,
            layers_per_block,
            downsample_padding,
            mid_block_scale_factor,
            act_fn,
            norm_num_groups,
            norm_eps,
            cross_attention_dim,
            encoder_hid_dim,
            attention_head_dim,
            dual_cross_attention,
            use_linear_projection,
            class_embed_type,
            addition_embed_type,
            num_class_embeds,
            upcast_attention,
            resnet_time_scale_shift,
            resnet_skip_time_act,
            resnet_out_scale_factor,
            time_embedding_type,
            time_embedding_dim,
            time_embedding_act_fn,
            timestep_post_act,
            time_cond_proj_dim,
            conv_in_kernel,
            conv_out_kernel,
            projection_class_embeddings_input_dim,
            class_embeddings_concat,
            mid_block_only_cross_attention,
            cross_attention_norm,
            addition_embed_type_num_heads)
        self.changed_satt_processors = self.replace_sattn_processors()

    def replace_sattn_processors(self):
        changed_processors = dict()
        for name, module in self.named_modules():  # attn1 modules are the ones doing self-attention:
            if name.endswith(".attn1"):
                module.processor = replace_AttnProcessor(module.processor)
                changed_processors[name] = module.processor
        if len(list(changed_processors.keys())) == 0:
            warnings.warn("No processors changed!")
        return changed_processors

    def update_changed_satt_processors(self):
        for k in self.changed_satt_processors:
            self.changed_satt_processors[k] = self.attn_processors[k + ".processor"]

    def get_trainable_refnet_parameters(self, return_pnames=False):
        unused_param_names = [
            "conv_out.bias",
            "conv_out.weight",
            "conv_norm_out.bias",
            "conv_norm_out.weight",
            "up_blocks.3.attentions.2.proj_out.bias",
            "up_blocks.3.attentions.2.proj_out.weight",
            "up_blocks.3.attentions.2.transformer_blocks.0.ff.net.2.bias",
            "up_blocks.3.attentions.2.transformer_blocks.0.ff.net.2.weight",
            "up_blocks.3.attentions.2.transformer_blocks.0.ff.net.0.proj.bias",
            "up_blocks.3.attentions.2.transformer_blocks.0.ff.net.0.proj.weight",
            "up_blocks.3.attentions.2.transformer_blocks.0.norm3.bias",
            "up_blocks.3.attentions.2.transformer_blocks.0.norm3.weight",
            "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_out.0.bias",
            "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_out.0.weight",
            "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_v.weight",
            "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_k.weight",
            "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_q.weight",
            "up_blocks.3.attentions.2.transformer_blocks.0.norm2.bias",
            "up_blocks.3.attentions.2.transformer_blocks.0.norm2.weight",
            "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_out.0.bias",
            "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_out.0.weight",
            "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_v.weight",
            "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_k.weight",
            "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_q.weight",
        ]
        trainable_params = [(n if return_pnames else p) for n, p in self.named_parameters() if n not in unused_param_names]
        return trainable_params

    def enable_sattn_recording(self, enable=True):
        for p in self.changed_satt_processors.values():
            p.store_hidden_states = enable

    def disable_sattn_recording(self):
        self.enable_sattn_recording(False)

    def extract_sattn_hidden_states(self):
        ret = dict()
        for n, m in self.changed_satt_processors.items():
            ret[n] = m.get_hidden_states()
        return ret

    def register_additional_sattn_hidden_states(self, d):
        for n, v in d.items():
            self.changed_satt_processors[n].set_hidden_states(v)
            self.changed_satt_processors[n].infere_with_stored_hidden_states = True

    def clear_additional_sattn_hidden_states(self):
        for m in self.changed_satt_processors.values():
            m.clear_storage()
            m.infere_with_stored_hidden_states = False

    def forward(self,
                return_sattn_hidden_states=False,
                return_hidden_states=False,
                additional_sattn_hidden_states=None,
                *args, **kwargs):
        """
        wraps around original unet forward pass to enable reference self-attention feature reading / returning
        """

        if return_sattn_hidden_states:
            self.enable_sattn_recording()
        if additional_sattn_hidden_states is not None:
            additional_sattn_hidden_states = flatten_sattn_refviews(additional_sattn_hidden_states)
            self.register_additional_sattn_hidden_states(additional_sattn_hidden_states)

        forward_result = self.original_forward(*args,
                                               return_hidden_states=return_hidden_states,
                                               **kwargs)

        if return_sattn_hidden_states:
            sattn_hidden_states = self.extract_sattn_hidden_states()
            forward_result.sattn_hidden_states = sattn_hidden_states

        self.clear_additional_sattn_hidden_states()  # important: otherwise, will continue to append additional keys and values during sattn
        self.disable_sattn_recording()
        return forward_result

    def original_forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            mid_block_additional_residual: Optional[torch.Tensor] = None,
            return_dict: bool = True,
            return_hidden_states=False,
    ) -> Union[UNet2DConditionOutput, DiffPortrait3DUNet2DConditionOutput, Tuple]:
        r"""
        COPIED FROM diffusers/models/unet_2d_condition.py:UNet2DConditionModel.forward() but add additional reference feature resnet blocks

        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2 ** self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
            emb = emb + aug_emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None:
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                    down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        ### CHANGED

        if return_hidden_states:
            hidden_states_mid_block = sample
            hidden_states_down_block_res_samples = list(down_block_res_samples)
        else:
            hidden_states_mid_block = None
            hidden_states_down_block_res_samples = None

        ### END OF CHANGED

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return DiffPortrait3DUNet2DConditionOutput(sample=sample, hidden_states_mid_block=hidden_states_mid_block, hidden_states_down_block_res_samples=hidden_states_down_block_res_samples, sattn_hidden_states=None)



def get_ref_sattn_features_n_hiddenstates(ref_unet, vae, ref_pixel_values, encoder_hidden_states, timesteps, num_images_per_prompt=1, do_classifier_free_guidance=False, ref_is_latent=False):
    """
    ref_pixel_values: N x nref x 3 x H x W normalized from 0 ... 1 or latent feature maps N x Nref x 4 x 64 x 64 if ref_is_latent==True
    returned features have shape (if do_classifier_free_guidance * 2) N  * num_image_per_prompt x nref x L x C
    """
    encoder_hidden_states = encoder_hidden_states.view(encoder_hidden_states.shape[0] // num_images_per_prompt, num_images_per_prompt, *encoder_hidden_states.shape[-2:])[:, 0]  # removing duplicates

    vae_dtype = vae.parameters().__next__().dtype
    n, nref, _, h, w = ref_pixel_values.shape

    if ref_is_latent:
        ref_latents = ref_pixel_values.to(vae_dtype).reshape(n * nref, 4, 64, 64)
    else:
        ref_vae_input = (ref_pixel_values * 2 - 1).to(vae_dtype).reshape(n * nref, 3, h, w)  # normalizing to -1 ... 1 as well
        ref_latents = vae.encode(ref_vae_input).latent_dist.sample()
        ref_latents = ref_latents * vae.config.scaling_factor

    if do_classifier_free_guidance:
        ref_latents = ref_latents.repeat(2, 1, 1, 1)  # unconditioning does not extend to reference images
        timesteps = timesteps.repeat(2)
        n *= 2

    expanded_encoder_hidden_states = encoder_hidden_states.repeat_interleave(nref, dim=0)
    expanded_timesteps = timesteps.repeat_interleave(nref, dim=0)
    results = ref_unet(sample=ref_latents,
                       timestep=expanded_timesteps,
                       encoder_hidden_states=expanded_encoder_hidden_states,
                       return_sattn_hidden_states=True,
                       return_hidden_states=True)
    ref_sattn_hidden_states = results.sattn_hidden_states
    hidden_states_mid_block = results.hidden_states_mid_block
    hidden_states_down_block_res_samples = results.hidden_states_down_block_res_samples

    # reshaping keys and values to n x nview x l x c and duplicating sattn features for num_images_per_prompt>1
    for k, v in ref_sattn_hidden_states.items():
        if v is not None:
            nbatchxnviews, l, c = v.shape
            v = v.view(n, nref, l, c)
            v = v.repeat_interleave(num_images_per_prompt, dim=0)
        ref_sattn_hidden_states[k] = v
    hidden_states_mid_block = hidden_states_mid_block.view(n, nref, *hidden_states_mid_block.shape[-3:]).repeat_interleave(num_images_per_prompt, dim=0)
    for i in range(len(hidden_states_down_block_res_samples)):
        hidden_states_down_block_res_samples[i] = hidden_states_down_block_res_samples[i].view(n, nref, *hidden_states_down_block_res_samples[i].shape[-3:]).repeat_interleave(num_images_per_prompt, dim=0)

    return ref_sattn_hidden_states, hidden_states_mid_block, hidden_states_down_block_res_samples



def flatten_sattn_refviews(sattn_feat_dict):
    out_dict = dict()
    for k, v in sattn_feat_dict.items():
        if isinstance(v, dict):
            out_dict[k] = flatten_sattn_refviews(v)
        elif isinstance(v, torch.Tensor):
            N, Nref, L, C = v.shape
            out_dict[k] = v.reshape(N, Nref * L, C)
    return out_dict


