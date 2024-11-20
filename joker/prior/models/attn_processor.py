from diffusers.models.attention_processor import Attention, AttnProcessor2_0, AttnProcessor
import torch
import torch.nn.functional as F


class ManipulativeAttnProcessor2_0(AttnProcessor):
    """ largely copied from diffusers/models/attention_processor.py:AttnProcessor2_0 but allows to store /insert self-attention keys and values"""

    def __init__(self, store_hidden_states=False, infere_with_stored_hidden_states=False):
        assert not (store_hidden_states and infere_with_stored_hidden_states)
        self.store_hidden_states = store_hidden_states
        self.infere_with_stored_hidden_states = infere_with_stored_hidden_states
        self.stored_hidden_states = None
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        inner_dim = hidden_states.shape[-1]

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is not None and (self.store_hidden_states or self.infere_with_stored_hidden_states):
            raise ValueError("Implementation is only intended for self-attention modules! Cross-attention module detected!")

        if encoder_hidden_states is None:  # storing / loading only for self-attention layers
            encoder_hidden_states = hidden_states
            if self.store_hidden_states:
                self.set_hidden_states(hidden_states)

            if self.infere_with_stored_hidden_states:
                encoder_hidden_states = torch.cat([encoder_hidden_states, self.stored_hidden_states], dim=-2)

        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    def set_hidden_states(self, hidden_states):
        # should have shape N x L' x C
        self.stored_hidden_states = hidden_states

    def get_hidden_states(self):
        return self.stored_hidden_states

    def clear_storage(self):
        self.stored_hidden_states = None


def replace_AttnProcessor(attnprocessor_module):
    if isinstance(attnprocessor_module, AttnProcessor):
        return ManipulativeAttnProcessor()
    elif isinstance(attnprocessor_module, AttnProcessor2_0):
        return ManipulativeAttnProcessor2_0()
    else:
        raise NotImplementedError(f"Not implemented for AttnProcessor module {type(attnprocessor_module)}")
