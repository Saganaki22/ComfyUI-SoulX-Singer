from transformers import LlamaConfig, LlamaModel
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
import math

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.llama.modeling_llama import BaseModelOutputWithPast

# Try to import rotary embedding functions for newer transformers
try:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    NEW_TRANSFORMERS = True
except ImportError:
    NEW_TRANSFORMERS = False


# sinusoidal positional encoding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :] * 1.0
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LlamaAdaptiveRMSNorm(nn.Module):
    def __init__(self, hidden_size=1024, eps=1e-6, dim_cond=1024):
        super().__init__()
        self.to_weight = nn.Linear(dim_cond, hidden_size)
        nn.init.zeros_(self.to_weight.weight)
        nn.init.ones_(self.to_weight.bias)
        self.variance_epsilon = eps
        self._is_hf_initialized = True  # disable automatic init

    def forward(self, hidden_states, cond_embedding):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        weight = self.to_weight(cond_embedding)
        if len(weight.shape) == 2:
            weight = weight.unsqueeze(1)

        return (weight * hidden_states).to(input_dtype)


class LlamaNARDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        """Override to adaptive layer norm"""
        super().__init__(config, layer_idx)  # init attention, mlp, etc.
        self.input_layernorm = LlamaAdaptiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dim_cond=config.hidden_size
        )
        self.post_attention_layernorm = LlamaAdaptiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dim_cond=config.hidden_size
        )

    # add `cond` in forward function
    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_embedding: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(
            hidden_states, cond_embedding=cond_embedding
        )

        # Self Attention
        # Handle both old and new transformers API (4.57+ requires position_embeddings)
        if NEW_TRANSFORMERS:
            # Compute rotary position embeddings for new transformers API
            batch_size, seq_length = hidden_states.shape[:2]
            if position_ids is None:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
            # Compute cos and sin for rotary embeddings
            # Get config values from self_attn
            head_dim = getattr(self.self_attn, 'head_dim', getattr(self.self_attn, 'hidden_size', 1024) // getattr(self.self_attn, 'num_heads', 16))
            max_position_embeddings = getattr(self.self_attn.config, 'max_position_embeddings', 4096)
            base = getattr(self.self_attn.config, 'rope_theta', 10000.0)
            
            # Compute inv_freq
            inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=hidden_states.device) / head_dim))
            
            # Get position frequencies
            freqs = torch.outer(position_ids.flatten().float(), inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            # View as (batch, seq_len, head_dim) - will broadcast across num_heads in apply_rotary_pos_emb
            cos = emb.cos().view(batch_size, seq_length, head_dim)
            sin = emb.sin().view(batch_size, seq_length, head_dim)
            # Cast to match hidden_states dtype (fp32/fp16/bf16)
            cos = cos.to(hidden_states.dtype)
            sin = sin.to(hidden_states.dtype)
            position_embeddings = (cos, sin)
            
            attn_outputs = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )
            # Handle variable return values from different attention implementations
            if len(attn_outputs) == 3:
                hidden_states, self_attn_weights, present_key_value = attn_outputs
            elif len(attn_outputs) == 2:
                hidden_states, self_attn_weights = attn_outputs
                present_key_value = None
            else:
                hidden_states = attn_outputs[0]
                self_attn_weights = None
                present_key_value = None
        else:
            # Old API - no position_embeddings
            attn_outputs = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            # Handle variable return values from different attention implementations
            if len(attn_outputs) == 3:
                hidden_states, self_attn_weights, present_key_value = attn_outputs
            elif len(attn_outputs) == 2:
                hidden_states, self_attn_weights = attn_outputs
                present_key_value = None
            else:
                hidden_states = attn_outputs[0]
                self_attn_weights = None
                present_key_value = None
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(
            hidden_states, cond_embedding=cond_embedding
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class DiffLlama(LlamaModel):
    def __init__(
        self,
        mel_dim=100,
        hidden_size=1024,
        num_heads=16,
        num_layers=16,
        dropout=0.1,
        ffn_dropout=0.1,
        attention_dropout=0.0,
        config=LlamaConfig(0, 256, 1024, 1, 1),
        attention_type="auto",
    ):
        super().__init__(config)

        # Map attention_type to transformers _attn_implementation
        # Note: flash_attention and eager removed due to compatibility issues
        # sageattention will be applied via patching if available
        attn_impl_map = {
            "sdpa": "sdpa",
            "sageattention": "sdpa",  # Use SDPA backend, but we'll patch with sageattn
        }
        attn_implementation = attn_impl_map.get(attention_type, "sdpa")

        # Build layer configs with proper _attn_implementation
        layer_configs = []
        for i in range(num_layers):
            layer_config = LlamaConfig(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                max_position_embeddings=4096,
                intermediate_size=hidden_size * 4,
                attn_implementation=attn_implementation,
            )
            # Also set the private attribute for compatibility
            layer_config._attn_implementation = attn_implementation
            layer_configs.append(layer_config)

        self.layers = nn.ModuleList(
            [
                LlamaNARDecoderLayer(layer_configs[i], layer_idx=i)
                for i in range(num_layers)
            ]
        )

        self.norm = LlamaAdaptiveRMSNorm(hidden_size, dim_cond=hidden_size)

        self.diff_step_embedding = SinusoidalPosEmb(hidden_size)
        self.diff_step_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.cond_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.mel_mlp = nn.Sequential(
            nn.Linear(mel_dim, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.mel_out_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, mel_dim),
        )

        for layer in self.layers:
            layer.input_layernorm = LlamaAdaptiveRMSNorm(
                hidden_size, dim_cond=hidden_size
            )
            layer.post_attention_layernorm = LlamaAdaptiveRMSNorm(
                hidden_size, dim_cond=hidden_size
            )

        self.embed_tokens = None

        self.post_init()

        # self.reset_parameters()

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create noncausal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None

        def _expand_mask(
            mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None
        ):
            """
            Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
            """
            bsz, src_len = mask.size()
            tgt_len = tgt_len if tgt_len is not None else src_len

            expanded_mask = (
                mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
            )

            inverted_mask = 1.0 - expanded_mask

            return inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(dtype).min
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        x,
        diffusion_step,
        cond,
        x_mask,
        input_ids: torch.LongTensor = None,  # [num_quant, B, T]
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # retrieve some shape info
        batch_size, seq_length, _ = x.shape

        # condtion mlp
        cond_embedding = self.cond_mlp(cond)  # (B, T, C)

        # condition mel
        x = self.mel_mlp(x)

        # diffusion step embedding
        diffusion_step = self.diff_step_embedding(diffusion_step).to(x.device)
        # Cast to match x dtype (handles fp16/fp32/bf16)
        diffusion_step = diffusion_step.to(x.dtype)
        diffusion_step = self.diff_step_mlp(diffusion_step)  # (B, C)
        x = x + cond_embedding

        inputs_embeds = x
        attention_mask = x_mask

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        all_layer_hidden_states = []

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:
                raise NotImplementedError

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cond_embedding=diffusion_step,
                )

            hidden_states = layer_outputs[0]
            all_layer_hidden_states.append(hidden_states.clone())

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states, cond_embedding=diffusion_step)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        hidden_states = self.mel_out_mlp(hidden_states)

        # if not return_dict:
        #     return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        # return BaseModelOutputWithPast(
        #     last_hidden_state=hidden_states,
        #     past_key_values=next_cache,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attns,
        # )
        if return_dict:
            return {
                "output": hidden_states,
                "hidden_states": all_layer_hidden_states,
            }

        return hidden_states
