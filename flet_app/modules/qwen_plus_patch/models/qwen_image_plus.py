import torch

from models.qwen_image import QwenImagePipeline


class QwenImagePlusPipeline(QwenImagePipeline):
    name = 'qwen_image_plus'

    def get_call_vae_fn(self, vae):
        def fn(*args):
            image = args[0]
            latents = vae.encode(image.to(vae.device, vae.dtype)).latent_dist.mode()
            latents = (latents - vae.latents_mean_tensor) / vae.latents_std_tensor
            result = {'latents': latents}
            if len(args) > 1:
                control_image = args[1]
                control_latents = vae.encode(control_image.to(vae.device, vae.dtype)).latent_dist.mode()
                control_latents = (control_latents - vae.latents_mean_tensor) / vae.latents_std_tensor
                result['control_latents'] = control_latents
            if len(args) > 2:
                reference_image = args[2]
                reference_latents = vae.encode(reference_image.to(vae.device, vae.dtype)).latent_dist.mode()
                reference_latents = (reference_latents - vae.latents_mean_tensor) / vae.latents_std_tensor
                result['reference_latents'] = reference_latents
            return result
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(caption, is_video, control_file: list[str] | None, reference_file: list[str] | None):
            # args are lists
            assert not any(is_video)
            # For triple image training, the control_file is the source image and the reference_file is the reference image.
            # The text encoder only needs the reference_file.
            prompt_embeds = self._get_qwen_prompt_embeds(caption, reference_file, device=text_encoder.device)
            return {'prompt_embeds': prompt_embeds}
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        prompt_embeds = inputs['prompt_embeds']
        mask = inputs['mask']
        device = latents.device

        # prompt embeds are variable length
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.bool, device=device) for e in prompt_embeds]
        max_seq_len = max([e.size(0) for e in prompt_embeds])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in prompt_embeds]
        )
        prompt_embeds_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        max_text_len = prompt_embeds_mask.sum(dim=1).max().item()
        prompt_embeds = prompt_embeds[:, :max_text_len, :]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_text_len]

        bs, channels, num_frames, h, w = latents.shape

        num_channels_latents = self.transformer.config.in_channels // 4
        assert num_channels_latents == channels
        latents = self._pack_latents(latents, bs, num_channels_latents, h, w)

        if mask is not None:
            mask = mask.unsqueeze(1).expand((-1, num_channels_latents, -1, -1))  # make mask (bs, c, img_h, img_w)
            mask = torch.nn.functional.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
            mask = mask.unsqueeze(2)  # add frame dimension
            mask = self._pack_latents(mask, bs, num_channels_latents, h, w)

        timestep_sample_method = self.model_config.get('timestep_sample_method', 'logit_normal')

        if timestep_sample_method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        elif timestep_sample_method == 'uniform':
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError()

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=device))
        else:
            t = dist.sample((bs,)).to(device)

        if timestep_sample_method == 'logit_normal':
            sigmoid_scale = self.model_config.get('sigmoid_scale', 1.0)
            t = t * sigmoid_scale
            t = torch.sigmoid(t)

        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)
        elif self.model_config.get('flux_shift', False):
            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
            t = time_shift(mu, 1.0, t)

        x_1 = latents
        x_0 = torch.randn_like(x_1)
        t_expanded = t.view(-1, 1, 1)
        x_t = (1 - t_expanded) * x_1 + t_expanded * x_0
        target = x_0 - x_1

        img_shapes = [(1, h // 2, w // 2)]
        
        extra = tuple()
        if 'control_latents' in inputs:
            control_latents = inputs['control_latents'].float()
            control_latents = self._pack_latents(control_latents, bs, num_channels_latents, h, w)
            assert control_latents.shape == latents.shape, (control_latents.shape, latents.shape)
            img_seq_len = torch.tensor(x_t.shape[1], device=x_t.device).repeat((bs,))
            extra = (img_seq_len,)
            x_t = torch.cat([x_t, control_latents], dim=1)
            img_shapes.append((1, h // 2, w // 2))

        if 'reference_latents' in inputs:
            reference_latents = inputs['reference_latents'].float()
            reference_latents = self._pack_latents(reference_latents, bs, num_channels_latents, h, w)
            assert reference_latents.shape == latents.shape, (reference_latents.shape, latents.shape)
            if not extra:
                 img_seq_len = torch.tensor(x_t.shape[1], device=x_t.device).repeat((bs,))
                 extra = (img_seq_len,)
            x_t = torch.cat([x_t, reference_latents], dim=1)
            img_shapes.append((1, h // 2, w // 2))

        img_shapes = torch.tensor([img_shapes], dtype=torch.int32, device=device).repeat((bs, 1, 1))
        txt_seq_lens = torch.tensor([max_text_len], dtype=torch.int32, device=device).repeat((bs,))
        img_attention_mask = torch.ones((bs, x_t.shape[1]), dtype=torch.bool, device=device)
        attention_mask = torch.cat([prompt_embeds_mask, img_attention_mask], dim=1)
        # Make broadcastable with attention weights, which are [bs, num_heads, query_len, key_value_len]
        attention_mask = attention_mask.view(bs, 1, 1, -1)
        assert attention_mask.dtype == torch.bool

        return (
            (x_t, prompt_embeds, attention_mask, t, img_shapes, txt_seq_lens) + extra,
            (target, mask),
        )