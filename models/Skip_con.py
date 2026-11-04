from typing import Any, Dict, Optional, Union,Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import SD3ControlNetModel, AutoencoderKL, DDPMScheduler, SD3Transformer2DModel, Transformer2DModel,FlowMatchEulerDiscreteScheduler,ControlNetModel,UNet2DConditionModel
from diffusers import DiffusionPipeline
from diffusers.loaders import FromSingleFileMixin, SD3IPAdapterMixin, SD3LoraLoaderMixin
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.controlnet import zero_module
from diffusers.models.embeddings import PatchEmbed, CombinedTimestepTextProjEmbeddings
from diffusers.models.attention import JointTransformerBlock
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img import retrieve_latents
from peft import LoraConfig

from models.pure_ctrlnet import SRCtrlnetPipeline

def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []
    # down
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    # middle
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample

def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        # up
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            # add skip
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample

class SRCtrlnetSkipConnectionPipeline(SRCtrlnetPipeline):
    def __init__(
        self,
        ctrlnet=None,
        target_timestep=150.0,
        ckpt_dir="./weights/stable-diffusion-3-medium-diffusers",
        gradient_checkpointing=True,
        num_ctrl_layers=6,
        num_extra_conditioning_channels=0,
        vae_path=None,
        use_fp16=False,
        lora_rank_vae=4,
        prompt_cache_path=None,
    ):
        super().__init__(ctrlnet,
            target_timestep,
            ckpt_dir,
            gradient_checkpointing,
            num_ctrl_layers,
            num_extra_conditioning_channels,
            vae_path,
            use_fp16,
            prompt_cache_path=prompt_cache_path,
        )
        self.lora_rank_vae = lora_rank_vae
        self.vae.encoder.forward = my_vae_encoder_fwd.__get__(self.vae.encoder, self.vae.encoder.__class__)
        self.vae.decoder.forward = my_vae_decoder_fwd.__get__(self.vae.decoder, self.vae.decoder.__class__)
        # add the skip connection convs
        self.vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.vae.decoder.ignore_skip = False
        self.vae.decoder.ignore_skip = False
        self.target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]
        vae_lora_config = LoraConfig(r=self.lora_rank_vae, init_lora_weights="gaussian", target_modules=self.target_modules_vae
        )

        self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        self.vae.decoder.gamma = 1
    
    
    def set_train(self):
        super().set_train()
        self.vae.train()
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

    def forward(self, lr_image, text_mask, height=512,width=512,num_images_per_prompt=1,cond_scale=1.0):
        # torch.set_printoptions(threshold=np.inf)
        # print(text_mask)
        # print("before prepare",lr_image.requires_grad)
        batch_size = lr_image.shape[0]
        indices = torch.tensor([self.closest_idx] * batch_size).long()
        timesteps = self.scheduler.timesteps[indices]
        sigmas = self.get_sigmas(timesteps,n_dim=lr_image.ndim,dtype=lr_image.dtype)
        dtype = self.prompt_embeds.dtype
        prompt_embeds = torch.cat([self.prompt_embeds]*batch_size)
        pooled_prompt_embeds = torch.cat([self.pooled_prompt_embeds]*batch_size)
        device = self.execution_device
        # print(lr_image.shape)
        lr_image = self.image_processor.preprocess(lr_image)
        control_image = self.prepare_image(
            image=text_mask,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=dtype,
        )
        timesteps = timesteps.to(self.execution_device)
        control_image = self.vae.encode(control_image).latent_dist.sample()
        try:
            control_image = (control_image - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        except:
            control_image = control_image * self.vae.config.scaling_factor
        # latents = self.prepare_latents(lr_image, batch_size, num_images_per_prompt, dtype, device)
        latents = (retrieve_latents(self.vae.encode(lr_image.to(device=device, dtype=dtype)))- self.vae.config.shift_factor) * self.vae.config.scaling_factor
        # self.timesteps = self.scheduler.timesteps.to(self.execution_device)
        control_block_samples = self.controlnet(
            hidden_states=latents,
            controlnet_cond=control_image,
            conditioning_scale=cond_scale,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            timestep=timesteps
        )[0]
        noise_pred = self.transformer(
            hidden_states=latents,  # b,16,64,64
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            block_controlnet_hidden_states=control_block_samples,
            joint_attention_kwargs=None,
            return_dict=False
        )[0]
        if noise_pred.shape != latents.shape:
            noise_pred = F.interpolate(
                noise_pred,
                size=(latents.shape[2], latents.shape[3]),
                mode='bilinear',
                align_corners=False
            )
        latents = noise_pred * (-sigmas) + latents
        try:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        except:
            latents = latents / self.vae.config.scaling_factor
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        # print(latents.shape)
        sr_image = (self.vae.decode(latents).sample).clamp(-1, 1)
        sr_image = self.image_processor.postprocess(sr_image, output_type="pt")

        return sr_image
    
    def save_model(self, outf):
        self.controlnet.eval()
        sd = {}
        sd["controlnet"] = self.controlnet.state_dict()
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        torch.save(sd, outf)

    def load_model(self, state_dict_path,has_vae=True):
        state_dict = torch.load(state_dict_path, map_location="cpu")
        ret = self.controlnet.load_state_dict(state_dict["controlnet"])
        print("controlnet", ret)
        if has_vae:
            _sd_vae = self.vae.state_dict()
            for k in state_dict["state_dict_vae"]:
                _sd_vae[k] = state_dict["state_dict_vae"][k]
            ret = self.vae.load_state_dict(_sd_vae)
            del _sd_vae,state_dict
            print("VAE lora and skip", ret)
