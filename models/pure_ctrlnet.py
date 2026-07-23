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

class FlowMatchEulerDiscreteScheduler1Step(FlowMatchEulerDiscreteScheduler):
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]

        prev_sample = sample + (sigma_next - sigma) * model_output

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        # self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

def make_1step_sched():
    noise_scheduler_1step = DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step

class SRCtrlnetPipeline(nn.Module):

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
    ):
        super().__init__()
        self.execution_device = torch.device("cuda")
        # 找到最接近150的时间步
        self.target_timestep = target_timestep
        if use_fp16:
            self.transformer = SD3Transformer2DModel.from_pretrained(
                ckpt_dir, subfolder="transformer", torch_dtype=torch.float16
            )
        else:
            self.transformer = SD3Transformer2DModel.from_pretrained(ckpt_dir, subfolder="transformer")
        if vae_path is None:
            if use_fp16:
                self.vae = AutoencoderKL.from_pretrained(
                    ckpt_dir, subfolder="vae", torch_dtype=torch.float16
                )
            else:
                self.vae = AutoencoderKL.from_pretrained(ckpt_dir, subfolder="vae")
        else:
            if use_fp16:
                self.vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
            else:
                self.vae = AutoencoderKL.from_pretrained(vae_path)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(ckpt_dir, subfolder="scheduler")
        # self.scheduler.set_timesteps(1)
        t_diff = torch.abs(self.scheduler.timesteps-self.target_timestep)
        self.closest_idx = torch.argmin(t_diff)
        if ctrlnet is None:
            self.controlnet = SD3ControlNetModel.from_transformer(self.transformer, num_layers=num_ctrl_layers,num_extra_conditioning_channels=num_extra_conditioning_channels)
            if use_fp16:
                self.controlnet = self.controlnet.to(torch.float16)
        else:
            self.controlnet = ctrlnet
        if gradient_checkpointing:
            self.controlnet.enable_gradient_checkpointing()
            self.transformer.enable_gradient_checkpointing()
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        saved_prompt_tokens = torch.load(f"{ckpt_dir}/saved_prompt_tokens_null_nocfg.pt", map_location="cpu")
        self.prompt_embeds = saved_prompt_tokens["prompt_embeds"].float()
        self.pooled_prompt_embeds = saved_prompt_tokens["pooled_prompt_embeds"].float()
        if use_fp16:
            self.prompt_embeds = self.prompt_embeds.to(dtype=torch.float16)
            self.pooled_prompt_embeds = self.pooled_prompt_embeds.to(dtype=torch.float16)

    def to_device(self,device=None):
        if device is not None:
            self.execution_device = device
        # self.scheduler.to(self.execution_device)
        self.transformer.to(self.execution_device)
        self.vae.to(self.execution_device)
        self.controlnet.to(self.execution_device)
        self.prompt_embeds = self.prompt_embeds.to(self.execution_device)
        self.pooled_prompt_embeds = self.pooled_prompt_embeds.to(self.execution_device)

    def set_eval(self):
        self.transformer.eval()
        self.transformer.requires_grad_(False)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.controlnet.eval()

    def set_train(self):
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.transformer.eval()
        self.transformer.requires_grad_(False)
        self.controlnet.train()
        # self.controlnet.requires_grad_(True)

    def prepare_latents(self, image, batch_size, num_images_per_prompt, dtype, device, generator=None):
        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt
        if image.shape[1] == self.vae.config.latent_channels:
            init_latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    retrieve_latents(self.vae.encode(image[i: i + 1]), generator=generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(self.vae.encode(image), generator=generator)
            try:
                init_latents = (init_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            except:
                init_latents = init_latents * self.vae.config.scaling_factor

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        latents = init_latents.to(device=device, dtype=dtype)

        return latents

    def prepare_image(self, image, width, height, batch_size, num_images_per_prompt, device, dtype):
        if not isinstance(image, torch.Tensor):
            image = self.image_processor.preprocess(image, height=height, width=width)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        return image

    """
    执行前向传播以生成超分辨率图像
    
    Args:
     lr_image (torch.Tensor): 输入的低分辨率图像
     text_mask (torch.Tensor): 文本掩码
     height (int, 可选): 输出图像的高度，默认为768
     width (int, 可选): 输出图像的宽度，默认为768
     num_images_per_prompt (int, 可选): 每个提示生成的图像数量，默认为1
     cond_scale (float, 可选): 条件缩放因子，默认为1.0
    
    Returns:
     torch.Tensor: 生成的超分辨率图像
    """
    def forward(self, lr_image, text_mask, height=512,width=512,num_images_per_prompt=1,cond_scale=1.0):
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
        latents = self.prepare_latents(lr_image, batch_size, num_images_per_prompt, dtype, device)
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
        sr_image = self.vae.decode(latents, return_dict=False)[0]
        sr_image = self.image_processor.postprocess(sr_image, output_type="pt")

        return sr_image

    def get_sigmas(self,timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.scheduler.sigmas.to(device=self.execution_device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(device=self.execution_device, dtype=dtype)
        timesteps = timesteps.to(device=self.execution_device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def save_model(self, outf):
        self.controlnet.eval()
        sd = {}
        sd["controlnet"] = self.controlnet.state_dict()
        torch.save(sd, outf)

    def load_model(self, state_dict_path):
        state_dict = torch.load(state_dict_path, map_location="cpu")
        ret = self.controlnet.load_state_dict(state_dict["controlnet"])
        print("controlnet", ret)
        