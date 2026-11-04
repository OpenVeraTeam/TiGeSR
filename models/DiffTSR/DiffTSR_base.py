import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as transforms
from diffusers import DDPMScheduler

from models.DiffTSR.IDM.utils.util import instantiate_from_config, default
from models.DiffTSR.IDM.modules.diffusionmodules.util import (
    extract_into_tensor,
)


from seg_loss.focal_loss import BinaryFocalLoss
from seg_loss.dice_loss import DiceLoss


def sigmoid_focal_loss(inputs, targets, num_masks, alpha=0.25, gamma=2):
    """Original stage-1 glyph loss from the training repository."""
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_masks


class Baseline_pipline(nn.Module):
    def __init__(
        self,
        max_length=24,
        scheduler_type="PNDM",
        num_classes=6736,
        transformer_dim=768,
        scale_factor=0.18215,
        IDM_Unet_config=None,
        transformer_encoder_config=None,
        VAE_model_config=None,
        Text_Prediction_config=None,
        scheduler_path="weights/DiffTSR/PNDMscheduler",
    ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.max_length = max_length
        self.num_classes = num_classes
        self.transformer_dim = transformer_dim
        self.scale_factor = scale_factor
        self.num_inference_steps = 20
        self.IDM_Unet = instantiate_from_config(IDM_Unet_config)
        self.transformer_encoder = instantiate_from_config(transformer_encoder_config)
        self.VAE_model = instantiate_from_config(VAE_model_config)
        self.Text_Prediction = instantiate_from_config(Text_Prediction_config)
        self.scheduler = DDPMScheduler.from_pretrained(scheduler_path)
        # self.scheduler.set_timesteps(num_inference_steps=self.inference_timesteps)

    def to_device(self, device=None):
        if device is not None:
            self.device = device
        self.IDM_Unet.to(self.device)
        self.transformer_encoder.to(self.device)
        self.VAE_model.to(self.device)
        self.Text_Prediction.to(self.device)

    def set_train(self):
        self.VAE_model.eval()
        self.VAE_model.requires_grad_(False)
        self.transformer_encoder.eval()
        self.transformer_encoder.requires_grad_(False)
        self.Text_Prediction.eval()
        self.Text_Prediction.requires_grad_(False)
        self.IDM_Unet.train()

    def set_eval(self):
        self.VAE_model.eval()
        self.VAE_model.requires_grad_(False)
        self.transformer_encoder.eval()
        self.transformer_encoder.requires_grad_(False)
        self.Text_Prediction.eval()
        self.Text_Prediction.requires_grad_(False)
        self.IDM_Unet.eval()
        self.IDM_Unet.requires_grad_(False)

    def load_model(
        self,
        path="ckpt/DiffTSR_new.ckpt",
        IDM_path="ckpt/IDM_Unet.ckpt",
        transformer_path="ckpt/Bert_only.ckpt",
        is_shape_diff=False,
    ):
        state_dict = torch.load(path, map_location="cpu")
        ret = self.IDM_Unet.load_state_dict(
            torch.load(IDM_path)["state_dict_dm"], strict=True
        )  # ["IDM_Unet"]
        print("IDM_Unet:", ret)
        ret = self.transformer_encoder.load_state_dict(
            torch.load(transformer_path), strict=True
        )
        print("transformer_encoder:", ret)
        ret = self.VAE_model.load_state_dict(state_dict["VAE_model"], strict=True)
        print("VAE_model:", ret)
        del state_dict
        torch.cuda.empty_cache()
        # print(f"VAE_model loaded from {path}")

    def IDM_q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_into_tensor(self.IDM_sqrt_alphas_cumprod, t, x_start.shape)
            * x_start
            + extract_into_tensor(
                self.IDM_sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )
            * noise
        )

    def Baseline_sample(self, input_image, text, disable_pbar=False):
        if not torch.is_tensor(input_image):
            lq_image = np.asarray(input_image)
            lq_image = lq_image / 255.0
            lq_image = np.ascontiguousarray(lq_image)
            lq_image = transforms.ToTensor()(lq_image).float()
        else:
            lq_image = input_image
        if lq_image.ndim != 4:
            lq_image = lq_image.unsqueeze(0)
        lq_image = lq_image.to(self.device)

        bsz = lq_image.shape[0]
        c_t = self.Text_Prediction.get_padding_label_from_text(text).to(self.device)

        z_LR = self.VAE_model.encode(lq_image).sample() * self.scale_factor
        b = z_LR.shape[0]
        noise = torch.randn_like(z_LR, device=z_LR.device)

        t = torch.full((b,), 999, device=self.device, dtype=torch.long)
        z_t = self.scheduler.add_noise(original_samples=z_LR, timesteps=t, noise=noise)
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        time_range = self.scheduler.timesteps
        total_steps = time_range.shape[0]
        if disable_pbar:
            iterator = time_range
        else:
            iterator = tqdm(time_range, desc="DiffTSR Sampling", total=total_steps)

        Ccond_t = self.transformer_encoder(c_t.unsqueeze(0).long()).to(self.device)
        for i, step in enumerate(iterator):
            ts = torch.full((b,), step, device=self.device, dtype=torch.long)
            e_t = self.IDM_Unet(
                x=torch.cat([z_t, z_LR], dim=1), timesteps=ts, context=Ccond_t
            )
            z_t = self.scheduler.step(e_t, ts, z_t, return_dict=False)[0]

        img = 1.0 / self.scale_factor * z_t
        img = self.VAE_model.decode(img)
        img = img.detach().cpu()
        img = torch.clamp(img, 0.0, 1.0) * 255
        img = img.numpy().astype(np.uint8)[0]
        img = np.transpose(img, (1, 2, 0))

        return img

    def forward(self, lq=None, hq=None, text=None):
        z_LR = self.VAE_model.encode(lq).sample() * self.scale_factor
        z_HQ = self.VAE_model.encode(hq).sample() * self.scale_factor
        cond_t = []
        for i in range(len(text)):
            c_t = self.Text_Prediction.get_padding_label_from_text(text[i]).to(
                self.device
            )
            cond_t.append(c_t)
        cond_t = torch.stack(cond_t, dim=0)
        cond_t = self.transformer_encoder(cond_t.long()).to(self.device)

        noise = torch.randn_like(z_HQ, device=self.device)
        # timesteps = torch.randint(0, 999, (z_HQ.shape[0],), device=self.device).long()
        timesteps = (
            torch.randint(0, 999, (1,), device=self.device).expand(z_HQ.shape[0]).long()
        )
        z_t = self.scheduler.add_noise(z_HQ, noise, timesteps)
        e_t = self.IDM_Unet(
            x=torch.cat([z_t, z_LR], dim=1), timesteps=timesteps, context=cond_t
        )
        loss = F.mse_loss(e_t.float(), noise.float(), reduction="mean")

        return loss


class MaskGen_wSegLoss(Baseline_pipline):
    def forward(self, lq=None, hq=None, mask=None, text=None):
        z_LR = self.VAE_model.encode(lq).sample() * self.scale_factor
        z_mask = self.VAE_model.encode(mask).sample() * self.scale_factor
        self.scheduler.timesteps = torch.from_numpy(np.arange(0, 1000)[::-1].copy()).to(
            z_LR.device
        )

        cond_t = []
        for i in range(len(text)):
            c_t = self.Text_Prediction.get_padding_label_from_text(text[i]).to(
                z_LR.device
            )
            cond_t.append(c_t)
        cond_t = torch.stack(cond_t, dim=0)
        cond_t = self.transformer_encoder(cond_t.long()).to(z_LR.device)

        noise = torch.randn_like(z_mask, device=z_LR.device)
        timesteps = (
            torch.randint(0, 999, (1,), device=z_LR.device)
            .expand(z_mask.shape[0])
            .long()
        )
        z_t = self.scheduler.add_noise(z_mask, noise, timesteps)
        e_t = self.IDM_Unet(
            x=torch.cat([z_t, z_LR], dim=1), timesteps=timesteps, context=cond_t
        )
        pred_original_sample = self.scheduler.step(
            e_t, timesteps[0], z_t, return_dict=False
        )[1]
        x_0 = self.VAE_model.decode(pred_original_sample / self.scale_factor)
        eps_loss = F.mse_loss(e_t.float(), noise.float(), reduction="mean")
        focal_loss = BinaryFocalLoss()(x_0, mask)
        dice_loss = DiceLoss()(x_0, mask)
        # loss = eps_loss + focal_loss + dice_loss

        return eps_loss, focal_loss  # ,dice_loss


class DualBranchMaskGen(Baseline_pipline):

    def load_model(
        self,
        path="ckpt/DiffTSR_new.ckpt",
        IDM_path="ckpt/IDM_Unet.ckpt",
        transformer_path="ckpt/Bert_only.ckpt",
        is_training=True,
    ):
        state_dict = torch.load(path, map_location="cpu")
        idm_state = torch.load(IDM_path)["state_dict_dm"]
        curr_state = self.IDM_Unet.state_dict()
        for key in curr_state.keys():
            if ("out.2" in key or "input_blocks.0.0.weight" in key) and is_training:
                continue
            else:
                curr_state[key] = idm_state[key]
        ret = self.IDM_Unet.load_state_dict(curr_state, strict=True)  # ["IDM_Unet"]
        print("IDM_Unet:", ret)
        ret = self.transformer_encoder.load_state_dict(
            torch.load(transformer_path, map_location="cpu"), strict=True
        )
        print("transformer_encoder:", ret)
        ret = self.VAE_model.load_state_dict(state_dict["VAE_model"], strict=True)
        print("VAE_model:", ret)
        del state_dict
        torch.cuda.empty_cache()
        # print(f"VAE_model loaded from {path}")

    def forward(self, lq=None, hq=None, mask=None, text=None):
        z_LR = self.VAE_model.encode(lq).sample() * self.scale_factor
        z_HR = self.VAE_model.encode(hq).sample() * self.scale_factor
        z_mask = self.VAE_model.encode(mask).sample() * self.scale_factor
        self.scheduler.timesteps = torch.from_numpy(np.arange(0, 1000)[::-1].copy()).to(
            z_LR.device
        )

        cond_t = []
        for i in range(len(text)):
            c_t = self.Text_Prediction.get_padding_label_from_text(text[i]).to(
                z_LR.device
            )
            cond_t.append(c_t)
        cond_t = torch.stack(cond_t, dim=0).to(z_LR.device)
        cond_t = self.transformer_encoder(cond_t.long()).to(z_LR.device)
        z_Union = torch.cat([z_mask, z_HR], dim=1)
        noise = torch.randn_like(z_Union, device=z_LR.device)
        timesteps = (
            torch.randint(0, 999, (1,), device=z_LR.device)
            .expand(z_Union.shape[0])
            .long()
        )
        z_t = self.scheduler.add_noise(z_Union, noise, timesteps)
        e_t = self.IDM_Unet(
            x=torch.cat([z_t, z_LR], dim=1), timesteps=timesteps, context=cond_t
        )
        pred_original_sample = self.scheduler.step(
            e_t, timesteps[0], z_t, return_dict=False
        )[1]
        z0_mask, z0_SR = pred_original_sample.chunk(chunks=2, dim=1)
        x0_mask = self.VAE_model.decode(z0_mask / self.scale_factor)
        eps_loss = F.mse_loss(e_t.float(), noise.float(), reduction="mean")
        focal_loss = BinaryFocalLoss()(x0_mask, mask)
        # dice_loss = DiceLoss()(x0_mask, mask)

        return eps_loss, focal_loss, x0_mask

    def Baseline_sample(self, input_image, text, disable_pbar=False):
        if not torch.is_tensor(input_image):
            lq_image = np.asarray(input_image)
            lq_image = lq_image / 255.0
            lq_image = np.ascontiguousarray(lq_image)
            lq_image = transforms.ToTensor()(lq_image).float()
        else:
            lq_image = input_image
        if lq_image.ndim != 4:
            lq_image = lq_image.unsqueeze(0)
        lq_image = lq_image.to(self.device)

        c_t = self.Text_Prediction.get_padding_label_from_text(text).to(self.device)

        z_LR = self.VAE_model.encode(lq_image).sample() * self.scale_factor
        b = z_LR.shape[0]
        z_Union = torch.cat([torch.zeros_like(z_LR), z_LR], dim=1)
        noise = torch.randn_like(z_Union, device=z_Union.device)

        t = torch.full((b,), 999, device=self.device, dtype=torch.long)
        z_t = self.scheduler.add_noise(
            original_samples=z_Union, timesteps=t, noise=noise
        )
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        time_range = self.scheduler.timesteps
        total_steps = time_range.shape[0]
        if disable_pbar:
            iterator = time_range
        else:
            iterator = tqdm(time_range, desc="DiffTSR Sampling", total=total_steps)

        Ccond_t = self.transformer_encoder(c_t.unsqueeze(0)).to(self.device)
        for i, step in enumerate(iterator):
            ts = torch.full((b,), step, device=self.device, dtype=torch.long)
            e_t = self.IDM_Unet(
                x=torch.cat([z_t, z_LR], dim=1), timesteps=ts, context=Ccond_t
            )
            z_t = self.scheduler.step(e_t, ts, z_t, return_dict=False)[0]

        img = 1.0 / self.scale_factor * z_t
        mask, img = img.chunk(chunks=2, dim=1)
        img = self.VAE_model.decode(img)

        mask = self.VAE_model.decode(mask)
        return mask, img

class UNet2OutPipeline(Baseline_pipline):
    def __init__(self, max_length=24, scheduler_type="PNDM", num_classes=6736, transformer_dim=768, scale_factor=0.18215,IDM_Unet_config=None, transformer_encoder_config=None, VAE_model_config=None, Text_Prediction_config=None, hi_sam_ckpt="weights/BTS-test-test_best.pth", hi_sam_model_type="vit_h", scheduler_path="weights/DiffTSR/PNDMscheduler"):
        super().__init__(max_length, scheduler_type, num_classes, transformer_dim, scale_factor, IDM_Unet_config, transformer_encoder_config, VAE_model_config, Text_Prediction_config, scheduler_path=scheduler_path)
        self.hi_sam_ckpt = hi_sam_ckpt
        self.hi_sam_model_type = hi_sam_model_type
        self.predictor = None

    def _get_predictor(self):
        # Training uses supplied glyph masks. Load Hi-SAM only when sampling
        # requires an initial mask, rather than on every model construction.
        if self.predictor is not None:
            return self.predictor
        from pathlib import Path

        if not Path(self.hi_sam_ckpt).is_file():
            raise FileNotFoundError(
                f"Missing Hi-SAM checkpoint: {self.hi_sam_ckpt}. "
                "See the README pretrained-asset instructions."
            )
        try:
            from hi_sam.modeling.predictor import SamPredictor
            from hi_sam.modeling.build import model_registry
        except ImportError as exc:
            raise ImportError(
                "The bundled Hi-SAM runtime could not be imported. "
                "Install requirements.txt and run from the repository root."
            ) from exc
        hi_sam_args = {
            'model_type': self.hi_sam_model_type,
            "checkpoint": self.hi_sam_ckpt,
            "attn_layers": 1,
            "prompt_len": 12,
            "hier_det": False,
        }
        hi_sam_args = argparse.Namespace(**hi_sam_args)
        hisam = model_registry[self.hi_sam_model_type](hi_sam_args)
        hisam.eval()
        hisam.to(self.device)
        self.predictor = SamPredictor(hisam)
        return self.predictor

    def load_model(
        self,
        path="ckpt/DiffTSR_new.ckpt",
        IDM_path="ckpt/IDM_Unet.ckpt",
        transformer_path="ckpt/Bert_only.ckpt",
        is_shape_diff=True,
    ):
        state_dict = torch.load(path, map_location="cpu")
        try:
            idm_state = torch.load(IDM_path, map_location="cpu")["state_dict_dm"]
        except:
            idm_state = torch.load(IDM_path, map_location="cpu")["IDM_Unet"]
        curr_state = self.IDM_Unet.state_dict()
        for key in curr_state.keys():
            if ("out_mask." in key or "out.2" in key) and is_shape_diff:
                continue
            else:
                curr_state[key] = idm_state[key]
        ret = self.IDM_Unet.load_state_dict(curr_state, strict=True)  # ["IDM_Unet"]
        print("IDM_Unet:", ret)
        ret = self.transformer_encoder.load_state_dict(
            torch.load(transformer_path, map_location="cpu"), strict=True
        )
        print("transformer_encoder:", ret)
        ret = self.VAE_model.load_state_dict(state_dict["VAE_model"], strict=True)
        print("VAE_model:", ret)
        del state_dict
        torch.cuda.empty_cache()
        # print(f"VAE_model loaded from {path}")

    def forward(self, lq=None, hq=None, mask=None, text=None):
        z_LR = self.VAE_model.encode(lq).sample() * self.scale_factor
        z_HR = self.VAE_model.encode(hq).sample() * self.scale_factor
        z_mask = self.VAE_model.encode(mask).sample() * self.scale_factor
        self.scheduler.timesteps = torch.from_numpy(np.arange(0, 1000)[::-1].copy()).to(
            z_LR.device
        )

        cond_t = []
        for i in range(len(text)):
            c_t = self.Text_Prediction.get_padding_label_from_text(text[i]).to(
                z_LR.device
            )
            cond_t.append(c_t)
        cond_t = torch.stack(cond_t, dim=0).to(z_LR.device)
        cond_t = self.transformer_encoder(cond_t.long()).to(self.device)
        # z_Union = torch.cat([z_mask, z_HR], dim=1)
        noise_mask = torch.randn_like(z_mask, device=self.device)
        noise = torch.randn_like(z_HR, device=self.device)
        timesteps = (
            torch.randint(0, 999, (1,), device=self.device)
            .expand(z_mask.shape[0])
            .long()
        )
        z_mask_t = self.scheduler.add_noise(z_mask, noise_mask, timesteps)
        z_t = self.scheduler.add_noise(z_HR, noise, timesteps)
        e_t, e_mask_t = self.IDM_Unet(
            x=torch.cat([z_t, z_mask_t, z_LR], dim=1),
            timesteps=timesteps,
            context=cond_t,
        )
        z0_mask = self.scheduler.step(
            e_mask_t, timesteps[0], z_mask_t, return_dict=False
        )[1]
        x0_mask = self.VAE_model.decode(z0_mask / self.scale_factor)
        eps_loss = F.mse_loss(e_t, noise, reduction="mean") + F.mse_loss(
            e_mask_t, noise_mask, reduction="mean"
        )
        focal_loss = BinaryFocalLoss()(x0_mask, mask)
        # dice_loss = DiceLoss()(x0_mask, mask)

        return eps_loss, focal_loss  # , dice_loss

    def Baseline_sample(self, input_image, text, disable_pbar=False):
        if not torch.is_tensor(input_image):
            lq_image = np.asarray(input_image)
            lq_image = lq_image / 255.0
            lq_image = np.ascontiguousarray(lq_image)
            lq_image = transforms.ToTensor()(lq_image).float()
        else:
            lq_image = input_image
        if lq_image.ndim != 4:
            lq_image = lq_image.unsqueeze(0)

        predictor = self._get_predictor()
        predictor.set_image(lq_image[0].permute(1, 2, 0).cpu().numpy())
        _, sub_mask, _, _ = predictor.predict_torch(multimask_output=False)
        lq_image = lq_image.to(self.device)
        c_t = self.Text_Prediction.get_padding_label_from_text(text).to(self.device)
        z_LR = self.VAE_model.encode(lq_image).sample() * self.scale_factor
        z_mask = self.VAE_model.encode(torch.cat([sub_mask.float()]*3,dim=1)).sample() * self.scale_factor
        b = z_LR.shape[0]
        # z_Union = torch.cat([z_LR]*2,dim=1)

        t = torch.full((b,), 999, device=self.device, dtype=torch.long)
        noise = torch.randn_like(z_LR, device=z_LR.device)
        z_t = self.scheduler.add_noise(original_samples=z_LR, timesteps=t, noise=noise)
        # z_t_mask = torch.randn_like(z_LR, device=z_LR.device)
        z_t_mask = self.scheduler.add_noise(
            original_samples=z_mask, timesteps=t, noise=noise
        )
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        time_range = self.scheduler.timesteps
        total_steps = time_range.shape[0]
        if disable_pbar:
            iterator = time_range
        else:
            iterator = tqdm(time_range, desc="DiffTSR Sampling", total=total_steps)

        Ccond_t = self.transformer_encoder(c_t.long().unsqueeze(0)).to(self.device)
        for i, step in enumerate(iterator):
            ts = torch.full((b,), step, device=self.device, dtype=torch.long)
            e_t, e_t_mask = self.IDM_Unet(
                x=torch.cat([z_t, z_t_mask, z_LR], dim=1), timesteps=ts, context=Ccond_t
            )
            z_t = self.scheduler.step(e_t, ts, z_t, return_dict=False)[0]
            z_t_mask = self.scheduler.step(e_t_mask, ts, z_t_mask, return_dict=False)[0]

        img = 1.0 / self.scale_factor * z_t
        mask = 1.0 / self.scale_factor * z_t_mask
        img = self.VAE_model.decode(img)

        mask = self.VAE_model.decode(mask)
        return mask, img

    def process_long_image(self, input_image, text):
        """
        处理可能具有超长宽比的图像。
        如果长宽比在模型处理范围内，则直接调用模型。
        如果超过范围，则分块处理。
        """
        # 1. 准备输入图像和初步掩码
        # 这部分逻辑与你 Baseline_sample 开头的部分类似
        lq_image = self.prepare_image_tensor(input_image) # 将输入统一转为Tensor

        # 获取初步的掩码 sub_mask
        predictor = self._get_predictor()
        predictor.set_image(lq_image[0].permute(1, 2, 0).cpu().numpy())
        _, sub_mask, _, _ = predictor.predict_torch(multimask_output=False)

            # 2. 检查长宽比
        C, H, W = lq_image.shape[1:] # 获取 C, H, W
        aspect_ratio = max(H, W) / min(H, W)

        # 设定一个阈值，例如30
        MAX_ASPECT_RATIO = 30 

        if aspect_ratio <= MAX_ASPECT_RATIO:
            # 长宽比在可接受范围内，直接调用原始模型
            print("Aspect ratio is within limits. Processing directly.")
            return self.Baseline_sample(lq_image, text) # 注意这里传入 lq_image
        else:
            # 长宽比超限，需要分块处理
            print(f"Aspect ratio {aspect_ratio:.2f} exceeds limit. Processing in chunks.")
            return self.process_in_chunks(lq_image, sub_mask, text)
        
    def process_in_chunks(self, lq_image, sub_mask, text):
        """
        将图像和掩码分块，分别处理后拼接。
        """
        C, H, W = lq_image.shape[1:]

        # 确定切分方向（是水平切还是垂直切）
        is_horizontal_split = W > H 
        long_dim = W if is_horizontal_split else H

        # 设定目标长宽比，例如 8:1
        TARGET_ASPECT_RATIO = 8 
        short_dim = min(H, W)
        chunk_size = int(short_dim * TARGET_ASPECT_RATIO)

        # 为了避免切口处的瑕疵，让块之间有一些重叠
        overlap = chunk_size // 8 # 例如，12.5% 的重叠区域

        chunks = []
        start = 0
        while start < long_dim:
            end = min(start + chunk_size, long_dim)
            chunks.append((start, end))
            if end == long_dim:
                break
            start = end - overlap

        # 存储每个块的处理结果
        processed_masks = []
        processed_images = []

        for i, (start, end) in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")

            # 根据方向切分 lq_image 和 sub_mask
            if is_horizontal_split:
                chunk_lq = lq_image[:, :, :, start:end]
                chunk_sub_mask = sub_mask[:, :, :, start:end]
            else:
                chunk_lq = lq_image[:, :, start:end, :]
                chunk_sub_mask = sub_mask[:, :, start:end, :]

            # !!! 关键：将分块后的 sub_mask 传入 Baseline_sample
            # 你需要修改 Baseline_sample 函数，让它能接收预先计算好的 sub_mask
            # 我们假设修改后的函数叫 Baseline_sample_with_mask
            processed_mask, processed_img = self.Baseline_sample_with_mask(chunk_lq, chunk_sub_mask, text)

            processed_masks.append(processed_mask)
            processed_images.append(processed_img)

        # 3. 拼接结果
        # 这里需要一个更智能的拼接函数，处理重叠区域
        final_mask = self.stitch_results(processed_masks, (C, H, W), chunks, overlap, is_horizontal_split)
        final_img = self.stitch_results(processed_images, (C, H, W), chunks, overlap, is_horizontal_split)

        return final_mask, final_img
    
    # 原始函数: Baseline_sample(self, input_image, text, ...)
    # 修改后:
    def Baseline_sample_with_mask(self, lq_image, sub_mask, text, disable_pbar=False):
        # lq_image 已经是 tensor 了，sub_mask 也已传入
        # 所以可以跳过函数开头的数据准备和 predictor.predict_torch() 部分

        if lq_image.ndim != 4:
            lq_image = lq_image.unsqueeze(0)

        # 直接从这里开始
        lq_image = lq_image.to(self.device)
        # 注意：sub_mask 也需要移动到正确的设备
        sub_mask = sub_mask.to(self.device)

        c_t = self.Text_Prediction.get_padding_label_from_text(text).to(self.device)
        z_LR = self.VAE_model.encode(lq_image).sample() * self.scale_factor
        # 使用传入的 sub_mask
        z_mask = self.VAE_model.encode(torch.cat([sub_mask.float()]*3,dim=1)).sample() * self.scale_factor

        b = z_LR.shape[0]
        # z_Union = torch.cat([z_LR]*2,dim=1)

        t = torch.full((b,), 999, device=self.device, dtype=torch.long)
        noise = torch.randn_like(z_LR, device=z_LR.device)
        z_t = self.scheduler.add_noise(original_samples=z_LR, timesteps=t, noise=noise)
        # z_t_mask = torch.randn_like(z_LR, device=z_LR.device)
        z_t_mask = self.scheduler.add_noise(
            original_samples=z_mask, timesteps=t, noise=noise
        )
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        time_range = self.scheduler.timesteps
        total_steps = time_range.shape[0]
        if disable_pbar:
            iterator = time_range
        else:
            iterator = tqdm(time_range, desc="DiffTSR Sampling", total=total_steps)

        Ccond_t = self.transformer_encoder(c_t.long().unsqueeze(0)).to(self.device)
        for i, step in enumerate(iterator):
            ts = torch.full((b,), step, device=self.device, dtype=torch.long)
            e_t, e_t_mask = self.IDM_Unet(
                x=torch.cat([z_t, z_t_mask, z_LR], dim=1), timesteps=ts, context=Ccond_t
            )
            z_t = self.scheduler.step(e_t, ts, z_t, return_dict=False)[0]
            z_t_mask = self.scheduler.step(e_t_mask, ts, z_t_mask, return_dict=False)[0]

        img = 1.0 / self.scale_factor * z_t
        mask = 1.0 / self.scale_factor * z_t_mask
        img = self.VAE_model.decode(img)

        mask = self.VAE_model.decode(mask)
        return mask, img





class Unet2OutPrunedPipeline(UNet2OutPipeline):

    def load_model(
        self,
        path="ckpt/DiffTSR_new.ckpt",
        IDM_path="ckpt/IDM_Unet.ckpt",
        transformer_path="ckpt/Bert_only.ckpt",
        is_shape_diff=True,
    ):
        state_dict = torch.load(path, map_location="cpu")
        idm_state = torch.load(IDM_path, map_location="cpu")["state_dict_dm"]
        curr_state = self.IDM_Unet.state_dict()
        # print(curr_state.keys())
        for key in curr_state.keys():
            if (
                "out_mask." in key
                or "out.2" in key
                or "middle_block." in key
                or "output_blocks.0.0" in key
            ) and is_shape_diff:
                continue
            elif "output_blocks." in key and is_shape_diff:
                curr_state[key] = idm_state[
                    key.replace(
                        "output_blocks." + f"{int(key.split('.')[1])}",
                        "output_blocks." + f"{int(key.split('.')[1]) + 3}",
                    )
                ]
            else:
                curr_state[key] = idm_state[key]
        ret = self.IDM_Unet.load_state_dict(curr_state, strict=True)  # ["IDM_Unet"]
        print("IDM_Unet:", ret)
        ret = self.transformer_encoder.load_state_dict(
            torch.load(transformer_path, map_location="cpu"), strict=True
        )
        print("transformer_encoder:", ret)
        ret = self.VAE_model.load_state_dict(state_dict["VAE_model"], strict=True)
        print("VAE_model:", ret)
        del state_dict, idm_state
        torch.cuda.empty_cache()
        # print(f"VAE_model loaded from {path}")


class UNet2OutPipeline_wEncoder(UNet2OutPipeline):
    """Stage-1 training model from DiffTSR/train_mask_2OutwEncoder_bucket.py.

    The inference-only UNet2OutPipeline returns two losses. This training
    variant restores the original four-loss interface and trains the encoder.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from models.DiffTSR.sobel_edge_loss import DifferentiableEdgeDetector

        self.edge_det = DifferentiableEdgeDetector().to(self.device).eval()
        self.edge_det.requires_grad_(False)

    def load_model(
        self,
        path="ckpt/DiffTSR_new.ckpt",
        IDM_path="ckpt/IDM_Unet.ckpt",
        transformer_path="ckpt/Bert_only.ckpt",
        is_shape_diff=True,
        is_pruned=False,
    ):
        state_dict = torch.load(path, map_location="cpu")
        idm_checkpoint = torch.load(IDM_path, map_location="cpu")
        idm_state = idm_checkpoint.get("state_dict_dm", idm_checkpoint.get("IDM_Unet"))
        if idm_state is None:
            raise KeyError("Stage-1 checkpoint must contain state_dict_dm or IDM_Unet")
        curr_state = self.IDM_Unet.state_dict()
        for key in curr_state:
            if (
                "out_mask." in key or "out.2" in key or "input_blocks.0.0" in key
            ) and (is_shape_diff or is_pruned):
                continue
            curr_state[key] = idm_state[key]
        print("IDM_Unet:", self.IDM_Unet.load_state_dict(curr_state, strict=True))

        encoder_state = torch.load(transformer_path, map_location="cpu")
        # The original training code keeps the new positional/logit layers
        # when initializing an encoder with a different sequence shape.
        curr_encoder = self.transformer_encoder.state_dict()
        for key in curr_encoder:
            if key in encoder_state and curr_encoder[key].shape == encoder_state[key].shape:
                curr_encoder[key] = encoder_state[key]
            elif not is_shape_diff:
                raise ValueError(f"Encoder checkpoint shape/key mismatch: {key}")
            elif key not in {
                "transformer.pos_emb.emb.weight",
                "transformer.to_logits.weight",
                "transformer.to_logits.bias",
            }:
                raise ValueError(f"Unexpected encoder checkpoint shape/key mismatch: {key}")
        print("transformer_encoder:", self.transformer_encoder.load_state_dict(curr_encoder, strict=True))
        print("VAE_model:", self.VAE_model.load_state_dict(state_dict["VAE_model"], strict=True))
        torch.cuda.empty_cache()

    def set_train(self):
        self.VAE_model.eval()
        self.VAE_model.requires_grad_(False)
        self.Text_Prediction.eval()
        self.Text_Prediction.requires_grad_(False)
        self.IDM_Unet.train()
        self.transformer_encoder.train()

    def set_unet_train(self):
        self.set_train()
        self.IDM_Unet.set_unet_train()

    def forward(self, lq=None, hq=None, mask=None, text=None, is_ver=None):
        z_LR = self.VAE_model.encode(lq).sample() * self.scale_factor
        z_HR = self.VAE_model.encode(hq).sample() * self.scale_factor
        z_mask = self.VAE_model.encode(mask).sample() * self.scale_factor
        self.scheduler.timesteps = torch.from_numpy(np.arange(0, 1000)[::-1].copy()).to(z_LR.device)
        cond_t = torch.stack([
            self.Text_Prediction.get_padding_label_from_text(item).to(z_LR.device)
            for item in text
        ])
        cond_t = self.transformer_encoder(cond_t.long()).to(self.device)
        noise_mask = torch.randn_like(z_mask, device=self.device)
        noise = torch.randn_like(z_HR, device=self.device)
        timesteps = torch.randint(0, 999, (1,), device=self.device).expand(z_mask.shape[0]).long()
        z_mask_t = self.scheduler.add_noise(z_mask, noise_mask, timesteps)
        z_t = self.scheduler.add_noise(z_HR, noise, timesteps)
        e_t, e_mask_t = self.IDM_Unet(
            x=torch.cat([z_t, z_mask_t, z_LR], dim=1),
            timesteps=timesteps,
            context=cond_t,
        )
        z0_mask = self.scheduler.step(e_mask_t, timesteps[0], z_mask_t, return_dict=False)[1]
        x0_mask = self.VAE_model.decode(z0_mask / self.scale_factor)
        eps_loss = F.mse_loss(e_t, noise) + F.mse_loss(e_mask_t, noise_mask)
        # Preserve the executed original experiment: the next-line unary
        # '+ dice_loss(...)' in the source did not contribute to mask_loss.
        mask_loss = 20 * sigmoid_focal_loss(x0_mask.flatten(1), mask.flatten(1), mask.shape[0])
        edge_loss = F.mse_loss(self.edge_det(x0_mask), self.edge_det(mask))
        l2_loss = F.mse_loss(x0_mask, mask)
        return eps_loss, mask_loss, l2_loss, edge_loss

    def Baseline_sample(self, *args, **kwargs):
        mask, image = super().Baseline_sample(*args, **kwargs)

        def as_image(tensor):
            tensor = tensor.detach().cpu().clamp(0.0, 1.0) * 255
            return tensor.numpy().astype(np.uint8)[0].transpose(1, 2, 0)

        return as_image(mask), as_image(image)
