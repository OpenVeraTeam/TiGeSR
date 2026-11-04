import gc
import json
import math
import os
import re
import sys

import torch.nn.functional

import diffusers
import lpips
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.data
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from torch import nn
from tqdm.auto import tqdm
from torchvision.utils import save_image

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.img_util import combine_sr_patches_weighted, extract_overlapping_patches
import utils.saver as saver
from utils.training_utils_hacked import RealCEDataset,RealCECropDataset, parse_args_paired_training

from accelerate.utils import DistributedDataParallelKwargs

from models.pure_ctrlnet import SRCtrlnetPipeline
from utils.losses import DifferentiableEdgeDetector, get_edge_loss

def print_in_main_process(accelerator, text):
    if accelerator.is_main_process:
        print(text)

def main(args):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=args.output_dir,
        kwargs_handlers=[kwargs]
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

    net_pix2pix = SRCtrlnetPipeline(target_timestep=args.timestep,
                                    ckpt_dir=args.sd3_dir,
                                    prompt_cache_path=args.prompt_cache_path,
                                    gradient_checkpointing=args.gradient_checkpointing, 
                                    num_ctrl_layers=args.ctrl_layers)
    trained_steps = 0
    trained_Epochs = 0
    # Whether to continue training or not?
    if args.ckpt_path is not None:
        net_pix2pix.load_model(args.ckpt_path)
        checkpoint_name = os.path.basename(args.ckpt_path)
        epoch_match = re.search(r"Epoch_(\d+)", checkpoint_name)
        step_match = re.search(r"model[_-](\d+)", checkpoint_name)
        if epoch_match:
            trained_Epochs = int(epoch_match.group(1)) + 1
            print_in_main_process(accelerator, f"=========== Continue training from {trained_Epochs} epoch ===========")
        else:
            trained_steps = int(step_match.group(1)) if step_match else 0
            trained_Epochs = 0
            print_in_main_process(accelerator, f"=========== Continue training from {trained_steps} step ===========")

    net_pix2pix.to_device()
    net_pix2pix.set_train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    # net_edge = DifferentiableEdgeDetector()
    # net_edge.eval()
    # net_edge.requires_grad_(False)

    # make the optimizer
    layers_to_opt = []
    for n, _p in net_pix2pix.controlnet.named_parameters():
        if _p.requires_grad:
            layers_to_opt.append(_p)

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                  eps=args.adam_epsilon, )
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                                 num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                 num_training_steps=args.max_train_steps * accelerator.num_processes,
                                 num_cycles=args.lr_num_cycles, power=args.lr_power, )

    dataset_train = RealCECropDataset(dataset_folder=args.train_folders)

    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True,
                                           num_workers=args.dataloader_num_workers)
    dataset_val = RealCEDataset(dataset_folder=args.test_folder)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # Prepare everything with our `accelerator`.
    net_pix2pix, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_pix2pix, optimizer, dl_train, lr_scheduler
    )
    num_update_steps_per_epoch = math.ceil(len(dl_train) / args.gradient_accumulation_steps)

    print_in_main_process(accelerator, f"num_update_steps_per_epoch:{num_update_steps_per_epoch}")

    global_step = 0
    if trained_steps != 0 and "Epoch" not in args.ckpt_path:
        global_step = trained_steps
        trained_Epochs = trained_steps //  num_update_steps_per_epoch
        initial_steps = trained_steps % num_update_steps_per_epoch
        print_in_main_process(accelerator, f"=========== Continue training from {trained_Epochs} epoch {trained_steps} step ===========")
    else:
        initial_steps = 0

    net_lpips = accelerator.prepare(net_lpips)
    # net_edge = accelerator.prepare(net_edge)

    weight_dtype = torch.float32

    net_pix2pix.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    # net_edge.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        # print(tracker_config)
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Read the GT Text positions
    json_file_path = os.path.join(args.train_folders,'bbox.json')
    # with open(json_file_path,'r') as json_file:
    #     bbox_data = json.load(json_file)

    # turn off eff. attn for the discriminator
    print_in_main_process(accelerator, "=========== Start the training loop ===========")
    for epoch in tqdm(
        range(trained_Epochs, args.num_training_epochs),
        desc="Epochs",
        disable=not accelerator.is_local_main_process,
    ):
        progress_bar = tqdm(dl_train, initial=initial_steps , desc="Steps", disable=not accelerator.is_local_main_process)
        if initial_steps!=0:
            initial_steps=0
        for step, batch in enumerate(progress_bar):
            # print_in_main_process(accelerator, f"Epoch: {epoch}")
            l_acc = [net_pix2pix]
            with accelerator.accumulate(*l_acc):
                x_tgt = batch['gt']
                x_src = batch['input']
                x_mask = batch['mask']
                # The name here doesn't contain suffix
                name = batch["name"]
                # x_tgt = F.interpolate(x_tgt,(512,512))
                # x_src = F.interpolate(x_src,(512,512))
                # x_mask = F.interpolate(x_mask,(512,512))
                B, C, H, W = x_src.shape

                # print(x_src.shape)
                # print(x_tgt.shape[-2:])
                # forward pass
                x_tgt_pred = net_pix2pix(x_src, x_mask, height=H,width=W)
                # print(x_tgt_pred.requires_grad)
                # Reconstruction loss
                if x_tgt_pred.shape != x_tgt.shape:
                    x_tgt = F.interpolate(
                        x_tgt,
                        size=(x_tgt_pred.shape[2], x_tgt_pred.shape[3]),
                        mode="nearest",
                    )
                loss_l2 = F.mse_loss(x_tgt_pred, x_tgt, reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred, x_tgt).mean() * args.lambda_lpips
                # loss_edge = get_edge_loss(net_edge, x_tgt_pred, x_tgt, bbox_data, name) * args.lambda_edge
                loss = loss_l2 + loss_lpips #$+ loss_edge

                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    logs["loss"] = loss.detach().item()
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    # logs["loss_edge"] = loss_edge.detach().item()
                    # logs["learning_rate"] = lr_scheduler.get_last_lr()[0]  # 也记录学习率
                    accelerator.log(logs, step=global_step)

                    progress_bar.set_postfix(**logs)

                    # checkpoint the model
                    if global_step != 0 and (
                        global_step % args.checkpointing_steps == 0
                        or global_step % num_update_steps_per_epoch == 0
                    ):
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_pix2pix).save_model(outf)

                    # compute validation set FID, L2, CLIP-SIM
                    if global_step != 0 and (
                        global_step % args.eval_freq == 0
                        or global_step % num_update_steps_per_epoch == 0
                    ):
                        print("Running Eval...")
                        for step, batch_val in enumerate(tqdm(dl_val,desc="Eval")):
                            if step >= args.num_samples_eval:
                                break
                            x_src = batch_val["input"].cuda()
                            x_mask = batch_val['mask'].cuda()
                            x_tgt = batch_val['gt'].cuda()
                            B, C, H, W = x_src.shape
                            x_src = F.interpolate(x_src, (H //8 *8, W //8 *8), mode="bilinear")
                            x_mask = F.interpolate(x_mask, (H //8 *8, W //8 *8), mode="bilinear")
                            B, C, H, W = x_src.shape
                            x_src = F.interpolate(x_src, (H * 2, W * 2), mode="bilinear")
                            x_mask = F.interpolate(x_mask, (H * 2, W * 2), mode="bilinear")
                            subs = extract_overlapping_patches(
                                torch.cat([x_src, x_mask], dim=0),
                                patch_size=(H, W),
                                overlap=(H // 2, W // 2),
                            )
                            sub_is,sub_ms = subs.chunk(2,dim=0)
                            sub_SRs = []
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                # forward pass
                                for sub_i,sub_m in zip(sub_is,sub_ms):
                                    sub_SR = accelerator.unwrap_model(net_pix2pix)(
                                        sub_i.unsqueeze(0),
                                        sub_m.unsqueeze(0),
                                        width=W,
                                        height=H,
                                    )
                                    sub_SRs.append(sub_SR)
                                sr_patches = torch.cat(sub_SRs, dim=0)
                                x_tgt_pred = combine_sr_patches_weighted(
                                    sr_patches,
                                    out_shape=(B, C, H * 2, W * 2),
                                    patch_size=(H, W),
                                    overlap=(H // 2, W // 2),
                                )
                                # x_tgt_pred = accelerator.unwrap_model(net_pix2pix)(x_src, x_mask, width=W,height=H)
                                save_path = os.path.join(args.output_dir, "results", "%05d" % global_step)
                                os.makedirs(save_path,exist_ok=True)
                                x_src = F.interpolate(x_src, (H, W), mode="bilinear")
                                x_mask = F.interpolate(x_mask, (H, W), mode="bilinear")
                                # print("If x_src==prd??", x_src==x_tgt_pred)
                                save_image(x_src, os.path.join(save_path,f"{batch_val['name'][0]}_input.png"))
                                save_image(x_tgt_pred, os.path.join(save_path,f"{batch_val['name'][0]}_pred.png"))
                                save_image(x_tgt, os.path.join(save_path,f"{batch_val['name'][0]}_gt.png"))
                                save_image(x_mask, os.path.join(save_path,f"{batch_val['name'][0]}_mask_tgt.png"))

                                gc.collect()
                                torch.cuda.empty_cache()
                        print("Eval Done")
                        # accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
