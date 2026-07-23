import gc
import json
import math
import os
import sys
# 设置 NCCL 超时时间（单位：毫秒）
os.environ['NCCL_TIMEOUT'] = '1800000'  # 30分钟
os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
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
from utils.training_utils_hacked import RealCEDataset,RealCECropDataset, parse_args_paired_training,EarlyStopping

from accelerate.utils import DistributedDataParallelKwargs

from models.Skip_con import SRCtrlnetSkipConnectionPipeline
from utils.losses import DifferentiableEdgeDetector, get_edge_loss

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def print_in_main_process(accelerator, text):
    if accelerator.is_main_process:
        print(text)

def resize_to_max_size(img, max_size=512):
    _, _, h, w = img.shape
    if h > w:
        new_h, new_w = max_size, int(w * max_size / h)
    else:
        new_h, new_w = int(h * max_size / w), max_size
    return F.interpolate(img, (new_h, new_w), mode="bilinear")

def main(args):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=args.output_dir,
        kwargs_handlers=[kwargs]
    )
    # 初始化early stopping
    early_stopping = None
    if args.early_stopping:
        mode = 'min' if args.early_stopping_monitor in ['eval/lpips'] else 'max'
        early_stopping = EarlyStopping(
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            monitor=args.early_stopping_monitor,
            mode=mode
        )
        print_in_main_process(accelerator, f"Early stopping enabled: monitor={args.early_stopping_monitor}, patience={args.early_stopping_patience}")

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

    net_pix2pix = SRCtrlnetSkipConnectionPipeline(target_timestep=args.timestep,
                                    ckpt_dir='./weights/stable-diffusion-3.5-medium', 
                                    gradient_checkpointing=args.gradient_checkpointing, 
                                    num_ctrl_layers=args.ctrl_layers,lora_rank_vae=4)
    trained_steps = 0
    trained_Epochs = 0
    # Whether to continue training or not?
    if args.ckpt_path is not None:
        net_pix2pix.load_model(args.ckpt_path,has_vae=args.has_vae)
        if "Epoch" in args.ckpt_path:
            trained_Epochs = int(args.ckpt_path.split("_")[-1].split(".")[0]) + 1
            print_in_main_process(accelerator, f"=========== Continue training from {trained_Epochs} epoch ===========")
        else:
            trained_steps = int(args.ckpt_path.split("_")[-1].split(".")[0])
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
    for n, _p in net_pix2pix.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt = layers_to_opt + list(net_pix2pix.vae.decoder.skip_conv_1.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_2.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_3.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_4.parameters())

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                  eps=args.adam_epsilon, )
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                                 num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                 num_training_steps=args.max_train_steps * accelerator.num_processes,
                                 num_cycles=args.lr_num_cycles, power=args.lr_power, )

    dataset_train = RealCEDataset(dataset_folder=args.train_folders)

    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True,
                                           num_workers=args.dataloader_num_workers)
    dataset_val = RealCEDataset(dataset_folder=args.test_folder,use_hisam=False)
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
    # json_file_path = os.path.join(args.train_folders,'bbox.json')
    # with open(json_file_path,'r') as json_file:
    #     bbox_data = json.load(json_file)

    # turn off eff. attn for the discriminator
    print_in_main_process(accelerator, "=========== Start the training loop ===========")
    for epoch in tqdm(
        range(0, args.num_training_epochs),
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
                    logs["loss/total"] = loss.detach().item()
                    logs["loss/l2"] = loss_l2.detach().item()
                    logs["loss/lpips"] = loss_lpips.detach().item()
                    # logs["loss_edge"] = loss_edge.detach().item()
                    # logs["learning_rate"] = lr_scheduler.get_last_lr()[0]  # 也记录学习率
                    accelerator.log(logs, step=global_step)

                    progress_bar.set_postfix(**logs)

                    # checkpoint the model
                    if global_step != 0 and (
                        global_step % args.checkpointing_steps == 0
                        # or global_step % num_update_steps_per_epoch == 0
                    ):
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_pix2pix).save_model(outf)

                    # compute validation set FID, L2, CLIP-SIM
                    if global_step == 1 or (global_step != 0 and (
                        global_step % args.eval_freq == 0
                        or global_step % num_update_steps_per_epoch == 0
                    )):
                        print("Running Eval...")
                        total_psnr = 0
                        total_ssim = 0
                        total_lpips = 0
                        eval_samples = 0
                        for step, batch_val in enumerate(tqdm(dl_val,desc="Eval")):
                            # if step >= args.num_samples_eval:
                            #     break
                            x_src = batch_val["input"].cuda()
                            x_mask = batch_val['mask'].cuda()
                            x_tgt = batch_val['gt'].cuda()
                            B, C, H, W = x_src.shape
                            x_src = F.interpolate(x_src, (H //8 *8, W //8 *8), mode="bilinear")
                            x_mask = F.interpolate(x_mask, (H //8 *8, W //8 *8), mode="bilinear")
                            x_tgt = F.interpolate(x_tgt,(H //8 *8, W //8 *8), mode="bilinear")
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                # forward pass
                                x_tgt_pred = accelerator.unwrap_model(net_pix2pix)(x_src, x_mask, width=W,height=H)
                                x_tgt_resized = resize_to_max_size(x_tgt, 512)
                                x_tgt_pred_resized = resize_to_max_size(x_tgt_pred, 512)
                                batch_lpips = net_lpips(x_tgt_pred,x_tgt).mean()
                                pred_np = x_tgt_pred_resized[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)
                                gt_np = x_tgt_resized[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)
                                # 计算PSNR
                                batch_psnr = psnr(gt_np, pred_np, data_range=1.0)
                                
                                # 计算SSIM
                                batch_ssim = ssim(
                                    gt_np, pred_np, 
                                    multichannel=True, 
                                    data_range=1.0, 
                                    channel_axis=2
                                )
                                total_psnr += batch_psnr
                                total_ssim += batch_ssim
                                total_lpips += batch_lpips
                                eval_samples += 1
                                
                                gc.collect()
                                torch.cuda.empty_cache()
                        # 计算平均值
                        avg_psnr = total_psnr / eval_samples
                        avg_ssim = total_ssim / eval_samples
                        avg_lpips = total_lpips / eval_samples

                        # 使用accelerator记录日志
                        eval_logs = {
                            "eval/psnr": avg_psnr,
                            "eval/ssim": avg_ssim,
                            "eval/lpips": avg_lpips,
                        }
                        accelerator.log(eval_logs, step=global_step)
                        gc.collect()
                        torch.cuda.empty_cache()
                        print(f"Eval Results - Step {global_step}: PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}")
                        if early_stopping is not None:
                            should_stop = False
                            if accelerator.is_main_process:
                                monitor_value = eval_logs[args.early_stopping_monitor]
                                if early_stopping(monitor_value):
                                    should_stop = True
                                    print_in_main_process(accelerator, f"Early stopping triggered at step {global_step}")
                                    print_in_main_process(accelerator, f"Best {args.early_stopping_monitor}: {early_stopping.best_score:.6f}")
                                    
                                    # 保存停止标志文件
                                    stop_file = os.path.join(args.output_dir, "early_stop_signal")
                                    with open(stop_file, 'w') as f:
                                        f.write(str(global_step))
                                else:
                                    print_in_main_process(accelerator, f"Early stopping: {early_stopping.counter}/{early_stopping.patience} - Best {args.early_stopping_monitor}: {early_stopping.best_score:.6f}")
                        
                            # 等待主进程完成文件操作
                            accelerator.wait_for_everyone()
                            
                            # 所有进程检查停止文件
                            stop_file = os.path.join(args.output_dir, "early_stop_signal")
                            if os.path.exists(stop_file):
                                if accelerator.is_main_process:
                                    # 保存最终模型
                                    final_outf = os.path.join(args.output_dir, "checkpoints", f"model_early_stop_{global_step}.pkl")
                                    accelerator.unwrap_model(net_pix2pix).save_model(final_outf)
                                    print_in_main_process(accelerator, f"Final model saved to {final_outf}")
                                    
                                    # 清理停止文件
                                    os.remove(stop_file)
                        
                                # 等待所有进程
                                accelerator.wait_for_everyone()
                                
                                # 结束训练
                                accelerator.end_training()
                                return

                                            # --- EarlyStopping 逻辑修复结束 ---
                        
                        gc.collect()
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
