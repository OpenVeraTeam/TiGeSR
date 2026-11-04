import gc
import math
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
import diffusers
import torch
import torch.nn.functional as F
import torch.utils.data
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed,ProjectConfiguration,DistributedDataParallelKwargs
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from models.DiffTSR.IDM.utils.util import instantiate_from_config
from data.CTR_dataset import CTRDatasetSynMask, CTRDatasetTextName
import argparse
from data.Bucket import AspectRatioBatchSampler,VariableSizeDataLoader,BucketDataset
from torchvision.utils import save_image
from pathlib import Path
from torch.utils.data import ConcatDataset

# Directly copied from DiffTSR config
scale_factor = 0.18215

def parse_args_training(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()
    # args for continue training
    parser.add_argument("--ckpt_path", default="weights/DiffTSR/DiffTSR_new.ckpt")
    parser.add_argument("--idm_ckpt", default="weights/uzst/stage1/model.pth")
    parser.add_argument("--encoder_ckpt", default="weights/uzst/stage1/encoder.pth")
    parser.add_argument("--config_yaml", default="configs/stage1_train.yaml")

    # args for the loss function
    parser.add_argument("--lambda_eps", type=float, default=1.0)
    parser.add_argument("--lambda_focal", type=float, default=1.0)
    parser.add_argument("--lambda_l2", type=float, default=1.0)
    parser.add_argument("--lambda_edge", type=float, default=1.0)

    # dataset options
    parser.add_argument("--train_folders", type=str)
    parser.add_argument("--test_folder", type=str)

    # validation eval args
    parser.add_argument("--eval_freq", default=20, type=int)
    parser.add_argument("--num_samples_eval", type=int, default=100, help="Number of samples to use for all evaluation")
    parser.add_argument("--num_samples_save", type=int, default=20, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="train_pix2pix_turbo",
                        help="The name of the wandb project to log to.")

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None, )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=20000, )
    parser.add_argument("--checkpointing_steps", type=int, default=500, )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--gradient_checkpointing", action="store_true", )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help=(
                            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                            ' "constant", "constant_with_warmup"]'
                        ),
                        )
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
                        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
                        )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0, )
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        help=(
                            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
                        ),
                        )
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true",
    #                     help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true", )
    parser.add_argument(
        "--is_shape_same",
        action="store_true",
    )
    parser.add_argument(
        "--unet_train",
        action="store_true",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    # args.train_folders = args.train_folders.split(',')
    # args.train_folders = args.train_folders.split(',')
    return args

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
        kwargs_handlers=[kwargs],
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

    # Model init
    tsr_config = OmegaConf.load(args.config_yaml)
    net_tsr = instantiate_from_config(tsr_config.model)
    weight_dtype = torch.float32

    # loading checkpoint
    is_shape_diff = True
    if args.is_shape_same:
        is_shape_diff = False
    # print(is_shape_diff)
    net_tsr.load_model(args.ckpt_path, args.idm_ckpt,transformer_path=args.encoder_ckpt, is_shape_diff=is_shape_diff)
    if args.unet_train:
        net_tsr.set_unet_train()
    else:
        net_tsr.set_train()
    net_tsr.to_device()

    # make the optimizer
    params_to_optimize = []
    for name, _param in net_tsr.IDM_Unet.named_parameters():
        _param.requires_grad = True
        params_to_optimize.append(_param)
    for name, _param in net_tsr.transformer_encoder.named_parameters():
        _param.requires_grad = True
        params_to_optimize.append(_param)

    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                  eps=args.adam_epsilon, )
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                                 num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                 num_training_steps=args.max_train_steps * accelerator.num_processes,
                                 num_cycles=args.lr_num_cycles, power=args.lr_power, )
    
    dataset_train = BucketDataset(dataset_folders=[substr.strip() for substr in args.train_folders.split(',')])
    sampler = AspectRatioBatchSampler(
        dataset_train.bucket_contents, 
        batch_size=args.train_batch_size,
        shuffle=True, drop_last=True,
        num_replicas=accelerator.num_processes,  # 总进程数
        rank=accelerator.process_index
    )  # 当前进程索引)
    dl_train = VariableSizeDataLoader(dataset_train, sampler,num_workers=args.dataloader_num_workers,prefetch_factor=8)
    # dataset_test = CTRDatasetTest(args.test_folder)
    # dl_test = torch.utils.data.DataLoader(dataset_test,batch_size=args.train_batch_size)
    # accelerator.wait_for_everyone()
    initial_steps = 0
    trained_Epochs = 0
    # Prepare everything with our `accelerator`

    net_tsr, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_tsr, optimizer, dl_train, lr_scheduler
    )
    num_update_steps_per_epoch = math.ceil(dataset_train.len_buckets / args.train_batch_size / args.gradient_accumulation_steps / accelerator.num_processes)

    net_tsr.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # turn off eff. attn for the discriminator
    print_in_main_process(accelerator, "=========== Start the training loop ===========")
    global_step = 0
    for epoch in tqdm(range(args.num_training_epochs), initial=trained_Epochs, desc="Epochs", disable=not accelerator.is_local_main_process):
        progress_bar = tqdm(range(num_update_steps_per_epoch), initial=initial_steps , desc="Steps", disable=not accelerator.is_local_main_process)
        for step, batch in enumerate(dl_train):
            l_acc = [net_tsr]
            with accelerator.accumulate(*l_acc):
                text = batch['text']
                lr_image = batch['lq'].to(
                    accelerator.device, non_blocking=True, dtype=weight_dtype
                )
                hr_image = batch['hq'].to(
                    accelerator.device, non_blocking=True, dtype=weight_dtype
                )
                mask_image = batch["mask"].to(
                    accelerator.device, non_blocking=True, dtype=weight_dtype
                )

                # forward pass
                eps_loss, mask_loss, l2_loss,edge_loss = net_tsr(
                    lr_image, hr_image, mask_image, text,
                )
                loss = eps_loss * args.lambda_eps + mask_loss * args.lambda_focal + l2_loss * args.lambda_l2 + args.lambda_edge * edge_loss
                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update()
                global_step += 1
                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    logs["loss/eps_loss"] = eps_loss.detach().item() * args.lambda_eps
                    logs["loss/mask_loss"] = mask_loss.detach().item() * args.lambda_focal
                    logs["loss/l2_loss"] = l2_loss.detach().item() * args.lambda_l2
                    logs["loss/edge_loss"] = edge_loss.detach().item() * args.lambda_edge

                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)
                    # checkpoint the model
                    if global_step != 0 and (global_step % num_update_steps_per_epoch==0 or global_step % args.checkpointing_steps == 0):
                        outf = os.path.join(args.output_dir, "checkpoints", f"model-{global_step}.pth")
                        torch.save({"state_dict_dm":accelerator.unwrap_model(net_tsr).eval().IDM_Unet.state_dict()},outf)
                        outf = os.path.join(args.output_dir, "checkpoints", f"encoder-{global_step}.pth")
                        torch.save(accelerator.unwrap_model(net_tsr).eval().transformer_encoder.state_dict(),outf.replace('.pkl','.pth'))

                    if (global_step % num_update_steps_per_epoch == 0) or (
                        global_step % args.eval_freq == 1
                    ):
                        input_lr_path = sorted(os.listdir(args.test_folder))
                        print("Running Eval...")
                        cnt = 0
                        for file_name in input_lr_path:
                            # print(file_name)
                            if cnt>=10:
                                break
                            cnt+=1
                            input_image = os.path.join(args.test_folder, file_name)
                            gt_mask = os.path.join(os.path.dirname(args.test_folder),"mask",file_name)
                            lq_image_pil = Image.open(input_image).convert('RGB')
                            # gt_mask_pil = Image.open(gt_mask).convert('RGB')
                            W,H = lq_image_pil.size
                            is_ver=False
                            if H/W>=2:
                                lq_image_pil = lq_image_pil.transpose(Image.ROTATE_90)
                                # gt_mask_pil = gt_mask_pil.transpose(Image.ROTATE_90)
                                is_ver=True
                                W,H = lq_image_pil.size
                            lq_image_pil = lq_image_pil.resize((round(128*W/H)//32*32, 128))
                            # gt_mask_pil = gt_mask_pil.resize((round(128*W/H)//32*32, 128))
                            if is_ver:
                                lq_image_pil = lq_image_pil.transpose(Image.ROTATE_270)
                                # gt_mask_pil = gt_mask_pil.transpose(Image.ROTATE_270)
                            # print(lq_image_pil.size)
                            
                            with torch.no_grad():
                                # forward pass
                                pred_mask, x_tgt_pred = accelerator.unwrap_model(net_tsr).Baseline_sample(lq_image_pil,Path(file_name).stem,disable_pbar=True)

                                pred_mask = Image.fromarray(pred_mask,"RGB")
                                x_tgt_pred = Image.fromarray(x_tgt_pred,"RGB")

                                save_dir = os.path.join(args.output_dir, "results", "%07d" % global_step)
                                os.makedirs(save_dir,exist_ok=True)
                                x_tgt_pred.save(
                                    os.path.join(
                                        save_dir, f"{Path(file_name).stem}_pred.png"
                                    )
                                )
                                pred_mask.save(
                                    os.path.join(
                                        save_dir,
                                        f"{Path(file_name).stem}_predmask.png",
                                    )
                                )
                                lq_image_pil.save(
                                    os.path.join(
                                        save_dir, f"{Path(file_name).stem}_input.png"
                                    )
                                )
                                del x_tgt_pred, lq_image_pil,pred_mask
                                gc.collect()
                                torch.cuda.empty_cache()
                            # accelerator.log(logs, step=global_step)
                        gc.collect()
                        torch.cuda.empty_cache()
                        print("Eval Done")
                # accelerator.wait_for_everyone()

    if global_step != 0 and accelerator.sync_gradients:
        outf = os.path.join(args.output_dir, "checkpoints", f"model-{global_step}.pth")
        torch.save({"state_dict_dm":accelerator.unwrap_model(net_tsr).eval().IDM_Unet.state_dict()},outf)
        outf = os.path.join(args.output_dir, "checkpoints", f"encoder-{global_step}.pth")
        torch.save(accelerator.unwrap_model(net_tsr).eval().transformer_encoder.state_dict(),outf.replace('.pkl','.pth')) 


if __name__ == "__main__":
    args = parse_args_training()
    main(args)
