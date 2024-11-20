# Part of this script are derived from the official example script of diffusers
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

# import pydevd_pycharm
# pydevd_pycharm.settrace('10.34.31.195', port=12345, stdoutToServer=True, stderrToServer=True)

import logging
import warnings
import math
import os
import shutil
import torch
import torch.utils.checkpoint
import sys
import diffusers
import transformers
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.tracking import WandBTracker, TensorBoardTracker
from accelerate.utils import set_seed
import numpy as np
from joker.prior.inference import inference_on_dataset
from joker.prior.evaluate import evaluate_folder
from joker.utils import import_obj, parse_config, config2argv
from diffusers import (
    DDPMScheduler,
    UNet2DConditionModel,
)
import glob
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPTokenizer
import time
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
from torchvision.transforms.functional import to_tensor

from joker.prior.models import Joker2DPrior

import matplotlib.pyplot as plt

logger = get_logger(__name__)


def train():
    # reading config
    config = parse_config([sys.argv[1]])
    OmegaConf.resolve(config)

    if config.general.get("debug", False):
        import pydevd_pycharm
        pydevd_pycharm.settrace('10.1.5.14', port=12345, stdoutToServer=True, stderrToServer=True,
                                suspend=False)
    config = config["train_kwargs"]
    accelerator_cfg = OmegaConf.to_container(config.accelerator)
    accelerator_cfg["log_with"] = list(config.logger.init_kwargs.keys())
    accelerator = Accelerator(**accelerator_cfg)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(os.path.join(config.output_dir, "checkpoints"), exist_ok=True)
        # os.system(f"rm {config.output_dir}/code")
        # os.system(f"ln -s {os.getcwd()} {config.output_dir}/code")
        os.system(f"cp {sys.argv[1]} {config.output_dir}/config.yaml")
    accelerator.wait_for_everyone()

    # if resubmit interval is specified and checkpoints exist, always resume from latest checkpoint
    resubmit_interval = config.get("resubmit_interval", -1)

    # Make one log on every process with the configuration for debugging.
    t = time.localtime()
    str_m_d_y_h_m_s = time.strftime("%m-%d-%Y_%H-%M-%S", t)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(config.output_dir, f"{str_m_d_y_h_m_s}.log")
            ),
        ]
        if accelerator.is_main_process
        else [],
    )

    # logging config
    logger.info("#### CONFIG ####")
    for line in str(OmegaConf.to_yaml(config)).split("\n"):
        logger.info(line)
    logger.info("#### ENDOFCONFIG ####")

    logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if config.seed is not None:
        set_seed(config.seed + accelerator.process_index)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="tokenizer",
    )

    model = Joker2DPrior.from_pretrained()

    # freeze all params in the model
    for param in model.parameters():
        param.requires_grad = False
        param.data = param.data.to(torch.float32)

    if config.load_model is not None:
        model.load_state_dict(
            torch.load(config.load_model, map_location="cpu")
        )

    if config.train_unet:
        model.unet.mid_block.requires_grad_(True)
        model.unet.up_blocks.requires_grad_(True)

    if config.train_refunet:
        for p in model.ref_unet.get_trainable_refnet_parameters():
            p.requires_grad_(True)

    if config.train_controlnet:
        model.controlnet.requires_grad_(True)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.scale_lr:
        config.learning_rate = (
                config.learning_rate
                * config.accelerator.gradient_accumulation_steps
                * config.train_batch_size
                * accelerator.num_processes
        )

    optimizer_cls = torch.optim.AdamW

    ref_unet_params = list([p for p in model.ref_unet.parameters() if p.requires_grad])
    unet_params = list([p for p in model.unet.parameters() if p.requires_grad])
    other_params = list(
        [p for n, p in model.named_parameters() if p.requires_grad and "unet" not in n]
    )
    parameters = ref_unet_params + unet_params + other_params

    optimizer = optimizer_cls(
        [
            {"params": ref_unet_params + unet_params, "lr": config.learning_rate * config.unet_lr_scale},
            {"params": other_params, "lr": config.learning_rate},
        ],
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

    train_dataset_class = import_obj(config.train_dataset._target)
    train_dataset = train_dataset_class(tokenizer=tokenizer, split="train", **config.train_dataset._kwargs)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.train_batch_size,
        num_workers=config.dataloader_num_workers,
    )

    # validation datasets
    named_val_datasets = dict()
    for val_ds_cfg in config.val_dataset:
        val_ds_name = val_ds_cfg["name"]
        val_ds_class = import_obj(val_ds_cfg._target)
        val_ds = val_ds_class(tokenizer=tokenizer, split="val", **val_ds_cfg._kwargs)
        named_val_datasets[val_ds_name] = val_ds

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.accelerator.gradient_accumulation_steps
    )
    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * config.accelerator.gradient_accumulation_steps,
        num_training_steps=config.max_train_steps * config.accelerator.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.accelerator.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    config.num_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    tracker_cfg = OmegaConf.to_container(config.logger)
    if accelerator.is_main_process:
        accelerator.init_trackers(**tracker_cfg)

    # Train!
    if accelerator.is_local_main_process:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"Trainable parameter: {name} with shape {param.shape}")

    total_batch_size = (
            config.train_batch_size
            * accelerator.num_processes
            * config.accelerator.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {config.accelerator.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # resume from latest if resubmit_interval and checkpoints exist
    if resubmit_interval > 0 and len(
            [d for d in os.listdir(os.path.join(config.output_dir, "checkpoints")) if d.startswith("checkpoint")]) > 0:
        config.resume_from_checkpoint = "latest"

    # Potentially load in the weights and states from a previous save
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint != "latest":
            path = config.resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(os.path.join(config.output_dir, "checkpoints"))
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = os.path.join(config.output_dir, "checkpoints", dirs[-1]) if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            config.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(path)
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * config.accelerator.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                    num_update_steps_per_epoch * config.accelerator.gradient_accumulation_steps
            )

            # move all the state to the correct device
            model.to(accelerator.device)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, config.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )

    def evaluate(save_path):
        ### Quantitative & Qualitative Evaluation
        eval_results = dict()
        for val_set_name, val_set in named_val_datasets.items():
            logger.info(f"Evaluating {val_set_name}")
            eval_dir = os.path.join(save_path, "evaluation", val_set_name)
            inference_dir = os.path.join(eval_dir, "inference")
            inference_on_dataset(model=accelerator.unwrap_model(model), dataset=val_set, output_dir=inference_dir, accelerator=accelerator, neval=config.neval)
            eval_results[val_set_name] = evaluate_folder(prediction_folder=inference_dir, use_mask=True, eval_resolution=512)

        ### LOGGING
        if accelerator.is_main_process:
            # quantitative logging
            avg_eval_results = dict()
            for k in list(eval_results.values())[0]:
                scores = np.array(
                    [val_set_results[k][0] for val_set_name, val_set_results in eval_results.items()
                     if named_val_datasets[val_set_name].use_for_avg_quant])
                stds = np.array([val_set_results[k][1] for val_set_name, val_set_results in eval_results.items()
                                 if named_val_datasets[val_set_name].use_for_avg_quant])
                avg_score = np.mean(scores)
                avg_std = np.sqrt(np.sum(stds ** 2)) / len(stds)
                avg_eval_results[k] = [avg_score, avg_std]
            eval_results["AVERAGE"] = avg_eval_results
            outfpath = os.path.join(save_path, "evaluation", "score.txt")
            avg_score_str = ""
            for k in avg_eval_results:
                avg_score_str += f"{k}: {avg_eval_results[k][0]} +- {avg_eval_results[k][1]}\n"
            with open(outfpath, "w") as f:
                f.write(avg_score_str)
            logger.info(f"\nAverage Scores: \n{avg_score_str}")

            accelerator.log(dict([(f"val/{val_set_name}-{metric_name}", metric_score[0])
                                  for val_set_name, val_set_results in eval_results.items()
                                  for metric_name, metric_score in val_set_results.items()]),
                            step=global_step)

            # qualitative logging
            for val_set_name in named_val_datasets:
                pred_img_paths = [Path(f) for f in sorted(
                    glob.glob(os.path.join(save_path, "evaluation", val_set_name, "inference", "*_pred.png")))]
                pred_img_paths = np.random.RandomState(0).choice(pred_img_paths, size=min(5, len(pred_img_paths)),
                                                                 replace=False)
                gt_img_paths = [f.parent / (f.name.split("_")[0] + "_gt.png") for f in pred_img_paths]
                ref_img_paths = [f.parent / (f.name.split("_")[0] + "_ref.png") for f in pred_img_paths]

                pred_imgs = [Image.open(p) for p in pred_img_paths]
                gt_imgs = [Image.open(p) for p in gt_img_paths]
                ref_imgs = [Image.open(p) for p in ref_img_paths]

                for tracker in accelerator.trackers:
                    if isinstance(tracker, WandBTracker):
                        tracker.log({f"val/{val_set_name}-img_pred": [
                            wandb.Image(img, caption=f"{global_step}-{f.name}", file_type="jpg") for img, f in
                            zip(pred_imgs, pred_img_paths)]}, step=global_step)
                        tracker.log({f"val/{val_set_name}-img_gt": [
                            wandb.Image(img, caption=f"{global_step}-{f.name}", file_type="jpg") for img, f in
                            zip(gt_imgs, gt_img_paths)]}, step=global_step)
                        tracker.log({f"val/{val_set_name}-img_ref": [
                            wandb.Image(img, caption=f"{global_step}-{f.name}", file_type="jpg") for img, f in
                            zip(ref_imgs, ref_img_paths)]}, step=global_step)
                    elif isinstance(tracker, TensorBoardTracker):
                        tracker.log_images({f"val/{val_set_name}-img_pred": torch.stack([to_tensor(img) for img in pred_imgs])}, step=global_step)
                        tracker.log_images({f"val/{val_set_name}-img_gt": torch.stack([to_tensor(img) for img in gt_imgs])}, step=global_step)
                        tracker.log_images({f"val/{val_set_name}-img_ref": torch.stack([to_tensor(img) for img in ref_imgs])}, step=global_step)
                    else:
                        warnings.warn(f"Image logging not supported for tracker of type {type(tracker)}")
        accelerator.wait_for_everyone()

    for epoch in range(first_epoch, config.num_train_epochs):
        model.train()
        train_loss = 0.0

        if config.resume_from_checkpoint and epoch == first_epoch and resume_step is not None:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            progress_bar.set_description("Global step: {}".format(global_step))

            with accelerator.accumulate(model), torch.backends.cuda.sdp_kernel(
                    enable_flash=not config.disable_flashattention
            ):
                return_dict = model(batch, noise_scheduler)
                loss = return_dict["loss"]

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(config.train_batch_size)).mean()
                train_loss += avg_loss.item() / config.accelerator.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(parameters, config.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:  # true
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {
                        "train_loss": train_loss,
                        "global_step": global_step
                    },
                    step=global_step,
                )
                train_loss = 0.0

                save_path = os.path.join(config.output_dir, "checkpoints", f"checkpoint-{global_step:06d}")

                if global_step % config.checkpointing_steps == 0:
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                    model.eval()
                    logger.info("Evaluating Model ...")
                    evaluate(save_path)
                    model.train()

                    if accelerator.is_local_main_process:
                        if config.keep_only_last_checkpoint:
                            # Remove all other checkpoints
                            for file in os.listdir(os.path.join(config.output_dir, "checkpoints")):
                                if file.startswith(
                                        "checkpoint"
                                ) and file != os.path.basename(save_path):
                                    ckpt_num = int(file.split("-")[1])
                                    if (
                                            config.keep_interval is None
                                            or ckpt_num % config.keep_interval != 0
                                    ):
                                        logger.info(f"Removing {file}")
                                        shutil.rmtree(os.path.join(config.output_dir, "checkpoints", file))

                # resubmitting job and exit if specified
                if resubmit_interval > 0 and global_step % resubmit_interval == 0:
                    command = f"condor/submit.py 25 {config.output_dir}/config.yaml"
                    if accelerator.is_main_process:
                        os.system(command)
                    accelerator.wait_for_everyone()
                    exit()

            logs = {
                "lr": lr_scheduler.get_last_lr()[0],
            }

            progress_bar.set_postfix(**logs)

            if global_step >= config.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()
    logger.info("SUCCESSFULLY FINISHED TRAINING")


if __name__ == "__main__":
    # assumes to be called like: python train.py PATH_TO_TRAIN_CONFIG.yaml
    train()
