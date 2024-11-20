import os
from dataclasses import dataclass, field

import torch
import torchvision.transforms.functional

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from threestudio.evaluation import evaluate
from pathlib import Path
import numpy as np
from PIL import Image
import wandb
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image


@threestudio.register("imagedream-system")
class ImageDreamSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False
        update_steps_per_target_update: int = 130
        test_start_from_center: bool=False

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        if self.cfg.guidance_type != "":
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
            self.guidance.requires_grad_(False)
        else:
            self.guidance = None
        self.prompt_processor = None

    def on_load_checkpoint(self, checkpoint):
        if self.guidance is None:
            # drop all stored weights starting with "guidance."
            for k in list(checkpoint["state_dict"]):
                if k.startswith("guidance."):
                    checkpoint["state_dict"].pop(k)
        else:
            for k in list(checkpoint["state_dict"].keys()):
                if k.startswith("guidance."):
                    return
            guidance_state_dict = {
                "guidance." + k: v for (k, v) in self.guidance.state_dict().items()
            }
            checkpoint["state_dict"] = {**checkpoint["state_dict"], **guidance_state_dict}
        return

    def on_save_checkpoint(self, checkpoint):
        for k in list(checkpoint["state_dict"].keys()):
            if k.startswith("guidance."):
                checkpoint["state_dict"].pop(k)
        return

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.renderer(**batch)

    def on_train_batch_start(self, batch, batch_idx, unused=0):
        super().on_train_batch_start(batch, batch_idx, unused)

        # updating target views
        train_set = self.trainer.train_dataloader.dataset
        target_cam_idcs = train_set.target_cam_idcs
        targets_complete = ~torch.any(torch.all(self.guidance.target_imgs.view(400, -1) == 0, dim=-1))
        if (self.true_global_step % self.cfg.update_steps_per_target_update == 0 or not targets_complete) and len(self.guidance.denoising_timesteps) > 0:
            print(f"UPDATING TARGETS AT STEP {self.true_global_step}")
            timestep = self.guidance.denoising_timesteps.pop(0)
            nsteps = None

            # last ddim steps at once if specified accordingly
            if self.guidance.cfg.last_ddim_steps_at_once > 0 and len(self.guidance.denoising_timesteps) == self.guidance.cfg.last_ddim_steps_at_once - 1:
                nsteps = self.guidance.cfg.last_ddim_steps_at_once
                self.guidance.denoising_timesteps = []

            with torch.no_grad():
                # rendering with bs 1 to avoid oom and because otherwise errors occured in tinycudann backend (tinycudann.modules.Module.forward()) when inferring in train mode
                target_nerf_inputs = dict()
                for aidx in target_cam_idcs:
                    target_batch = train_set.get_data_batch([aidx])
                    out = self(target_batch)
                    # out = dict(comp_rgb=torch.zeros(1, 512, 512, 3, device=self.device, dtype=torch.float32))
                    target_nerf_inputs[aidx] = out["comp_rgb"][0]
                    self.guidance.update_targets(out["comp_rgb"],
                                                 self.prompt_utils,
                                                 timestep=timestep,
                                                 nsteps=nsteps,
                                                 **target_batch)
                    # # visualization
                    # fig, axes = plt.subplots(ncols=2)
                    # axes[0].imshow(target_nerf_inputs[aidx].cpu().float())
                    # axes[1].imshow(new_target_imgs[aidx].permute(1, 2, 0).cpu().float())
                    # plt.show()
            if self.true_global_step % (self.cfg.update_steps_per_target_update * 10) == 0:  # only log every tenth target update
                target_cam_idcs = [target_cam_idcs[i] for i in np.round(np.linspace(0, len(target_cam_idcs) - 1, 10)).astype(int)]
                new_target_wandbimgs = [
                    wandb.Image(torchvision.transforms.functional.to_pil_image(self.guidance.target_imgs[aidx].cpu().float()), caption=f"{self.true_global_step}-{aidx}", file_type="jpg")
                    for aidx in target_cam_idcs]
                target_nerf_inputs_wandbimgs = [
                    wandb.Image(torchvision.transforms.functional.to_pil_image(target_nerf_inputs[aidx].cpu().float().permute(2, 0, 1)), caption=f"{self.true_global_step}-{aidx}", file_type="jpg")
                    for aidx in target_cam_idcs]
                self.logger.experiment.log({
                    f"val/target_imgs": new_target_wandbimgs,
                    f"val/target_nerf_inputs": target_nerf_inputs_wandbimgs,
                    "trainer/global_step": self.true_global_step}, step=self.true_global_step)

    def on_fit_start(self) -> None:
        super().on_fit_start()

        prompt = self.trainer.datamodule.train_dataset.get_caption()
        negative_prompt = ""
        self.cfg.prompt_processor.prompt = prompt
        self.cfg.prompt_processor.negative_prompt = negative_prompt
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()

    def training_step(self, batch, batch_idx):
        out = self(batch)
        guidance_out = self.guidance(out["comp_rgb"],
                                     self.prompt_utils,
                                     comp_rgb_bg=out["comp_rgb_bg"],
                                     **batch)

        loss = 0.0
        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                              out["weights"].detach()
                              * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                          ).sum() / (out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        if self.C(self.cfg.loss.lambda_sparsity) > 0:
            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        if self.C(self.cfg.loss.lambda_opaque) > 0:
            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
        # helps reduce floaters and produce solid geometry
        if self.C(self.cfg.loss.lambda_z_variance) > 0:
            loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
            self.log("train/loss_z_variance", loss_z_variance)
            loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

        if (
            hasattr(self.cfg.loss, "lambda_eikonal")
            and self.C(self.cfg.loss.lambda_eikonal) > 0
        ):
            loss_eikonal = (
                (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
            ).mean()
            self.log("train/loss_eikonal", loss_eikonal)
            loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        save_dir_name = f"it{self.true_global_step:06d}"
        save_dir = self.get_save_path(save_dir_name)
        os.makedirs(save_dir, exist_ok=True)
        self.save_image_grid(
            f"{save_dir_name}/{batch['index'][0]:02d}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            # + (
            #     [
            #         {
            #             "type": "rgb",
            #             "img": out["comp_normal"][0],
            #             "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
            #         }
            #     ]
            #     if "comp_normal" in out
            #     else []
            # )
            # + [
            #     {
            #         "type": "grayscale",
            #         "img": out["opacity"][0, :, :, 0],
            #         "kwargs": {"cmap": None, "data_range": (0, 1)},
            #     },
            # ]
            ,
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        save_dir_name = f"it{self.true_global_step:06d}"
        save_dir = self.get_save_path(save_dir_name)
        log_dict = dict()

        log_img_paths = [p for p in sorted(Path(save_dir).iterdir()) if p.name.endswith(".png")]
        log_img_paths = [log_img_paths[i] for i in np.round(np.linspace(0, len(log_img_paths) - 1, 5)).astype(int)]
        log_imgs = [Image.open(p).crop((0, 0, 1024, 512)) for p in log_img_paths]
        log_dict.update({f"val/img_sweep": [wandb.Image(img, caption=f"{self.true_global_step:06d}-{p.name}", file_type="jpg") for img, p in zip(log_imgs, log_img_paths)]})

        # quantitative evaluation
        with torch.no_grad():
            evaluation_batch = self.trainer.datamodule.val_dataset.get_evaluation_batch()
            out = self(evaluation_batch)
            evaluation_results = evaluate.evaluate_imgs(preds=out["comp_rgb"].permute(0, 3, 1, 2),
                                                        gts=evaluation_batch["imgs"].permute(0, 3, 1, 2),
                                                        masks=evaluation_batch["masks"].permute(0, 3, 1, 2)
                                                        )
            for k in evaluation_results:
                log_dict[f"val/{k}"] = np.mean(evaluation_results[k])

            out_img = to_pil_image(out["comp_rgb"].cpu()[0].permute(2, 0, 1))
            if "comp_normal" in out:
                out_normal = to_pil_image(out["comp_normal"].cpu()[0].permute(2, 0, 1))
            else:
                out_normal = to_pil_image(torch.zeros_like(out["comp_rgb"].cpu()[0].permute(2, 0, 1)))
            gt_img = to_pil_image(evaluation_batch["imgs"].cpu()[0].permute(2, 0, 1))
            log_dict.update({
                f"val/img_pred": wandb.Image(out_img, caption=f"{self.true_global_step:06d}-pred", file_type="jpg"),
                f"val/img_normal": wandb.Image(out_normal, caption=f"{self.true_global_step:06d}-normal", file_type="jpg"),
                f"val/img_gt": wandb.Image(gt_img, caption=f"{self.true_global_step:06d}-gt", file_type="jpg"),
            })
            out_img.save(self.get_save_path(f"{save_dir_name}/it{self.true_global_step:06d}-pred_eval.png"))
            out_normal.save(self.get_save_path(f"{save_dir_name}/it{self.true_global_step:06d}-normal_eval.png"))
            gt_img.save(self.get_save_path(f"{save_dir_name}/it{self.true_global_step:06d}-gt_eval.png"))

        self.save_img_sequence(
            f"{save_dir_name}/sweep",
            save_dir_name,
            "(\d+)\.png",
            save_format="mp4",
            fps=3,
            name="val/sweep_vid",
            step=self.true_global_step,
            caption=save_dir_name,
            forward_backward=True,
        )

        # target visualization
        with torch.no_grad():
            train_set = self.trainer.datamodule.train_dataset
            target_nerf_preds = dict()
            target_cam_idcs = [train_set.target_cam_idcs[i] for i in np.round(np.linspace(0, len(train_set.target_cam_idcs) - 1, 10)).astype(int)]
            # target_cam_idcs = train_set.target_cam_idcs
            for aidx in target_cam_idcs:
                target_batch = train_set.get_data_batch([aidx])
                out = self(target_batch)
                target_nerf_preds[aidx] = out["comp_rgb"][0]
            nerf_imgs_wandb = [wandb.Image(to_pil_image(target_nerf_preds[aidx].cpu().float().permute(2, 0, 1)), caption=f"{self.true_global_step}-{aidx}", file_type="jpg") for aidx in target_cam_idcs]
            target_imgs_wandb = [wandb.Image(to_pil_image(self.guidance.target_imgs[aidx].cpu().float()), caption=f"{self.true_global_step}-{aidx}", file_type="jpg") for aidx in target_cam_idcs]
            log_dict.update({f"val/target_imgs": target_imgs_wandb,
                             "val/nerf_preds": nerf_imgs_wandb})

        log_dict["trainer/global_step"] = self.true_global_step
        self.logger.experiment.log(log_dict, step=self.true_global_step)

    def test_step(self, batch, batch_idx):
        out = self(batch)
        img_name = f"it{self.true_global_step:06d}-test/{batch['index'][0]:03d}.png"
        self.save_image_grid(
            img_name,
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            # + (
            #     [
            #         {
            #             "type": "rgb",
            #             "img": out["comp_normal"][0],
            #             "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
            #         }
            #     ]
            #     if "comp_normal" in out
            #     else []
            # )
            # + [
            #     {
            #         "type": "grayscale",
            #         "img": out["opacity"][0, :, :, 0],
            #         "kwargs": {"cmap": None, "data_range": (0, 1)},
            #     },
            # ],
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step:06d}-test_sweep",
            f"it{self.true_global_step:06d}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test/full_sweep_vid",
            step=self.true_global_step,
        )


def img_idx_to_row_and_col(i, nrows):
    col_idx = i // nrows
    row_idx = i % nrows
    return row_idx, col_idx
